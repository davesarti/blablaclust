"""Dispatch Claude's structured operations to the actual clustering functions.

After f_output returns a list of operations (merge, split, rename), this
function routes each one to the corresponding P2 function that executes the
real embedding-based reassignment.

Transaction model (mirrors cluster_operations.py)
--------------------------------------------------
Does NOT call db.commit() — the caller owns the transaction so a turn that
applies several operations stays atomic. On any error the caller's session is
still clean and can be rolled back.

turn_number handling
--------------------
P2's merge and split functions require turn_number to be strictly greater than
the latest existing snapshot. If a single oracle turn triggers multiple
operations that write new snapshots (e.g. merge + split), each needs its own
incrementing turn_number. This function starts at the given turn_number and
increments by 1 for each snapshot-writing operation (merge, split). Rename
never writes a new snapshot so it never consumes a turn_number.

The final turn_number used is returned so the caller can record it.
"""

from sqlalchemy.orm import Session

from src.engine.cluster_operations import merge_clusters, rename_cluster, split_cluster


def f_apply_operations(
    operations: list[dict],
    session_id: str,
    turn_number: int,
    db: Session,
) -> int:
    """Route each operation from Claude's response to the right clustering function.

    Args:
        operations: List of operation dicts from Claude's JSON response.
                    Each must have a "type" key: "merge", "split", or "rename".
                    Unknown types are silently skipped.
        session_id: The session these operations belong to.
        turn_number: Starting turn number for snapshot-writing operations.
                     Must be strictly greater than the latest existing snapshot.
        db: SQLAlchemy session. No commit is made here — caller's responsibility.

    Returns:
        The next available turn_number after all operations. If no snapshot-writing
        operations ran (empty list, only renames, or only unknown types) this
        equals the input turn_number unchanged.

    Raises:
        ValueError: propagated from merge/split/rename when an operation is
                    invalid (unknown cluster, already dissolved, stale turn_number).
                    The DB session is untouched so the caller can roll back.
        KeyError: if a required field is missing from an operation dict.
    """
    current_turn = turn_number

    for op in operations:
        op_type = op.get("type")

        if op_type == "merge":
            # Writes a full soft-assignment snapshot → consumes a turn_number.
            merge_clusters(
                cluster_ids=op["cluster_ids"],
                session_id=session_id,
                turn_number=current_turn,
                db=db,
            )
            current_turn += 1

        elif op_type == "split":
            # Writes a full soft-assignment snapshot → consumes a turn_number.
            split_cluster(
                cluster_id=op["cluster_id"],
                session_id=session_id,
                turn_number=current_turn,
                db=db,
            )
            current_turn += 1

        elif op_type == "rename":
            # Only updates name/description — no new snapshot, no turn_number needed.
            rename_cluster(
                cluster_id=op["cluster_id"],
                new_name=op.get("new_name", ""),
                new_description=op.get("new_description", ""),
                db=db,
            )

        # Unknown types are silently skipped — Claude may return op types we
        # don't support yet and that must never crash a turn.

    return current_turn
