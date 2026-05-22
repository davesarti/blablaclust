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

from src.engine.cluster_operations import (
    merge_clusters,
    move_points,
    rename_cluster,
    split_cluster,
)
from src.engine.cluster_naming import name_clusters
from src.models import DataPoint, SoftAssignment


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
        ValueError: propagated from merge/split/rename when the LLM emits an
            invalid op (unknown cluster_id, already-dissolved cluster, stale
            turn_number).  The caller is expected to convert this into a 4xx
            HTTP response so the failure is visible to the oracle — silent
            skipping is forbidden because it hides real bugs in the prompt or
            the LLM's output.
        KeyError: when an operation dict is missing a required field
            (e.g. cluster_ids on a merge).  Same rationale.
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
            # Flush so the next operation in this same turn sees the updated
            # snapshot rows and dissolved cluster state.  The session uses
            # autoflush=False, so without this the second split/merge would
            # read the pre-operation DB state and carry forward wrong clusters.
            db.flush()
            current_turn += 1

        elif op_type == "split":
            # Writes a full soft-assignment snapshot → consumes a turn_number.
            new_clusters = split_cluster(
                cluster_id=op["cluster_id"],
                session_id=session_id,
                turn_number=current_turn,
                db=db,
            )
            # Same flush reason as merge above.
            db.flush()

            # Ask the LLM to name the two new sub-clusters from their
            # representative points, replacing the generic "part 1/2" labels.
            new_cluster_ids = [c.id for c in new_clusters]
            sub_assignments = (
                db.query(SoftAssignment)
                .filter(
                    SoftAssignment.cluster_id.in_(new_cluster_ids),
                    SoftAssignment.turn_number == current_turn,
                )
                .all()
            )
            point_ids = list({a.data_point_id for a in sub_assignments})
            sub_points = db.query(DataPoint).filter(DataPoint.id.in_(point_ids)).all()
            name_clusters(new_clusters, sub_assignments, sub_points)

            current_turn += 1

        elif op_type == "move":
            # Writes a full soft-assignment snapshot → consumes a turn_number.
            move_points(
                point_ids=op["point_ids"],
                target_cluster_id=op["target_cluster_id"],
                session_id=session_id,
                turn_number=current_turn,
                db=db,
            )
            # Same flush reason as merge/split above.
            db.flush()
            current_turn += 1

        elif op_type == "rename":
            # Only updates name/description — no new snapshot, no turn_number needed.
            rename_cluster(
                cluster_id=op["cluster_id"],
                new_name=op.get("new_name", ""),
                new_description=op.get("new_description", ""),
                db=db,
            )

        # Unknown op_type values are skipped intentionally so a future protocol
        # extension does not crash older clients.  Missing op_type, however, is
        # treated as a structural error and falls through to the KeyError
        # raised by the dispatch above (or by op["..."] field accesses).

    return current_turn
