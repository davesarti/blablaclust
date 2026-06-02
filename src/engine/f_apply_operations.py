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

import difflib

from sqlalchemy.orm import Session

from src.engine.cluster_operations import (
    merge_clusters,
    move_points,
    rename_cluster,
    split_cluster,
)
from src.logger import log
from src.models import Cluster as DbCluster


def _resolve_cluster_id(raw_id: str, active_ids: list[str], active_set: set[str]) -> str:
    """Map an LLM-emitted cluster id onto a real active cluster id.

    The prompt asks the LLM to copy a 36-char UUID verbatim; in practice models
    occasionally mistype a single character (e.g. ``…99a5305bc071`` →
    ``…99a5303bc071``), which would otherwise fail downstream with an opaque
    "clusters not found". Active session UUIDs are random, so two real ids are
    never close — a unique near-match at high similarity is almost certainly the
    intended cluster with a transcription error. Correct it; if there is no
    match (or it is ambiguous), leave the id untouched so the op still fails
    loudly rather than silently mutating into the wrong cluster.
    """
    if not isinstance(raw_id, str) or raw_id in active_set:
        return raw_id
    matches = difflib.get_close_matches(raw_id, active_ids, n=2, cutoff=0.8)
    if not matches:
        return raw_id
    best = matches[0]
    r0 = difflib.SequenceMatcher(None, raw_id, best).ratio()
    if len(matches) == 1:
        chosen = best if r0 >= 0.8 else raw_id
    else:
        r1 = difflib.SequenceMatcher(None, raw_id, matches[1]).ratio()
        # Only accept when the top match is both strong and clearly unique.
        chosen = best if (r0 >= 0.9 and r0 - r1 >= 0.1) else raw_id
    if chosen != raw_id:
        log.warning(
            "f_apply_operations: corrected mistyped cluster_id %s -> %s "
            "(similarity %.3f)", raw_id, chosen, r0
        )
    return chosen


def _normalize_cluster_ids(operations: list[dict], session_id: str, db: Session) -> None:
    """Repair transcription errors in cluster ids in-place before dispatch.

    Resolves ``cluster_ids`` (merge), ``cluster_id`` (split/rename) and
    ``target_cluster_id`` (move) against the clusters that are currently active
    in the session. Point ids are deliberately left alone — they are not drawn
    from a small known set and the oracle names them explicitly.
    """
    active_ids = [
        c.id
        for c in db.query(DbCluster.id)
        .filter(DbCluster.session_id == session_id, DbCluster.dissolved_at_turn.is_(None))
        .all()
    ]
    if not active_ids:
        return
    active_set = set(active_ids)
    for op in operations:
        if isinstance(op.get("cluster_ids"), list):
            op["cluster_ids"] = [
                _resolve_cluster_id(cid, active_ids, active_set)
                for cid in op["cluster_ids"]
            ]
        if op.get("cluster_id") is not None:
            op["cluster_id"] = _resolve_cluster_id(op["cluster_id"], active_ids, active_set)
        if op.get("target_cluster_id") is not None:
            op["target_cluster_id"] = _resolve_cluster_id(
                op["target_cluster_id"], active_ids, active_set
            )


def f_apply_operations(
    operations: list[dict],
    session_id: str,
    turn_number: int,
    db: Session,
    axis_hint: str | None = None,
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
        axis_hint: When provided, forwarded to merge/split so newly-named
                   clusters reflect the session's semantic axis.

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

    # Repair any single-character UUID transcription slips the LLM made before
    # the ids reach the clustering functions (which fail hard on unknown ids).
    _normalize_cluster_ids(operations, session_id, db)

    for op in operations:
        op_type = op.get("type")

        if op_type == "merge":
            # The prompt lets the LLM put an inline new_name on a merge op so
            # the oracle can say "merge these and call it X" in a single turn —
            # without it the rename would need a forward-reference to a UUID
            # that does not exist yet (forbidden, see #28).
            inline_new_name = (op.get("new_name") or "").strip()
            inline_new_desc = (op.get("new_description") or "").strip()

            # Always let auto-name run — it produces BOTH name and description
            # from the merged content, and we want the description even when
            # the oracle is overriding the name (otherwise the cluster ends up
            # with the oracle's chosen name but an empty description).
            new_cluster = merge_clusters(
                cluster_ids=op["cluster_ids"],
                session_id=session_id,
                turn_number=current_turn,
                db=db,
                axis_hint=axis_hint,
            )
            # Flush so the next operation in this same turn sees the updated
            # snapshot rows and dissolved cluster state.  The session uses
            # autoflush=False, so without this the second split/merge would
            # read the pre-operation DB state and carry forward wrong clusters.
            db.flush()

            if inline_new_name or inline_new_desc:
                # Preserve whichever side the oracle did NOT specify so we
                # don't blow away the placeholder/auto-name we kept above.
                rename_cluster(
                    cluster_id=new_cluster.id,
                    new_name=inline_new_name or (new_cluster.name or ""),
                    new_description=inline_new_desc or (new_cluster.description or ""),
                    db=db,
                )
            current_turn += 1

        elif op_type == "split":
            # Optional inline child names — same single-turn-naming pattern as
            # the merge branch.  Without this, the LLM has to defer naming the
            # children to the next turn because it cannot forward-reference
            # not-yet-existing UUIDs (#28 rule).
            inline_new_names = [
                (n or "").strip()
                for n in (op.get("new_names") or [])
            ]
            k = int(op.get("k", 2))

            # Always let auto-name run — it generates BOTH name and description
            # per child from the actual content. We override only the name(s)
            # below, so each child keeps a content-appropriate description even
            # when the oracle supplied an explicit name.
            new_clusters = split_cluster(
                cluster_id=op["cluster_id"],
                session_id=session_id,
                turn_number=current_turn,
                db=db,
                k=k,
                axis_hint=axis_hint,
            )
            # Same flush reason as merge above.
            db.flush()

            # K-means returns children in no oracle-meaningful order, so this
            # mapping is best-effort: name[i] -> child[i].  Acceptable because
            # the oracle can rename a misaligned child in the next turn.
            for child, name in zip(new_clusters, inline_new_names):
                if not name:
                    continue
                rename_cluster(
                    cluster_id=child.id,
                    new_name=name,
                    new_description=child.description or "",
                    db=db,
                )
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
            # The LLM usually emits only new_name on a rename op; without this
            # lookup, rename_cluster would clobber a meaningful auto-generated
            # description with the empty default. Preserve whichever side the
            # oracle did NOT explicitly set (same pattern as the inline rename
            # branches on merge/split).
            inline_new_name = (op.get("new_name") or "").strip()
            inline_new_desc = (op.get("new_description") or "").strip()
            existing = (
                db.query(DbCluster)
                .filter(DbCluster.id == op["cluster_id"])
                .first()
            )
            rename_cluster(
                cluster_id=op["cluster_id"],
                new_name=inline_new_name or (existing.name if existing else ""),
                new_description=inline_new_desc or (existing.description if existing else ""),
                db=db,
            )

        # Unknown op_type values are skipped intentionally so a future protocol
        # extension does not crash older clients.  Missing op_type, however, is
        # treated as a structural error and falls through to the KeyError
        # raised by the dispatch above (or by op["..."] field accesses).

    return current_turn
