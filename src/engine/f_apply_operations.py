"""Dispatch Claude's structured operations to the actual clustering functions.

After f_output returns a list of operations (merge, split, move, rename), this
function routes each one to the corresponding P2 function. The operations all
mutate a :class:`~src.engine.turn_builder.TurnBuilder` rather than the DB
directly; the caller (router) commits the builder once at the end of the
conversation turn so that every snapshot the engine writes lines up with the
conv turn it came from.

Transaction model
-----------------
No DB writes happen here; the builder stages everything in memory. On any
error the caller can simply drop the builder and rollback its own DB session;
nothing was persisted.
"""

import difflib

from src.engine.cluster_operations import (
    auto_name_cluster,
    batch_move_points,
    merge_clusters,
    rename_cluster,
    split_cluster,
)
from src.engine.turn_builder import TurnBuilder
from src.logger import log


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


def _normalize_cluster_ids(operations: list[dict], builder: TurnBuilder) -> None:
    """Repair transcription errors in cluster ids in-place before dispatch.

    Resolves ``cluster_ids`` (merge), ``cluster_id`` (split/rename) and
    ``target_cluster_id`` (move) against the clusters that are currently
    considered active by the builder. Point ids are deliberately left alone —
    they are not drawn from a small known set and the oracle names them
    explicitly.
    """
    active_ids = [c.id for c in builder.active_clusters()]
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
    builder: TurnBuilder,
    axis_hint: str | None = None,
) -> None:
    """Route each operation from Claude's response to the right clustering function.

    Args:
        operations: List of operation dicts from Claude's JSON response.
                    Each must have a "type" key: "merge", "split", "move",
                    or "rename". Unknown types are silently skipped.
        builder: The conversation turn's in-memory staging area. All ops
                 mutate the builder; the caller commits once at the end.
        axis_hint: When provided, forwarded to merge/split so newly-named
                   clusters reflect the session's semantic axis.

    Raises:
        ValueError: propagated from merge/split/move/rename when the LLM emits
            an invalid op (unknown cluster_id, already-dissolved cluster).
            The caller is expected to convert this into a 4xx HTTP response so
            the failure is visible to the oracle — silent skipping is forbidden
            because it hides real bugs in the prompt or the LLM's output.
        KeyError: when an operation dict is missing a required field
            (e.g. cluster_ids on a merge). Same rationale.
    """
    _normalize_cluster_ids(operations, builder)

    for op in operations:
        op_type = op.get("type")

        if op_type == "merge":
            inline_new_name = (op.get("new_name") or "").strip()
            inline_new_desc = (op.get("new_description") or "").strip()

            new_cluster = merge_clusters(
                cluster_ids=op["cluster_ids"],
                builder=builder,
                axis_hint=axis_hint,
            )

            if inline_new_name or inline_new_desc:
                # Preserve whichever side the oracle did NOT specify so we
                # don't blow away the placeholder/auto-name we kept above.
                rename_cluster(
                    cluster_id=new_cluster.id,
                    new_name=inline_new_name or (new_cluster.name or ""),
                    new_description=inline_new_desc or (new_cluster.description or ""),
                    builder=builder,
                )

        elif op_type == "split":
            inline_new_names = [
                (n or "").strip()
                for n in (op.get("new_names") or [])
            ]
            k = int(op.get("k", 2))

            new_clusters = split_cluster(
                cluster_id=op["cluster_id"],
                builder=builder,
                k=k,
                axis_hint=axis_hint,
            )

            # K-means returns children in no oracle-meaningful order, so this
            # mapping is best-effort: name[i] -> child[i].
            for child, name in zip(new_clusters, inline_new_names):
                if not name:
                    continue
                rename_cluster(
                    cluster_id=child.id,
                    new_name=name,
                    new_description=child.description or "",
                    builder=builder,
                )

        elif op_type == "move":
            target = op["target_cluster_id"]
            batch_move_points(
                [(pid, target) for pid in op["point_ids"]],
                builder=builder,
            )

        elif op_type == "rename":
            inline_new_name = (op.get("new_name") or "").strip()
            inline_new_desc = (op.get("new_description") or "").strip()
            if not inline_new_name and not inline_new_desc:
                # Bare rename ("rename cluster X") — regenerate name and
                # description from the cluster's current contents.
                auto_name_cluster(op["cluster_id"], builder, axis_hint=axis_hint)
            else:
                existing = builder.get_cluster(op["cluster_id"])
                rename_cluster(
                    cluster_id=op["cluster_id"],
                    new_name=inline_new_name or (existing.name if existing else ""),
                    new_description=inline_new_desc or (existing.description if existing else ""),
                    builder=builder,
                )

        # Unknown op_type values are skipped intentionally so a future protocol
        # extension does not crash older clients. Missing op_type, however, is
        # treated as a structural error and falls through to the KeyError
        # raised by op["..."] field accesses above.
