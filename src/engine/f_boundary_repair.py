"""Post-operation boundary repair: LLM-guided reassignment of uncertain points.

After a merge or split, the geometry-only k-means may have placed boundary
points in the wrong cluster relative to the oracle's intent. This module
samples the N most uncertain points from each affected cluster (smallest
margin between their top-2 soft-assignment probabilities), asks the LLM
whether they belong where they are, and returns move operations for any
misplaced ones.

One LLM call per turn regardless of how many merge/split ops ran.
Out-of-band from the oracle — applied transparently before the turn is
finalised.
"""

from __future__ import annotations

import json

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.harness import (
    call_llm,
    estimate_cost_usd,
    hash_prompt,
    loads_llm_json,
    render_prompt,
)
from src.logger import log, log_llm_call
from src.models import Cluster as DbCluster, DataPoint
from src.models import SoftAssignment as DbSoftAssignment

BOUNDARY_POINTS_PER_CLUSTER = 10


def _boundary_points(
    affected_ids: list[str],
    session_id: str,
    db: Session,
) -> dict[str, list[dict]]:
    """Return the N most uncertain hard-assigned points per affected cluster.

    Uncertainty = smallest margin between top-2 soft-assignment probabilities.
    Returns {cluster_id: [{point_id, text, margin}, ...]}.
    """
    active_ids = [
        row.id
        for row in db.query(DbCluster.id)
        .filter(DbCluster.session_id == session_id, DbCluster.dissolved_at_turn.is_(None))
        .all()
    ]
    if not active_ids:
        return {}

    latest_turn = (
        db.query(func.max(DbSoftAssignment.turn_number))
        .filter(DbSoftAssignment.cluster_id.in_(active_ids))
        .scalar()
    )
    if latest_turn is None:
        return {}

    assignments = (
        db.query(DbSoftAssignment)
        .filter(
            DbSoftAssignment.cluster_id.in_(active_ids),
            DbSoftAssignment.turn_number == latest_turn,
        )
        .all()
    )

    # Group by point: {point_id: {cluster_id: prob}}
    by_point: dict[str, dict[str, float]] = {}
    for a in assignments:
        by_point.setdefault(a.data_point_id, {})[a.cluster_id] = a.probability

    affected_set = set(affected_ids)
    candidates: dict[str, list[tuple[str, float]]] = {cid: [] for cid in affected_ids}

    for point_id, dist in by_point.items():
        hard = max(dist, key=dist.get)
        if hard not in affected_set:
            continue
        probs = sorted(dist.values(), reverse=True)
        margin = probs[0] - (probs[1] if len(probs) > 1 else 0.0)
        candidates[hard].append((point_id, margin))

    all_point_ids: list[str] = []
    top_n: dict[str, list[tuple[str, float]]] = {}
    for cid in affected_ids:
        ranked = sorted(candidates[cid], key=lambda x: x[1])[:BOUNDARY_POINTS_PER_CLUSTER]
        top_n[cid] = ranked
        all_point_ids.extend(pid for pid, _ in ranked)

    if not all_point_ids:
        return {}

    text_by_id: dict[str, str] = {}
    for dp in db.query(DataPoint).filter(DataPoint.id.in_(all_point_ids)).all():
        text_by_id[dp.id] = dp.text or ""

    result: dict[str, list[dict]] = {}
    for cid, pts in top_n.items():
        result[cid] = [
            {"point_id": pid, "text": text_by_id.get(pid, ""), "margin": round(m, 4)}
            for pid, m in pts
            if text_by_id.get(pid, "")
        ]
    return result


def f_boundary_repair(
    session_id: str,
    affected_cluster_ids: list[str],
    oracle_text: str,
    db: Session,
) -> list[dict]:
    """Sample boundary points from affected clusters and ask the LLM to validate.

    Returns a list of {point_id, target_cluster_id} dicts for points that
    should move. Empty list when nothing needs to change or on any error
    (repair is best-effort and must never abort the turn).
    """
    if not affected_cluster_ids:
        return []

    boundary = _boundary_points(affected_cluster_ids, session_id, db)
    if not boundary:
        return []

    # Build a flat list of all candidates so the LLM sees them together.
    all_candidates: list[dict] = []
    for cid, pts in boundary.items():
        for p in pts:
            all_candidates.append({**p, "current_cluster_id": cid})

    if not all_candidates:
        return []

    # Active clusters for context.
    active_clusters = (
        db.query(DbCluster)
        .filter(DbCluster.session_id == session_id, DbCluster.dissolved_at_turn.is_(None))
        .all()
    )
    clusters_block = "\n".join(
        f"[{i}] id={c.id}  name={json.dumps(c.name)}  description: {c.description or ''}"
        for i, c in enumerate(active_clusters, 1)
    )
    points_block = "\n\n".join(
        f"point_id: {p['point_id']}\n"
        f"current_cluster: {p['current_cluster_id']}\n"
        f"margin: {p['margin']}\n"
        f"text: {p['text']}"
        for p in all_candidates
    )

    prompt = render_prompt(
        "f_boundary_repair",
        oracle_text=oracle_text,
        clusters_block=clusters_block,
        points_block=points_block,
    )

    _MAX_ATTEMPTS = 3
    decisions: list = []
    for attempt in range(_MAX_ATTEMPTS):
        try:
            msg = call_llm([{"role": "user", "content": prompt}], system="")
            log_llm_call(
                session_id=session_id,
                prompt_name="f_boundary_repair",
                prompt_hash=hash_prompt("f_boundary_repair"),
                usage=msg.usage,
                cost_usd=estimate_cost_usd(msg.usage),
            )
            decisions = loads_llm_json(msg.text).get("decisions", [])
            break  # success
        except Exception as exc:
            if attempt < _MAX_ATTEMPTS - 1:
                log.warning(
                    "f_boundary_repair: attempt %d/%d failed (%s), retrying.",
                    attempt + 1, _MAX_ATTEMPTS, exc,
                )
            else:
                log.warning(
                    "f_boundary_repair: all %d attempts failed — skipping repair. Error: %s",
                    _MAX_ATTEMPTS, exc,
                )
                return []

    moves: list[dict] = []
    active_id_set = {c.id for c in active_clusters}
    for d in decisions:
        if d.get("keep"):
            continue
        pid = d.get("point_id", "")
        tid = d.get("target_cluster_id", "")
        if pid and tid and tid in active_id_set:
            moves.append({"point_id": pid, "target_cluster_id": tid})

    return moves
