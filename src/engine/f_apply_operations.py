"""Persist cluster and soft-assignment changes from a Claude f_output response.

Called after each turn by the turns router.  The `raw` dict is the parsed
JSON object returned by f_output; this function is the only place that writes
cluster metadata or SoftAssignment rows post-initial-clustering.
"""

import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.models import Cluster as DbCluster, SoftAssignment as DbSoftAssignment


def f_apply_operations(
    raw: dict,
    session_id: str,
    turn_number: int,
    db: Session,
) -> None:
    """Apply Claude's clustering decision to the DB.

    No-op for explain/no_change or when clusters_updated is empty.
    For relabel: updates cluster name/description in place.
    For split/merge/reassign: dissolves old clusters, creates new ones, and
    writes new SoftAssignment rows at turn_number by redistributing the
    dissolved clusters' probability mass equally across the replacement clusters.
    """
    action = raw.get("action", "no_change")
    clusters_updated = raw.get("clusters_updated", [])

    if action in ("explain", "no_change") or not clusters_updated:
        return

    # ------------------------------------------------------------------
    # 1. Resolve IDs: map "new_<label>" slugs to real UUIDs
    # ------------------------------------------------------------------
    id_map: dict[str, str] = {}
    for c in clusters_updated:
        raw_id = c["id"]
        id_map[raw_id] = str(uuid.uuid4()) if raw_id.startswith("new_") else raw_id

    # ------------------------------------------------------------------
    # 2. Load active clusters; determine which are dissolved
    # ------------------------------------------------------------------
    active_clusters = (
        db.query(DbCluster)
        .filter(DbCluster.session_id == session_id, DbCluster.dissolved_at_turn.is_(None))
        .all()
    )
    active_by_id: dict[str, DbCluster] = {c.id: c for c in active_clusters}
    updated_real_ids = set(id_map.values())
    dissolved_ids = set(active_by_id.keys()) - updated_real_ids

    # ------------------------------------------------------------------
    # 3. Apply cluster metadata changes
    # ------------------------------------------------------------------
    for c_data in clusters_updated:
        raw_id = c_data["id"]
        real_id = id_map[raw_id]
        if raw_id.startswith("new_"):
            db.add(DbCluster(
                id=real_id,
                session_id=session_id,
                name=c_data["name"],
                description=c_data["description"],
                created_at_turn=turn_number,
            ))
        else:
            cluster = active_by_id.get(real_id)
            if cluster:
                cluster.name = c_data["name"]
                cluster.description = c_data["description"]

    for cid in dissolved_ids:
        active_by_id[cid].dissolved_at_turn = turn_number

    if not dissolved_ids:
        return  # pure relabel — no point reassignment needed

    # ------------------------------------------------------------------
    # 4. Redistribute SoftAssignments to new clusters at turn_number
    # ------------------------------------------------------------------
    all_active_ids = list(active_by_id.keys())
    latest_turn = (
        db.query(func.max(DbSoftAssignment.turn_number))
        .filter(DbSoftAssignment.cluster_id.in_(all_active_ids))
        .scalar()
    )
    if latest_turn is None:
        return

    existing = (
        db.query(DbSoftAssignment)
        .filter(
            DbSoftAssignment.cluster_id.in_(all_active_ids),
            DbSoftAssignment.turn_number == latest_turn,
        )
        .all()
    )

    # Build per-point probability dict from existing assignments
    by_point: dict[str, dict[str, float]] = {}
    for a in existing:
        by_point.setdefault(a.data_point_id, {})[a.cluster_id] = a.probability

    new_cluster_ids = [id_map[c["id"]] for c in clusters_updated if c["id"].startswith("new_")]
    surviving_ids = updated_real_ids - set(new_cluster_ids)
    n_new = len(new_cluster_ids) or 1

    new_rows: list[DbSoftAssignment] = []
    for point_id, probs in by_point.items():
        # Copy surviving cluster assignments forward unchanged
        for cid in surviving_ids:
            if cid in probs:
                new_rows.append(DbSoftAssignment(
                    data_point_id=point_id,
                    cluster_id=cid,
                    turn_number=turn_number,
                    probability=probs[cid],
                ))
        # Distribute dissolved-cluster probability equally across replacement clusters
        dissolved_prob = sum(probs.get(cid, 0.0) for cid in dissolved_ids)
        if dissolved_prob > 0.0 and new_cluster_ids:
            per_new = dissolved_prob / n_new
            for new_cid in new_cluster_ids:
                new_rows.append(DbSoftAssignment(
                    data_point_id=point_id,
                    cluster_id=new_cid,
                    turn_number=turn_number,
                    probability=per_new,
                ))

    for row in new_rows:
        db.add(row)
