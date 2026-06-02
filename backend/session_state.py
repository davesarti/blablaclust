from sqlalchemy import func
from sqlalchemy.orm import Session

from src.models import ChatSession, Cluster as DbCluster, DataPoint, SoftAssignment, Turn
from src.schemas import (
    ChatSessionState,
    Cluster as ClusterSchema,
    FeedbackEntry,
)


_REPR_PCT = 0.10   # fraction of hard-assigned points kept as representatives
_REPR_CAP = 30    # upper bound regardless of cluster size


def hard_cluster_stats(
    db: Session, session_id: str,
) -> dict[str, tuple[int, list[str]]]:
    """Hard-assignment size and representative points per cluster of a session.

    Each data point is assigned to its single highest-probability cluster at the
    latest turn that has soft assignments; a cluster's size is how many points
    land in it. Representatives are 10% of the cluster (capped at 30), split
    equally between the top-probability members (most confident, canonical
    examples) and the bottom-probability members (boundary cases), giving the
    eval judge and the UI a balanced view of the cluster's range.
    The argmax must run over *all* the session's clusters at once — scoping it to
    a subset would make every point trivially "belong" to whatever it is compared
    against. Returns zero-size entries when no soft assignments exist yet.
    """
    cluster_ids = [
        cid
        for (cid,) in db.query(DbCluster.id)
        .filter(
            DbCluster.session_id == session_id,
            DbCluster.dissolved_at_turn.is_(None),
        )
        .all()
    ]
    stats: dict[str, tuple[int, list[str]]] = {cid: (0, []) for cid in cluster_ids}
    if not cluster_ids:
        return stats

    latest_turn = (
        db.query(func.max(SoftAssignment.turn_number))
        .filter(SoftAssignment.cluster_id.in_(cluster_ids))
        .scalar()
    )
    if latest_turn is None:
        return stats

    assignments = (
        db.query(SoftAssignment)
        .filter(
            SoftAssignment.cluster_id.in_(cluster_ids),
            SoftAssignment.turn_number == latest_turn,
        )
        .all()
    )

    best_by_point: dict[str, tuple[str, float]] = {}
    for a in assignments:
        current = best_by_point.get(a.data_point_id)
        if current is None or a.probability > current[1]:
            best_by_point[a.data_point_id] = (a.cluster_id, a.probability)

    members: dict[str, list[tuple[str, float]]] = {cid: [] for cid in cluster_ids}
    for point_id, (cid, prob) in best_by_point.items():
        members.setdefault(cid, []).append((point_id, prob))
    for cid, points in members.items():
        points.sort(key=lambda item: item[1], reverse=True)
        size = len(points)
        n = min(max(1, int(size * _REPR_PCT)), _REPR_CAP)
        n_top = (n + 1) // 2   # ceil: if n is odd, top gets the extra one
        n_bot = n // 2
        top_ids = [pid for pid, _ in points[:n_top]]
        # bottom: walk from the end, skip any already in top (tiny clusters)
        seen = set(top_ids)
        bot_ids = [
            pid for pid, _ in reversed(points[n_top:])
            if pid not in seen
        ][:n_bot]
        stats[cid] = (size, top_ids + bot_ids)
    return stats


def build_cluster_schemas(
    db: Session,
    clusters: list[DbCluster],
    stats: dict[str, tuple[int, list[str]]] | None = None,
) -> list[ClusterSchema]:
    """Turn DB cluster rows into ClusterSchema with hard sizes/representatives.

    Pass precomputed `stats` (from hard_cluster_stats) to avoid re-querying;
    otherwise stats are computed once per distinct session in `clusters`.
    """
    if stats is None:
        stats = {}
        for session_id in {cluster.session_id for cluster in clusters}:
            stats.update(hard_cluster_stats(db, session_id))

    # Resolve all representative point IDs to their display texts in one query.
    all_rep_ids = [pid for _, (_, reps) in stats.items() for pid in reps]
    text_by_id: dict[str, str] = {}
    if all_rep_ids:
        for dp in db.query(DataPoint).filter(DataPoint.id.in_(all_rep_ids)).all():
            d = dp.data or {}
            text_by_id[dp.id] = d.get("text") or d.get("title") or dp.id

    schemas: list[ClusterSchema] = []
    for cluster in clusters:
        size, rep_ids = stats.get(cluster.id, (0, []))
        schemas.append(
            ClusterSchema(
                id=cluster.id,
                session_id=cluster.session_id,
                name=cluster.name,
                description=cluster.description,
                created_at_turn=cluster.created_at_turn,
                dissolved_at_turn=cluster.dissolved_at_turn,
                size=size,
                representative_points=[text_by_id.get(pid, pid) for pid in rep_ids],
            )
        )
    return schemas


def build_session_state(db: Session, session: ChatSession) -> ChatSessionState:
    """Assemble the current ChatSessionState for a session from the DB.

    turn_number is the latest persisted conversation turn (0 if none). Clusters
    reflect the active (non-dissolved) clusters with hard-assignment sizes.
    """
    latest_turn_number = (
        db.query(func.max(Turn.turn_number))
        .filter(Turn.session_id == session.id)
        .scalar()
        or 0
    )

    turns = (
        db.query(Turn)
        .filter(Turn.session_id == session.id)
        .order_by(Turn.turn_number.asc())
        .all()
    )
    feedback_history: list[FeedbackEntry] = []
    for turn in turns:
        oracle_input = turn.oracle_input
        feedback_history.append(
            FeedbackEntry(
                turn=turn.turn_number,
                type=oracle_input.get("feedback_type", "global"),
                content=oracle_input.get("raw_text", ""),
                target_cluster_ids=oracle_input.get("target_cluster_ids") or [],
                target_point_ids=oracle_input.get("target_point_ids") or [],
            )
        )

    clusters = (
        db.query(DbCluster)
        .filter(
            DbCluster.session_id == session.id,
            DbCluster.dissolved_at_turn.is_(None),
        )
        .all()
    )
    cluster_states = build_cluster_schemas(
        db, clusters, hard_cluster_stats(db, session.id)
    )

    return ChatSessionState(
        session_id=session.id,
        turn_number=latest_turn_number,
        dataset_name=session.dataset.name if session.dataset else "",
        embedding_model=session.embedding_model,
        status=session.status,
        clusters=cluster_states,
        feedback_history=feedback_history,
        oracle_preference_summary=session.preference_summary or None,
    )
