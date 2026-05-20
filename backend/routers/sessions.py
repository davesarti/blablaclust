import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.main import get_db
from src.engine.cluster_naming import name_clusters
from src.engine.initial_clustering import initial_clustering, silhouette_for_k, sweep_k
from src.models import ChatSession, Cluster as DbCluster, DataPoint, SoftAssignment, Turn
from src.schemas import (
    ChatSessionState,
    Cluster as ClusterSchema,
    FeedbackEntry,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    dataset_name: str


class ClusteringRequest(BaseModel):
    k: int = Field(default=5, ge=1, le=50)
    generate_names: bool = Field(
        default=True,
        description="Ask the LLM to name each cluster (set false to skip LLM calls)",
    )


@router.get("")
def read_sessions(db: Session = Depends(get_db)):
    chat_sessions = db.query(ChatSession).all()
    return [
        {
            "id": session.id,
            "dataset_name": session.dataset_name,
            "embedding_model": session.embedding_model,
            "status": session.status,
        }
        for session in chat_sessions
    ]


@router.post("")
def create_session(payload: CreateSessionRequest, db: Session = Depends(get_db)):
    dataset_exists = (
        db.query(DataPoint)
        .filter(DataPoint.dataset_name == payload.dataset_name)
        .first()
    )
    if dataset_exists is None:
        raise HTTPException(
            status_code=422,
            detail=f"No datapoints found for dataset '{payload.dataset_name}'",
        )
    new_session = ChatSession(
        id=str(uuid.uuid4()),
        dataset_name=payload.dataset_name,
        embedding_model="default",
        status="active",
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return {
        "id": new_session.id,
        "dataset_name": new_session.dataset_name,
        "embedding_model": new_session.embedding_model,
        "status": new_session.status,
    }


@router.get("/{session_id}/state", response_model=ChatSessionState)
def read_session_state(session_id: str, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    latest_turn = (
        db.query(func.max(Turn.turn_number))
        .filter(Turn.session_id == session_id)
        .scalar()
    )
    latest_turn_number = latest_turn or 0

    turns = (
        db.query(Turn)
        .filter(Turn.session_id == session_id)
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
                target_cluster_id=oracle_input.get("target_cluster_id"),
                target_point_ids=oracle_input.get("target_point_ids") or [],
            )
        )

    clusters = (
        db.query(DbCluster)
        .filter(DbCluster.session_id == session_id, DbCluster.dissolved_at_turn.is_(None))
        .all()
    )
    cluster_ids = [cluster.id for cluster in clusters]
    assignments_by_cluster: dict[str, list[tuple[str, float]]] = {}
    if latest_turn_number > 0 and cluster_ids:
        assignments = (
            db.query(SoftAssignment)
            .filter(
                SoftAssignment.cluster_id.in_(cluster_ids),
                SoftAssignment.turn_number == latest_turn_number,
            )
            .all()
        )
        for assignment in assignments:
            assignments_by_cluster.setdefault(assignment.cluster_id, []).append(
                (assignment.data_point_id, assignment.probability)
            )
        for cluster_id, items in assignments_by_cluster.items():
            items.sort(key=lambda item: item[1], reverse=True)

    cluster_states: list[ClusterSchema] = []
    for cluster in clusters:
        assignments = assignments_by_cluster.get(cluster.id, [])
        representative_points = [point_id for point_id, _ in assignments[:3]]
        cluster_states.append(
            ClusterSchema(
                id=cluster.id,
                session_id=cluster.session_id,
                name=cluster.name,
                description=cluster.description,
                created_at_turn=cluster.created_at_turn,
                dissolved_at_turn=cluster.dissolved_at_turn,
                size=len(assignments),
                representative_points=representative_points,
            )
        )

    return ChatSessionState(
        session_id=session.id,
        turn_number=latest_turn_number,
        dataset_name=session.dataset_name,
        status=session.status,
        clusters=cluster_states,
        feedback_history=feedback_history,
    )


@router.post("/{session_id}/clustering")
def run_initial_clustering(
    session_id: str,
    payload: ClusteringRequest,
    db: Session = Depends(get_db),
):
    """Run k-means on the session's dataset embeddings and persist the result.

    Can only be called once per session — returns 409 if clusters already exist.
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    existing = db.query(DbCluster).filter(DbCluster.session_id == session_id).first()
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail="Clustering already exists for this session — create a new session to re-cluster",
        )

    data_points = (
        db.query(DataPoint)
        .filter(DataPoint.dataset_name == session.dataset_name)
        .all()
    )
    if not data_points:
        raise HTTPException(
            status_code=422,
            detail=f"No data points found for dataset '{session.dataset_name}'",
        )

    try:
        db_clusters, db_assignments = initial_clustering(
            data_points=data_points,
            k=payload.k,
            session_id=session_id,
            turn_number=1,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if payload.generate_names:
        name_clusters(db_clusters, db_assignments, data_points)

    for cluster in db_clusters:
        db.add(cluster)
    for assignment in db_assignments:
        db.add(assignment)
    db.commit()

    # Hard cluster size = number of points whose highest probability is this cluster.
    best_cluster: dict[str, tuple[str, float]] = {}
    for a in db_assignments:
        current = best_cluster.get(a.data_point_id)
        if current is None or a.probability > current[1]:
            best_cluster[a.data_point_id] = (a.cluster_id, a.probability)
    cluster_sizes: dict[str, int] = {}
    for cluster_id, _ in best_cluster.values():
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

    # Silhouette score is only defined for 2 <= k < n_points.
    silhouette: float | None = None
    if 2 <= payload.k < len(data_points):
        try:
            silhouette = silhouette_for_k(data_points, payload.k)
        except ValueError:
            silhouette = None

    return {
        "session_id": session_id,
        "k": payload.k,
        "silhouette_score": silhouette,
        "clusters": [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "size": cluster_sizes.get(c.id, 0),
            }
            for c in db_clusters
        ],
        "assignments_created": len(db_assignments),
    }


@router.get("/{session_id}/clustering/suggest-k")
def suggest_k(
    session_id: str,
    k_min: int = 2,
    k_max: int = 10,
    db: Session = Depends(get_db),
):
    """Sweep k over [k_min, k_max] and return the silhouette score for each.

    A diagnostic to help the oracle choose the number of clusters before
    running POST /clustering — does not modify any state.
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    data_points = (
        db.query(DataPoint)
        .filter(DataPoint.dataset_name == session.dataset_name)
        .all()
    )
    if not data_points:
        raise HTTPException(
            status_code=422,
            detail=f"No data points found for dataset '{session.dataset_name}'",
        )

    try:
        scores = sweep_k(data_points, k_min=k_min, k_max=k_max)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if not scores:
        raise HTTPException(
            status_code=422,
            detail="Not enough data points to evaluate the requested k range",
        )

    best_k = max(scores, key=scores.get)
    return {
        "session_id": session_id,
        "scores": scores,
        "recommended_k": best_k,
    }
