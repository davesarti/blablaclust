from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.main import get_db
from backend.session_state import build_cluster_schemas
from src.engine.cluster_naming import name_clusters
from src.engine.f_parse_clustering_intent import f_parse_clustering_intent
from src.engine.initial_clustering import initial_clustering, sweep_k
from src.models import ChatSession, Cluster as DbCluster, DataPoint, SoftAssignment
from src.schemas import Cluster as ClusterSchema, ClusterPointsResponse, ClusterPoint

router = APIRouter(prefix="/clusters", tags=["clusters"])


class ClusteringRequest(BaseModel):
    k: int = Field(default=5, ge=1, le=50)
    generate_names: bool = Field(
        default=True,
        description="Ask the LLM to name each cluster (set false to skip LLM calls)",
    )
    oracle_intent: str | None = Field(
        default=None,
        description=(
            "Free-text description of how the oracle wants to cluster "
            "(e.g. 'separate positive from negative reviews'). "
            "When provided, Claude extracts k and axis automatically "
            "and the k field is ignored."
        ),
    )


def _get_session_or_404(session_id: str, db: Session) -> ChatSession:
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/{session_id}")
def run_initial_clustering(
    session_id: str,
    payload: ClusteringRequest,
    db: Session = Depends(get_db),
):
    """Run k-means on the session's dataset embeddings and persist the result.

    Can only be called once per session — returns 409 if clusters already exist.
    """
    session = _get_session_or_404(session_id, db)

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

    # If the oracle described their intent in natural language, let Claude extract k.
    # Otherwise fall back to the explicit k from the request body.
    if payload.oracle_intent:
        parsed = f_parse_clustering_intent(
            payload.oracle_intent,
            k_min=1,
            k_max=min(50, len(data_points)),
        )
        k = parsed["k"]
    else:
        k = payload.k

    try:
        db_clusters, db_assignments, silhouette = initial_clustering(
            data_points=data_points,
            k=k,
            session_id=session_id,
            turn_number=0,
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

    # silhouette is whatever initial_clustering computed for THIS exact k-means
    # fit (None when k < 2 or k >= n_points). Reusing it avoids a second full
    # k-means run and guarantees the returned value matches the logged one.

    return {
        "session_id": session_id,
        "k": k,
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


@router.get("/{session_id}/suggest-k")
def suggest_k(
    session_id: str,
    k_min: int = 2,
    k_max: int = 10,
    db: Session = Depends(get_db),
):
    """Sweep k over [k_min, k_max] and return the silhouette score for each.

    A diagnostic to help the oracle choose the number of clusters before
    running POST /clusters/{session_id} — does not modify any state.
    """
    session = _get_session_or_404(session_id, db)

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


@router.get("/active", response_model=list[ClusterSchema])
def list_active_clusters(
    session_id: str = Query(...),
    db: Session = Depends(get_db),
):
    _get_session_or_404(session_id, db)
    clusters = (
        db.query(DbCluster)
        .filter(
            DbCluster.session_id == session_id,
            DbCluster.dissolved_at_turn.is_(None),
        )
        .order_by(DbCluster.created_at_turn.asc())
        .all()
    )
    return build_cluster_schemas(db, clusters)


@router.get("/history", response_model=list[ClusterSchema])
def list_cluster_history(
    session_id: str = Query(...),
    db: Session = Depends(get_db),
):
    _get_session_or_404(session_id, db)
    clusters = (
        db.query(DbCluster)
        .filter(DbCluster.session_id == session_id)
        .order_by(DbCluster.created_at_turn.asc())
        .all()
    )
    return build_cluster_schemas(db, clusters)


@router.get("/{cluster_id}", response_model=ClusterSchema)
def read_cluster(cluster_id: str, db: Session = Depends(get_db)):
    cluster = db.query(DbCluster).filter(DbCluster.id == cluster_id).first()
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    return build_cluster_schemas(db, [cluster])[0]


@router.get("/{cluster_id}/points", response_model=ClusterPointsResponse)
def list_cluster_points(cluster_id: str, db: Session = Depends(get_db)):
    cluster = db.query(DbCluster).filter(DbCluster.id == cluster_id).first()
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    cluster_ids = [
        cid
        for (cid,) in db.query(DbCluster.id)
        .filter(DbCluster.session_id == cluster.session_id)
        .all()
    ]
    if not cluster_ids:
        return ClusterPointsResponse(
            cluster_id=cluster_id,
            session_id=cluster.session_id,
            turn_number=None,
            points=[],
        )

    latest_turn = (
        db.query(func.max(SoftAssignment.turn_number))
        .filter(SoftAssignment.cluster_id.in_(cluster_ids))
        .scalar()
    )
    if latest_turn is None:
        return ClusterPointsResponse(
            cluster_id=cluster_id,
            session_id=cluster.session_id,
            turn_number=None,
            points=[],
        )

    assignments = (
        db.query(SoftAssignment)
        .filter(
            SoftAssignment.cluster_id.in_(cluster_ids),
            SoftAssignment.turn_number == latest_turn,
        )
        .all()
    )

    best_by_point: dict[str, tuple[str, float]] = {}
    for assignment in assignments:
        current = best_by_point.get(assignment.data_point_id)
        if current is None or assignment.probability > current[1]:
            best_by_point[assignment.data_point_id] = (
                assignment.cluster_id,
                assignment.probability,
            )

    cluster_members = [
        (point_id, prob)
        for point_id, (cid, prob) in best_by_point.items()
        if cid == cluster_id
    ]
    cluster_members.sort(key=lambda item: item[1], reverse=True)

    point_ids = [point_id for point_id, _ in cluster_members]
    points = (
        db.query(DataPoint).filter(DataPoint.id.in_(point_ids)).all()
        if point_ids
        else []
    )
    data_by_id = {point.id: point.data for point in points}

    return ClusterPointsResponse(
        cluster_id=cluster_id,
        session_id=cluster.session_id,
        turn_number=latest_turn,
        points=[
            ClusterPoint(
                id=point_id,
                data=data_by_id.get(point_id, {}),
                probability=probability,
            )
            for point_id, probability in cluster_members
        ],
    )
