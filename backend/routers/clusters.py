from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.main import get_db
from backend.session_state import build_cluster_schemas
from src.engine.cluster_naming import name_clusters
from src.engine.initial_clustering import initial_clustering, silhouette_for_k, sweep_k
from src.models import ChatSession, Cluster as DbCluster, DataPoint
from src.schemas import Cluster as ClusterSchema

router = APIRouter(prefix="/clusters", tags=["clusters"])


class ClusteringRequest(BaseModel):
    k: int = Field(default=5, ge=1, le=50)
    generate_names: bool = Field(
        default=True,
        description="Ask the LLM to name each cluster (set false to skip LLM calls)",
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
