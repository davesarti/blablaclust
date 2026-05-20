from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.main import get_db
from src.models import ChatSession, Cluster as DbCluster
from src.schemas import Cluster as ClusterSchema

router = APIRouter(prefix="/clusters", tags=["clusters"])


def _get_session_or_404(session_id: str, db: Session) -> ChatSession:
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _build_cluster_schemas(clusters: list[DbCluster]) -> list[ClusterSchema]:
    return [
        ClusterSchema(
            id=cluster.id,
            session_id=cluster.session_id,
            name=cluster.name,
            description=cluster.description,
            created_at_turn=cluster.created_at_turn,
            dissolved_at_turn=cluster.dissolved_at_turn,
            size=0,
            representative_points=[],
        )
        for cluster in clusters
    ]


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
    return _build_cluster_schemas(clusters)


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
    return _build_cluster_schemas(clusters)


@router.get("/{cluster_id}", response_model=ClusterSchema)
def read_cluster(cluster_id: str, db: Session = Depends(get_db)):
    cluster = db.query(DbCluster).filter(DbCluster.id == cluster_id).first()
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    return _build_cluster_schemas([cluster])[0]
