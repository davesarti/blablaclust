import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.main import get_db
from src.models import ChatSession, Cluster as DbCluster, DataPoint, SoftAssignment, Turn
from src.schemas import (
    ChatSessionState,
    Cluster as ClusterSchema,
    FeedbackEntry,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    dataset_name: str


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
