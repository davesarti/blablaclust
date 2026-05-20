import uuid

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.main import get_db
from backend.session_state import build_session_state
from src.models import ChatSession, DataPoint
from src.schemas import ChatSessionState

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    dataset_name: str


class PatchSessionStateRequest(BaseModel):
    status: Optional[Literal["active", "converged", "closed"]] = None


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

    return build_session_state(db, session)


@router.patch("/{session_id}/state", response_model=ChatSessionState)
def patch_session_state(
    session_id: str, payload: PatchSessionStateRequest, db: Session = Depends(get_db)
):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if payload.status is None:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    session.status = payload.status
    db.commit()
    db.refresh(session)

    return build_session_state(db, session)


@router.delete("/{session_id}/delete")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    db.delete(session)
    db.commit()
    return {"id": session_id, "status": "deleted"}
