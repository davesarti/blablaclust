from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.main import get_db
from src.models import ChatSession, Turn

router = APIRouter(prefix="/turns", tags=["turns"])


class TurnPlaceholder(BaseModel):
    session_id: str
    turn_number: int
    oracle_input: dict
    system_output: dict


@router.get("", response_model=list[TurnPlaceholder])
def list_turns(
    session_id: str = Query(...),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    turns = (
        db.query(Turn)
        .filter(Turn.session_id == session_id)
        .order_by(Turn.turn_number.asc())
        .limit(limit)
        .all()
    )

    return [
        TurnPlaceholder(
            session_id=turn.session_id,
            turn_number=turn.turn_number,
            oracle_input=turn.oracle_input,
            system_output=turn.system_output,
        )
        for turn in turns
    ]


@router.get("/{session_id}/{turn_number}", response_model=TurnPlaceholder)
def read_turn(session_id: str, turn_number: int, db: Session = Depends(get_db)):
    turn = (
        db.query(Turn)
        .filter(Turn.session_id == session_id, Turn.turn_number == turn_number)
        .first()
    )
    if turn is None:
        raise HTTPException(status_code=404, detail="Turn not found")

    return TurnPlaceholder(
        session_id=turn.session_id,
        turn_number=turn.turn_number,
        oracle_input=turn.oracle_input,
        system_output=turn.system_output,
    )
