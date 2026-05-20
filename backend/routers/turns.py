import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.main import get_db
from backend.session_state import build_session_state
from src.engine.f_apply_operations import f_apply_operations
from src.engine.f_output import f_output
from src.harness import ConversationContext
from src.models import ChatSession, Cluster as DbCluster, DataPoint, Turn
from src.schemas import InputOracle, TurnRead

router = APIRouter(prefix="/turns", tags=["turns"])


@router.get("", response_model=list[TurnRead])
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

    return [TurnRead.model_validate(turn, from_attributes=True) for turn in turns]


@router.get("/{session_id}", response_model=list[TurnRead])
def list_turns_for_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    turns = (
        db.query(Turn)
        .filter(Turn.session_id == session_id)
        .order_by(Turn.turn_number.asc())
        .all()
    )

    return [TurnRead.model_validate(turn, from_attributes=True) for turn in turns]


@router.get("/{session_id}/{turn_number}", response_model=TurnRead)
def read_turn(session_id: str, turn_number: int, db: Session = Depends(get_db)):
    turn = (
        db.query(Turn)
        .filter(Turn.session_id == session_id, Turn.turn_number == turn_number)
        .first()
    )
    if turn is None:
        raise HTTPException(status_code=404, detail="Turn not found")

    return TurnRead.model_validate(turn, from_attributes=True)


@router.post("", response_model=TurnRead, status_code=201)
def create_turn(payload: InputOracle, db: Session = Depends(get_db)):
    """Run one engine interaction step and persist it as a turn.

    Applies the oracle's feedback to the current clustering by calling the
    executor (f_output -> LLM) and stores the raw engine response as the turn's
    system_output. Records the turn only — it does not mutate the clusters or
    soft_assignments tables.
    """
    session = (
        db.query(ChatSession).filter(ChatSession.id == payload.session_id).first()
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status == "closed":
        raise HTTPException(
            status_code=409, detail="Session is closed — cannot add turns"
        )

    clusters = (
        db.query(DbCluster)
        .filter(
            DbCluster.session_id == session.id,
            DbCluster.dissolved_at_turn.is_(None),
        )
        .all()
    )
    if not clusters:
        raise HTTPException(
            status_code=409,
            detail="No active clusters for this session — run clustering first",
        )

    if payload.target_cluster_ids:
        active_ids = {cluster.id for cluster in clusters}
        unknown = [cid for cid in payload.target_cluster_ids if cid not in active_ids]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"target_cluster_ids contains unknown cluster(s): {unknown}",
            )

    state = build_session_state(db, session)

    # Replay stored turns so the LLM sees the full conversation history.
    context = ConversationContext(session_id=session.id)
    prior_turns = (
        db.query(Turn)
        .filter(Turn.session_id == session.id)
        .order_by(Turn.turn_number.asc())
        .all()
    )
    for turn in prior_turns:
        context.add_oracle_turn(turn.oracle_input)
        context.add_system_turn(turn.system_output)

    total_points = (
        db.query(DataPoint)
        .filter(DataPoint.dataset_name == session.dataset_name)
        .count()
    )

    try:
        raw, _usage = f_output(state, payload, context, total_points)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Engine returned a malformed response: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Engine call failed: {exc}"
        )

    latest_turn = (
        db.query(func.max(Turn.turn_number))
        .filter(Turn.session_id == session.id)
        .scalar()
    )
    new_turn_number = (latest_turn or 0) + 1

    new_turn = Turn(
        session_id=session.id,
        turn_number=new_turn_number,
        oracle_input=payload.model_dump(),
        system_output=raw,
    )
    db.add(new_turn)
    db.commit()

    f_apply_operations(raw, session_id=session.id, turn_number=new_turn_number, db=db)
    db.commit()

    db.refresh(new_turn)

    return TurnRead.model_validate(new_turn, from_attributes=True)
