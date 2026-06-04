import json
import os
import statistics
import uuid

from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.eval_cache import compute_cache_key, get as cache_get, put as cache_put
from backend.main import get_db
from backend.session_state import build_session_state
from src.engine.f_eval import (
    f_eval_coherence,
    f_eval_compliance,
    f_eval_contradiction,
    f_eval_overall,
)
from src.models import ChatSession, DataPoint, Dataset, EvalCache, SoftAssignment, Turn
from src.schemas import ChatSessionState

router = APIRouter(prefix="/sessions", tags=["sessions"])

_FEEDBACK_TYPE_WEIGHTS = {
    "global": 2.0,
    "cluster": 1.0,
    "point": 0.5,
    "instructional": 0.0,
}


class CreateSessionRequest(BaseModel):
    dataset_id: str
    name: Optional[str] = None
    oracle_kind: Literal["human", "persona"] = "human"
    persona_snapshot: Optional[Dict[str, Any]] = Field(default=None)


class PatchSessionStateRequest(BaseModel):
    status: Optional[Literal["active", "converged", "closed"]] = None


class A1Metrics(BaseModel):
    silhouette_initial: Optional[float] = None
    silhouette_final: Optional[float] = None
    trend: list[float] = []


class A2Metrics(BaseModel):
    turns: int
    weighted_turns: float
    termination: str  # "converged" | "cognitive_overload"


class A3Metrics(BaseModel):
    cognitive_load_by_turn: list[int]
    mean_cognitive_load: Optional[float] = None
    cognitive_load_driver_by_turn: list[str] = []
    final_cognitive_load_breakdown: Optional[Dict[str, int]] = None


class ClusterCoherence(BaseModel):
    cluster_id: str
    cluster_name: str = ""
    coherence: float
    reasoning: str


class B1Metrics(BaseModel):
    """Overall synthesis verdict combining B2, B3, B4."""
    overall_score: float
    notes: str


class B2Metrics(BaseModel):
    """Per-cluster internal coherence."""
    coherence_mean: Optional[float] = None
    coherence_min: Optional[float] = None
    per_cluster: list[ClusterCoherence] = []


class B3Metrics(BaseModel):
    """Fidelity of system operations to oracle requests."""
    compliance_score: float
    notes: str = ""


class B4Metrics(BaseModel):
    """Oracle contradiction / ambiguity (how hard was the oracle to understand)."""
    contradiction_score: float
    notes: str = ""
    examples: list[str] = []


N_TOP_COHERENCE = 3
N_BOTTOM_COHERENCE = 2


class EvalResponse(BaseModel):
    session_id: str
    k_final: int
    cached: bool = False
    A1: A1Metrics
    A2: A2Metrics
    A3: A3Metrics
    B1: B1Metrics
    B2: B2Metrics
    B3: B3Metrics
    B4: B4Metrics


def _silhouette_trend(session_id: str) -> A1Metrics:
    path = "logs/clustering_runs.jsonl"
    if not os.path.exists(path):
        return A1Metrics()
    trend: list[float] = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("session_id") == session_id:
                v = rec.get("silhouette")
                if v is not None:
                    trend.append(v)
    return A1Metrics(
        silhouette_initial=trend[0] if trend else None,
        silhouette_final=trend[-1] if trend else None,
        trend=trend,
    )


def _coherence_samples(db: Session, state: ChatSessionState) -> list[dict]:
    """Top-N + bottom-N hard members per cluster, with resolved text.

    Top = highest soft-assignment probability (most representative); bottom =
    lowest (the weakest-fitting edge cases the coherence judge stress-tests).
    """
    cluster_ids = [c.id for c in state.clusters]
    if not cluster_ids:
        return []

    latest_turn = (
        db.query(func.max(SoftAssignment.turn_number))
        .filter(SoftAssignment.cluster_id.in_(cluster_ids))
        .scalar()
    )
    if latest_turn is None:
        return []

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
        members[cid].append((point_id, prob))
    for cid in members:
        members[cid].sort(key=lambda item: item[1], reverse=True)  # high → low

    plan: dict[str, tuple[list, list]] = {}
    needed_ids: set[str] = set()
    for cid, pts in members.items():
        if not pts:
            continue
        top = pts[:N_TOP_COHERENCE]
        bottom = pts[N_TOP_COHERENCE:][-N_BOTTOM_COHERENCE:]
        plan[cid] = (top, bottom)
        for point_id, _ in top + bottom:
            needed_ids.add(point_id)

    text_by_id: dict[str, str] = {}
    if needed_ids:
        for dp in db.query(DataPoint).filter(DataPoint.id.in_(needed_ids)).all():
            text_by_id[dp.id] = dp.text or dp.id

    cluster_by_id = {c.id: c for c in state.clusters}
    samples: list[dict] = []
    for cid, (top, bottom) in plan.items():
        samples.append({
            "cluster": cluster_by_id[cid],
            "top_texts": [text_by_id.get(pid, pid) for pid, _ in top],
            "bottom_texts": [text_by_id.get(pid, pid) for pid, _ in bottom],
        })
    return samples


def _final_clusters_block(state: ChatSessionState) -> str:
    return "\n\n".join(
        f"ID: {c.id}\n"
        f"Name: {c.name}\n"
        f"Description: {c.description or ''}\n"
        f"Size: {c.size} points\n"
        "Representatives:\n"
        + ("\n".join(f"- {r}" for r in c.representative_points) or "(none)")
        for c in state.clusters
    )


def _compliance_turns(db: Session, session_id: str) -> list[dict]:
    """Per-turn oracle request paired with the operations the system performed."""
    turns = (
        db.query(Turn)
        .filter(Turn.session_id == session_id)
        .order_by(Turn.turn_number.asc())
        .all()
    )
    payload: list[dict] = []
    for t in turns:
        oi = t.oracle_input or {}
        so = t.system_output or {}
        ops = (so.get("state_snapshot") or {}).get("operations", []) or []
        payload.append({
            "turn": t.turn_number,
            "oracle_request": {
                "raw_text": oi.get("raw_text", ""),
                "type": oi.get("feedback_type", "global"),
                "target_cluster_ids": oi.get("target_cluster_ids") or [],
                "target_point_ids": oi.get("target_point_ids") or [],
            },
            "operations": ops,
        })
    return payload


def _quality_metrics(
    state: ChatSessionState,
    coherence_samples: list[dict],
    compliance_turns: list[dict],
) -> tuple[B1Metrics, B2Metrics, B3Metrics, B4Metrics]:
    """Run the four LLM judges and return (B1 overall, B2 coherence, B3 compliance, B4 contradiction)."""
    coherence_results = (
        f_eval_coherence(state, coherence_samples) if coherence_samples else []
    )
    cvals = [r["coherence"] for r in coherence_results]
    coherence_mean = round(statistics.mean(cvals), 3) if cvals else None
    coherence_min = round(min(cvals), 3) if cvals else None

    compliance = f_eval_compliance(
        state, _final_clusters_block(state), compliance_turns
    )
    contradiction = f_eval_contradiction(state)

    overall = f_eval_overall(
        state,
        coherence_results,
        coherence_mean if coherence_mean is not None else 0.0,
        coherence_min if coherence_min is not None else 0.0,
        compliance,
        contradiction,
    )

    return (
        B1Metrics(
            overall_score=overall["overall_score"],
            notes=overall["notes"],
        ),
        B2Metrics(
            coherence_mean=coherence_mean,
            coherence_min=coherence_min,
            per_cluster=[ClusterCoherence(**r) for r in coherence_results],
        ),
        B3Metrics(
            compliance_score=compliance["compliance_score"],
            notes=compliance["notes"],
        ),
        B4Metrics(
            contradiction_score=contradiction["contradiction_score"],
            notes=contradiction["notes"],
            examples=contradiction["examples"],
        ),
    )


def _math_metrics(
    db: Session, session_id: str, state: ChatSessionState
) -> tuple[A2Metrics, A3Metrics]:
    turns = (
        db.query(Turn)
        .filter(Turn.session_id == session_id)
        .order_by(Turn.turn_number.asc())
        .all()
    )

    cog_loads: list[int] = []
    driver_by_turn: list[str] = []
    last_term_reason: Optional[str] = None
    last_breakdown: Optional[Dict[str, int]] = None

    for t in turns:
        so = t.system_output or {}
        snapshot = so.get("state_snapshot") or {}
        cog_loads.append(int(so.get("cognitive_load_score", 1)))
        driver = snapshot.get("cognitive_load_driver")
        if isinstance(driver, str):
            driver_by_turn.append(driver)
        last_term_reason = snapshot.get("reason")
        breakdown = snapshot.get("cognitive_load_breakdown")
        if isinstance(breakdown, dict):
            last_breakdown = {k: int(v) for k, v in breakdown.items()}

    if last_term_reason == "cognitive_overload":
        termination = "cognitive_overload"
    else:
        termination = "converged"

    weighted_turns = sum(
        _FEEDBACK_TYPE_WEIGHTS.get(f.type, 1.0) for f in state.feedback_history
    )

    return (
        A2Metrics(
            turns=len(turns),
            weighted_turns=round(weighted_turns, 2),
            termination=termination,
        ),
        A3Metrics(
            cognitive_load_by_turn=cog_loads,
            mean_cognitive_load=round(statistics.mean(cog_loads), 2) if cog_loads else None,
            cognitive_load_driver_by_turn=driver_by_turn,
            final_cognitive_load_breakdown=last_breakdown,
        ),
    )


@router.get("")
def read_sessions(db: Session = Depends(get_db)):
    chat_sessions = db.query(ChatSession).all()
    return [
        {
            "id": session.id,
            "name": session.name,
            "dataset_id": session.dataset_id,
            "dataset_name": session.dataset.name if session.dataset else "",
            "embedding_model": session.embedding_model,
            "status": session.status,
            "oracle_kind": session.oracle_kind,
            "persona_snapshot": session.persona_snapshot,
        }
        for session in chat_sessions
    ]


@router.post("")
def create_session(payload: CreateSessionRequest, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).one_or_none()
    if dataset is None:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset '{payload.dataset_id}' not found",
        )
    has_points = (
        db.query(DataPoint.id)
        .filter(DataPoint.dataset_id == dataset.id)
        .first()
        is not None
    )
    if not has_points:
        raise HTTPException(
            status_code=422,
            detail=f"No datapoints found for dataset '{dataset.name}'",
        )
    if payload.oracle_kind == "persona" and payload.persona_snapshot is None:
        raise HTTPException(
            status_code=422,
            detail="oracle_kind='persona' requires a persona_snapshot",
        )
    new_session = ChatSession(
        id=str(uuid.uuid4()),
        name=payload.name,
        dataset_id=dataset.id,
        embedding_model="default",
        status="active",
        oracle_kind=payload.oracle_kind,
        persona_snapshot=payload.persona_snapshot,
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return {
        "id": new_session.id,
        "name": new_session.name,
        "dataset_id": new_session.dataset_id,
        "dataset_name": dataset.name,
        "embedding_model": new_session.embedding_model,
        "status": new_session.status,
        "oracle_kind": new_session.oracle_kind,
        "persona_snapshot": new_session.persona_snapshot,
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


@router.get("/{session_id}/eval", response_model=EvalResponse)
def get_eval_cached(session_id: str, db: Session = Depends(get_db)):
    row = (
        db.query(EvalCache)
        .filter(EvalCache.session_id == session_id)
        .order_by(EvalCache.created_at.desc())
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="No cached eval for this session")
    data = json.loads(row.response_json)
    data["cached"] = True
    return EvalResponse(**data)


@router.post("/{session_id}/eval", response_model=EvalResponse)
def eval_session(
    session_id: str,
    force: bool = Query(False, description="Bypass the eval cache and recompute."),
    db: Session = Depends(get_db),
):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state = build_session_state(db, session)
    coherence_samples = _coherence_samples(db, state)
    compliance_turns = _compliance_turns(db, session_id)
    cache_key = compute_cache_key(state, coherence_samples, compliance_turns)

    if not force:
        cached = cache_get(db, cache_key)
        if cached is not None:
            cached["cached"] = True
            return EvalResponse(**cached)

    a1 = _silhouette_trend(session_id)
    a2, a3 = _math_metrics(db, session_id, state)
    b1, b2, b3, b4 = _quality_metrics(state, coherence_samples, compliance_turns)

    response = EvalResponse(
        session_id=session_id,
        k_final=len(state.clusters),
        cached=False,
        A1=a1,
        A2=a2,
        A3=a3,
        B1=b1,
        B2=b2,
        B3=b3,
        B4=b4,
    )
    cache_put(db, cache_key, session_id, response.model_dump())
    return response


@router.delete("/{session_id}/delete")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    db.delete(session)
    db.commit()
    return {"id": session_id, "status": "deleted"}
