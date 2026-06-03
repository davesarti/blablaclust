import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.main import get_db
from backend.session_state import build_session_state
from src.engine.cluster_operations import batch_move_points
from src.engine.f_apply_operations import f_apply_operations
from src.engine.f_boundary_repair import f_boundary_repair
from src.engine.f_cognitive_load import f_cognitive_load
from src.engine.f_next_best_step import f_next_best_step
from src.engine.f_output import f_output
from src.engine.f_semantic_reembed import AxisNotDiscriminativeError
from src.engine.f_uncertainty import f_cluster_uncertainty
from src.engine.f_update_preferences import f_update_preferences
from src.engine.semantic_clustering import semantic_clustering
from src.engine.turn_builder import TurnBuilder
from src.harness import ConversationContext, estimate_cost_usd
from src.models import ChatSession, Cluster as DbCluster, DataPoint, Turn
from src.schemas import Display, InputOracle, SystemTurn, TurnRead

router = APIRouter(prefix="/turns", tags=["turns"])


def _active_axis_from_history(prior_turns: list[Turn]) -> str | None:
    """Return the axis_label of the most recent semantic_reembed op, if any."""
    for turn in reversed(prior_turns):
        output = turn.system_output or {}
        ops = (output.get("state_snapshot") or {}).get("operations") or []
        for op in ops:
            if isinstance(op, dict) and op.get("type") == "semantic_reembed":
                # Older sessions stored the axis under "axis_hint"; new ones use
                # "axis_label". Accept either so resumed sessions still work.
                label = op.get("axis_label") or op.get("axis_hint")
                if label:
                    return str(label)
    return None


_AFFIRMATION_WORDS = frozenset({
    "yes", "sì", "si", "ok", "okay", "sure", "yep", "yeah",
    "go ahead", "do it", "correct", "confirm", "proceed",
    "please", "absolutely", "definitely", "do that", "please do",
    "vai", "fallo", "procedi", "confermo", "esatto",
})


def _is_affirmation(text: str) -> bool:
    normalized = text.strip().lower().rstrip("!.?, ")
    return normalized in _AFFIRMATION_WORDS


def _pending_clarify_axis(prior_turns: list[Turn]) -> str | None:
    """Return the pending axis_label stored by the most recent clarify turn, if any."""
    if not prior_turns:
        return None
    last = prior_turns[-1]
    ops = ((last.system_output or {}).get("state_snapshot") or {}).get("operations") or []
    for op in ops:
        if isinstance(op, dict) and op.get("type") == "clarify_pending":
            label = op.get("axis_label")
            return str(label) if label else None
    return None


def _run_semantic_reembed(
    session: ChatSession,
    axis_label: str,
    builder: TurnBuilder,
):
    """Re-cluster all dataset points along axis_label, staging on the builder."""
    print(
        f"[turns] semantic-reembed  session={session.id}  "
        f"axis_label='{axis_label}'  turn={builder.turn_number}",
        flush=True,
    )
    all_data_points = (
        builder.db.query(DataPoint)
        .filter(DataPoint.dataset_id == session.dataset_id)
        .all()
    )
    return semantic_clustering(
        data_points=all_data_points,
        axis_hint=axis_label,
        builder=builder,
    )


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
    executor (f_output -> LLM), stores the raw engine response as the turn's
    system_output, and applies any returned operations to clusters and
    soft_assignments.
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

    # The next turn number is known from state: state.turn_number is the last
    # persisted oracle turn (0 if none), so the next one is state.turn_number + 1.
    new_turn_number = state.turn_number + 1

    # Active semantic axis = axis_label from the most recent prior semantic_reembed
    # op, if any. Used so merge/split on subsequent turns name children along the
    # same axis. None when the session never re-embedded.
    session_axis_hint = _active_axis_from_history(prior_turns)

    # ── Clarify-confirmation short-circuit ───────────────────────────────────
    # If the previous turn stored a clarify_pending axis and the oracle's reply
    # is a short affirmation, skip the LLM and directly trigger semantic_reembed.
    pending_axis = _pending_clarify_axis(prior_turns)
    if pending_axis and _is_affirmation(payload.raw_text):
        print(
            f"[turns] clarify confirmed  session={session.id}  "
            f"axis='{pending_axis}'  oracle_text={payload.raw_text!r}",
            flush=True,
        )
        raw: dict = {
            "action": "semantic_reembed",
            "operations": [{"type": "semantic_reembed", "axis_label": pending_axis}],
            "display": f"Reorganising all clusters along the '{pending_axis}' axis.",
        }
        context.add_oracle_turn(payload.model_dump())
        context.add_system_turn(raw)
        usage: dict = {}
        cost: float = 0.0
    else:
        # Normal path: the LLM is the single intent-classification step.
        total_points = (
            db.query(DataPoint)
            .filter(DataPoint.dataset_id == session.dataset_id)
            .count()
        )
        try:
            raw, usage = f_output(state, payload, context, total_points)
            cost = estimate_cost_usd(usage)
            print(
                f"[turns] f_output  session={session.id}  turn={new_turn_number}  "
                f"input_tokens={usage.get('input_tokens', 0)}  "
                f"output_tokens={usage.get('output_tokens', 0)}  "
                f"cost_usd=${cost:.4f}",
                flush=True,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Engine returned a malformed response: {exc}",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"Engine call failed: {exc}"
            )

    if isinstance(raw, list):
        operations = raw
    else:
        operations = raw.get("operations", [])
    print(
        f"[turns] operations  session={session.id}  turn={new_turn_number}  "
        f"count={len(operations)}  types={[op.get('type') for op in operations]}",
        flush=True,
    )

    raw_display = raw.get("display") if isinstance(raw, dict) else None
    turn_usage = usage
    turn_cost = cost

    # ── Clarify action: store pending axis, execute nothing ───────────────────
    # When f_output asks for clarification, save the candidate axis in the
    # state_snapshot so the next turn can detect it with _pending_clarify_axis.
    snapshot_operations: list = []
    if isinstance(raw, dict) and raw.get("action") == "clarify":
        pending_label = (raw.get("axis_label") or "").strip()
        if pending_label:
            snapshot_operations = [{"type": "clarify_pending", "axis_label": pending_label}]
            print(
                f"[turns] clarify stored  session={session.id}  "
                f"axis='{pending_label}'",
                flush=True,
            )
        operations = []  # no clustering changes this turn

    # ── Run the conv turn's clustering changes on an in-memory builder ───────
    # The builder accumulates merge/split/move/reembed mutations plus boundary
    # repair, and commits ONCE so the snapshot writes at exactly turn=new_turn_number.
    # That keeps Turn.turn_number, SoftAssignment.turn_number, and
    # Cluster.created_at_turn / dissolved_at_turn in lockstep — no separate
    # snapshot axis.
    reembed_op = next(
        (op for op in operations if op.get("type") == "semantic_reembed"), None
    )
    if operations:
        try:
            builder = TurnBuilder.load(session.id, new_turn_number, db)
        except ValueError as exc:
            raise HTTPException(
                status_code=409, detail=f"Cannot start turn: {exc}"
            )

    if reembed_op is not None:
        # The prompt declares semantic_reembed exclusive; if the LLM accidentally
        # mixes it with structural ops we take the re-embed (it would dissolve the
        # other ops' target clusters anyway) and log a warning.
        if len(operations) > 1:
            print(
                f"[turns] semantic_reembed mixed with other ops — taking re-embed only  "
                f"session={session.id}  other_types="
                f"{[op.get('type') for op in operations if op is not reembed_op]}",
                flush=True,
            )
        axis_label = (reembed_op.get("axis_label") or "").strip()
        if not axis_label:
            raise HTTPException(
                status_code=422,
                detail="semantic_reembed operation requires a non-empty axis_label",
            )
        try:
            new_clusters = _run_semantic_reembed(
                session=session,
                axis_label=axis_label,
                builder=builder,
            )
            builder.commit()
            db.commit()
        except AxisNotDiscriminativeError:
            db.rollback()
            return TurnRead(
                session_id=session.id,
                turn_number=new_turn_number - 1,  # don't advance — let oracle retry
                oracle_input=payload,
                system_output=SystemTurn(
                    session_id=session.id,
                    turn_number=new_turn_number - 1,
                    action="ask",
                    clusters_updated=False,
                    display=Display(
                        type="text",
                        content=(
                            f"The axis \"{axis_label}\" doesn't vary enough across "
                            f"the dataset to produce meaningful clusters. "
                            f"Try a different axis — one that is clearly present and "
                            f"spans a range in the data."
                        ),
                    ),
                    cognitive_load_score=1,
                ),
            )
        except (ValueError, RuntimeError) as exc:
            db.rollback()
            raise HTTPException(
                status_code=422,
                detail=f"Semantic re-embedding failed: {exc}",
            )

        # Normalize the persisted op shape — keep the axis under axis_label so
        # _active_axis_from_history can find it on the next turn.
        operations = [
            {
                "type": "semantic_reembed",
                "axis_label": axis_label,
                "clusters_created": len(new_clusters),
            }
        ]

    elif operations:
        # ── Structural ops + (optional) boundary repair on the same builder ──
        try:
            f_apply_operations(
                operations,
                builder=builder,
                axis_hint=session_axis_hint,
            )
        except (ValueError, KeyError) as exc:
            db.rollback()
            raise HTTPException(
                status_code=422,
                detail=f"Engine produced an invalid operation: {exc}",
            )

        structural_types = {op.get("type") for op in operations}
        if structural_types & {"merge", "split"}:
            # Boundary repair targets the clusters created during THIS turn —
            # those are the ones whose k-means placement might have boundary
            # mistakes the LLM can correct.
            affected_clusters = [
                c.id
                for c in builder.new_clusters.values()
                if c.id not in builder.dissolved_ids
            ]
            moves = f_boundary_repair(
                builder=builder,
                affected_cluster_ids=affected_clusters,
                oracle_text=payload.raw_text,
            )
            if moves:
                try:
                    batch_move_points(
                        [(m["point_id"], m["target_cluster_id"]) for m in moves],
                        builder=builder,
                    )
                    print(
                        f"[turns] boundary-repair  session={session.id}  "
                        f"moved={len(moves)}",
                        flush=True,
                    )
                except Exception as exc:
                    log_repair_skip = f"boundary repair moves failed ({exc}), skipping"
                    print(f"[turns] {log_repair_skip}", flush=True)

        try:
            builder.commit()
            db.commit()
        except Exception as exc:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to commit turn: {exc}",
            )

    # ── Common path: planner + persist turn ───────────────────────────────────
    updated_state = build_session_state(db, session)
    uncertainty = f_cluster_uncertainty(session.id, db)
    cognitive_load = f_cognitive_load(updated_state, context)
    system_turn = f_next_best_step(updated_state, uncertainty, cognitive_load)
    system_turn.clusters_updated = bool(operations)
    system_turn.token_usage = turn_usage
    system_turn.cost_usd = turn_cost

    # Always surface the LLM's actual reply, regardless of action.
    # Guard: only use the LLM's display text if it is plain prose, not JSON.
    # Small/free models sometimes put structured JSON inside the display field
    # instead of a human-readable explanation, which would show raw JSON in
    # the chat. If the value parses as JSON or starts with { / [, fall back
    # to the f_next_best_step message which is always a proper English string.
    if isinstance(raw_display, str) and raw_display.strip():
        stripped = raw_display.strip()
        is_json = stripped.startswith(("{", "["))
        if not is_json:
            try:
                json.loads(stripped)
                is_json = True
            except (ValueError, TypeError):
                pass
        if not is_json:
            system_turn.display.content = raw_display

    # Close the session when the planner decides to stop.
    if system_turn.action == "stop":
        session.status = "closed"

    ops_to_store = snapshot_operations if snapshot_operations else operations
    if ops_to_store:
        system_turn.state_snapshot["operations"] = ops_to_store

    # Persist the turn once, with the final SystemTurn as system_output. Creating
    # the row only here (rather than up-front) keeps the stored shape always valid
    # against TurnRead/SystemTurn — a turn that fails mid-processing never lands
    # a half-baked row in the DB.
    new_turn = Turn(
        session_id=session.id,
        turn_number=new_turn_number,
        oracle_input=payload.model_dump(),
        system_output=system_turn.model_dump(),
    )
    db.add(new_turn)
    db.commit()
    db.refresh(new_turn)

    # Update the rolling oracle preference summary.  This is a best-effort
    # background step: if the LLM call inside f_update_preferences fails, we
    # keep the previous summary and never raise.  We build a fresh state that
    # includes the turn we just persisted so the summary covers all turns.
    final_state = build_session_state(db, session)
    new_summary = f_update_preferences(final_state)
    if new_summary is not None:
        session.preference_summary = new_summary
        db.commit()

    return TurnRead.model_validate(new_turn, from_attributes=True)
