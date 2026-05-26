"""Tests for f_next_best_step — no API calls needed."""

from unittest.mock import MagicMock

from src.engine.f_next_best_step import MAX_TURNS, f_next_best_step
from src.engine.f_uncertainty import BoundaryPoint
from src.schemas import ChatSessionState, Cluster, FeedbackEntry


def _make_state(turn_number=1, n_clusters=3, contradictions=None):
    clusters = [
        Cluster(
            id=f"c{i}",
            session_id="s1",
            name=f"Cluster {i}",
            description="",
            created_at_turn=1,
            size=10,
            representative_points=[],
        )
        for i in range(n_clusters)
    ]
    return ChatSessionState(
        session_id="s1",
        turn_number=turn_number,
        dataset_name="test",
        embedding_model="default",
        status="active",
        clusters=clusters,
        feedback_history=[],
        contradictions=contradictions or [],
    )


def _make_context(load_score=1):
    ctx = MagicMock()
    ctx.get_cognitive_load_score.return_value = load_score
    return ctx


def _bp(score):
    return BoundaryPoint(
        point_id="p1",
        text_preview="some review text",
        cluster_scores={"c1": 1 - score, "c2": score},
        uncertainty_score=score,
    )


# ── tests ──────────────────────────────────────────────────────────────────

def test_returns_system_turn():
    from src.schemas import SystemTurn
    state = _make_state()
    ctx = _make_context()
    result = f_next_best_step(state, [], ctx)
    assert isinstance(result, SystemTurn)


def test_action_show_when_no_uncertainty():
    state = _make_state(turn_number=2)
    ctx = _make_context(load_score=1)
    result = f_next_best_step(state, [], ctx)
    assert result.action == "show"


def test_action_show_when_high_uncertainty():
    # Rule 2 (ask on uncertainty) was removed — high uncertainty now gives show,
    # not ask. The oracle can address ambiguous points on their own initiative.
    state = _make_state(turn_number=2)
    ctx = _make_context(load_score=1)
    uncertain_points = [_bp(0.9)]
    result = f_next_best_step(state, uncertain_points, ctx)
    assert result.action == "show"


def test_action_stop_when_cognitive_load_high():
    state = _make_state(turn_number=5)
    ctx = _make_context(load_score=4)
    result = f_next_best_step(state, [], ctx)
    assert result.action == "stop"


def test_action_stop_when_too_many_turns():
    state = _make_state(turn_number=MAX_TURNS + 1)
    ctx = _make_context(load_score=1)
    result = f_next_best_step(state, [], ctx)
    assert result.action == "stop"


def test_stop_takes_priority_over_show():
    # Even with high uncertainty, stop wins if load is too high
    state = _make_state(turn_number=5)
    ctx = _make_context(load_score=5)
    uncertain_points = [_bp(0.9)]
    result = f_next_best_step(state, uncertain_points, ctx)
    assert result.action == "stop"


def test_contradiction_detected_flag():
    state = _make_state(contradictions=["some contradiction"])
    ctx = _make_context()
    result = f_next_best_step(state, [], ctx)
    assert result.contradiction_detected is True


def test_no_contradiction_when_clean():
    state = _make_state(contradictions=[])
    ctx = _make_context()
    result = f_next_best_step(state, [], ctx)
    assert result.contradiction_detected is False


def test_cognitive_load_score_in_result():
    state = _make_state()
    ctx = _make_context(load_score=3)
    result = f_next_best_step(state, [], ctx)
    assert result.cognitive_load_score == 3


def test_uncertainty_always_gives_show():
    # Any uncertainty level now gives show — ask was removed
    state = _make_state()
    ctx = _make_context()
    result = f_next_best_step(state, [_bp(0.1)], ctx)
    assert result.action == "show"
    result2 = f_next_best_step(state, [_bp(0.9)], ctx)
    assert result2.action == "show"