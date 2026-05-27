"""Tests for f_next_best_step — no API calls needed."""

from unittest.mock import MagicMock

from src.engine.f_next_best_step import MAX_TURNS, f_next_best_step
from src.engine.f_uncertainty import ClusterCohesion, ClusterOverlap, ClusterUncertainty
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


def _empty_uncertainty():
    return ClusterUncertainty()


def _overlap_uncertainty(fraction=0.2):
    return ClusterUncertainty(
        overlaps=[ClusterOverlap(
            cluster_a_id="c0", cluster_b_id="c1",
            cluster_a_name="Cluster 0", cluster_b_name="Cluster 1",
            overlap_fraction=fraction, n_overlap=int(fraction * 100),
        )]
    )


def _cohesion_uncertainty(mean_max_prob=0.45):
    return ClusterUncertainty(
        low_cohesion=[ClusterCohesion(
            cluster_id="c0", cluster_name="Cluster 0",
            mean_max_prob=mean_max_prob,
        )]
    )


# ── tests ──────────────────────────────────────────────────────────────────

def test_returns_system_turn():
    from src.schemas import SystemTurn
    state = _make_state()
    ctx = _make_context()
    result = f_next_best_step(state, _empty_uncertainty(), ctx)
    assert isinstance(result, SystemTurn)


def test_action_show_when_no_uncertainty():
    state = _make_state(turn_number=2)
    ctx = _make_context(load_score=1)
    result = f_next_best_step(state, _empty_uncertainty(), ctx)
    assert result.action == "show"


def test_action_ask_merge_when_clusters_overlap():
    state = _make_state(turn_number=2)
    ctx = _make_context(load_score=1)
    result = f_next_best_step(state, _overlap_uncertainty(0.2), ctx)
    assert result.action == "ask"
    assert "Cluster 0" in result.display.content
    assert "Cluster 1" in result.display.content
    assert "overlap" in result.display.content.lower() or "ambiguous" in result.display.content.lower()


def test_ask_merge_message_uses_cluster_names_not_ids():
    state = _make_state(turn_number=2)
    ctx = _make_context(load_score=1)
    result = f_next_best_step(state, _overlap_uncertainty(0.2), ctx)
    # Must show names, not raw UUIDs
    assert "c0" not in result.display.content
    assert "c1" not in result.display.content
    assert "Cluster 0" in result.display.content


def test_action_ask_split_when_low_cohesion():
    state = _make_state(turn_number=2)
    ctx = _make_context(load_score=1)
    result = f_next_best_step(state, _cohesion_uncertainty(0.45), ctx)
    assert result.action == "ask"
    assert "Cluster 0" in result.display.content
    assert "split" in result.display.content.lower() or "cohesion" in result.display.content.lower()


def test_overlap_takes_priority_over_cohesion():
    state = _make_state(turn_number=2)
    ctx = _make_context(load_score=1)
    both = ClusterUncertainty(
        overlaps=_overlap_uncertainty(0.2).overlaps,
        low_cohesion=_cohesion_uncertainty(0.45).low_cohesion,
    )
    result = f_next_best_step(state, both, ctx)
    assert result.action == "ask"
    # Should mention merge (overlap), not split (cohesion)
    assert "Cluster 1" in result.display.content  # cluster_b_name only in overlap message


def test_action_stop_when_cognitive_load_high():
    # Threshold is now 5 (was 4) — only maximum load triggers auto-stop.
    state = _make_state(turn_number=5)
    ctx = _make_context(load_score=5)
    result = f_next_best_step(state, _empty_uncertainty(), ctx)
    assert result.action == "stop"


def test_action_show_when_cognitive_load_is_4():
    # load=4 used to trigger stop prematurely — now it gives show.
    state = _make_state(turn_number=5)
    ctx = _make_context(load_score=4)
    result = f_next_best_step(state, _empty_uncertainty(), ctx)
    assert result.action == "show"


def test_action_stop_when_too_many_turns():
    state = _make_state(turn_number=MAX_TURNS + 1)
    ctx = _make_context(load_score=1)
    result = f_next_best_step(state, _empty_uncertainty(), ctx)
    assert result.action == "stop"


def test_stop_takes_priority_over_show():
    state = _make_state(turn_number=5)
    ctx = _make_context(load_score=5)
    result = f_next_best_step(state, _overlap_uncertainty(0.9), ctx)
    assert result.action == "stop"


def test_contradiction_detected_flag():
    state = _make_state(contradictions=["some contradiction"])
    ctx = _make_context()
    result = f_next_best_step(state, _empty_uncertainty(), ctx)
    assert result.contradiction_detected is True


def test_no_contradiction_when_clean():
    state = _make_state(contradictions=[])
    ctx = _make_context()
    result = f_next_best_step(state, _empty_uncertainty(), ctx)
    assert result.contradiction_detected is False


def test_cognitive_load_score_in_result():
    state = _make_state()
    ctx = _make_context(load_score=3)
    result = f_next_best_step(state, _empty_uncertainty(), ctx)
    assert result.cognitive_load_score == 3


def test_uncertainty_empty_gives_show():
    state = _make_state()
    ctx = _make_context()
    assert f_next_best_step(state, _empty_uncertainty(), ctx).action == "show"
