"""Tests for f_next_best_step — no API calls needed."""

from src.engine.f_next_best_step import f_next_best_step
from src.engine.f_uncertainty import ClusterCohesion, ClusterOverlap, ClusterUncertainty
from src.schemas import ChatSessionState, Cluster, CognitiveLoad, FeedbackEntry


def _make_state(turn_number=1, n_clusters=3):
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
    )


def _make_load(score=1, driver="turns", turns_used=0, tokens_used=0, clusters_count=0):
    """Build a CognitiveLoad with the composite `score` driven by `driver`.

    The per-signal score for `driver` is set to `score`; the others to 1 so the
    composite (max) equals `score` and the priority resolution names `driver`.
    Caveat: this trick depends on tokens > clusters > turns priority. When the
    test wants driver="turns", we drop tokens_score and clusters_score below
    the composite explicitly.
    """
    sub = {"turns": 1, "tokens": 1, "clusters": 1}
    sub[driver] = score
    return CognitiveLoad(
        score=score,
        driver=driver,
        turns_score=sub["turns"],
        tokens_score=sub["tokens"],
        clusters_score=sub["clusters"],
        turns_used=turns_used,
        tokens_used=tokens_used,
        clusters_count=clusters_count,
    )


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
    result = f_next_best_step(state, _empty_uncertainty(), _make_load())
    assert isinstance(result, SystemTurn)


def test_action_show_when_no_uncertainty():
    state = _make_state(turn_number=2)
    result = f_next_best_step(state, _empty_uncertainty(), _make_load(score=1))
    assert result.action == "show"


def test_action_ask_merge_when_clusters_overlap():
    state = _make_state(turn_number=2)
    result = f_next_best_step(state, _overlap_uncertainty(0.2), _make_load(score=1))
    assert result.action == "ask"
    assert "Cluster 0" in result.display.content
    assert "Cluster 1" in result.display.content
    assert "overlap" in result.display.content.lower() or "ambiguous" in result.display.content.lower()


def test_ask_merge_message_uses_cluster_names_not_ids():
    state = _make_state(turn_number=2)
    result = f_next_best_step(state, _overlap_uncertainty(0.2), _make_load(score=1))
    assert "c0" not in result.display.content
    assert "c1" not in result.display.content
    assert "Cluster 0" in result.display.content


def test_action_ask_split_when_low_cohesion():
    state = _make_state(turn_number=2)
    result = f_next_best_step(state, _cohesion_uncertainty(0.45), _make_load(score=1))
    assert result.action == "ask"
    assert "Cluster 0" in result.display.content
    assert "split" in result.display.content.lower() or "cohesion" in result.display.content.lower()


def test_overlap_takes_priority_over_cohesion():
    state = _make_state(turn_number=2)
    both = ClusterUncertainty(
        overlaps=_overlap_uncertainty(0.2).overlaps,
        low_cohesion=_cohesion_uncertainty(0.45).low_cohesion,
    )
    result = f_next_best_step(state, both, _make_load(score=1))
    assert result.action == "ask"
    assert "Cluster 1" in result.display.content


def test_action_stop_when_cognitive_load_saturated():
    state = _make_state(turn_number=5)
    result = f_next_best_step(state, _empty_uncertainty(), _make_load(score=5))
    assert result.action == "stop"
    assert result.state_snapshot["reason"] == "cognitive_overload"


def test_action_show_when_cognitive_load_is_4():
    state = _make_state(turn_number=5)
    result = f_next_best_step(state, _empty_uncertainty(), _make_load(score=4))
    assert result.action == "show"


def test_no_auto_stop_on_high_turn_count_without_overload():
    state = _make_state(turn_number=100)
    result = f_next_best_step(state, _empty_uncertainty(), _make_load(score=1))
    assert result.action != "stop"


def test_stop_takes_priority_over_show():
    state = _make_state(turn_number=5)
    result = f_next_best_step(state, _overlap_uncertainty(0.9), _make_load(score=5))
    assert result.action == "stop"


def test_cognitive_load_score_in_result():
    state = _make_state()
    result = f_next_best_step(state, _empty_uncertainty(), _make_load(score=3, driver="tokens"))
    assert result.cognitive_load_score == 3


def test_uncertainty_empty_gives_show():
    state = _make_state()
    assert f_next_best_step(state, _empty_uncertainty(), _make_load()).action == "show"


# ── new behaviour: driver + breakdown in stop snapshot ─────────────────────


def test_stop_snapshot_records_load_driver():
    state = _make_state(turn_number=10)
    load = _make_load(score=5, driver="tokens", tokens_used=16000)
    result = f_next_best_step(state, _empty_uncertainty(), load)
    assert result.state_snapshot["cognitive_load_driver"] == "tokens"


def test_converged_when_oracle_signals_end_and_state_stable():
    state = _make_state(turn_number=5)
    result = f_next_best_step(
        state,
        _empty_uncertainty(),
        _make_load(score=2),
        oracle_signaled_end=True,
    )
    assert result.action == "stop"
    assert result.state_snapshot["reason"] == "converged"
    assert result.state_snapshot["cluster_count"] == len(state.clusters)
    assert "converged" in result.display.content.lower()


def test_show_when_oracle_did_not_signal_end():
    """Silence / no_change / explain must NOT trigger convergence — the
    oracle has to explicitly express satisfaction or close intent."""
    state = _make_state(turn_number=2)
    result = f_next_best_step(
        state,
        _empty_uncertainty(),
        _make_load(score=2),
        oracle_signaled_end=False,
    )
    assert result.action == "show"


def test_oracle_end_beats_unresolved_overlaps():
    """An explicit close from the oracle terminates the session as converged
    even when overlaps are still flagged — the oracle's stated intent to
    stop wins over any pending structural question."""
    state = _make_state(turn_number=5)
    result = f_next_best_step(
        state,
        _overlap_uncertainty(0.3),
        _make_load(score=2),
        oracle_signaled_end=True,
    )
    assert result.action == "stop"
    assert result.state_snapshot["reason"] == "converged"


def test_oracle_end_beats_cognitive_overload():
    """An explicit close from the oracle is always recorded as a converged
    termination, even when cognitive load would otherwise saturate. The
    oracle's stated intent wins over the deterministic overload signal."""
    state = _make_state(turn_number=5)
    result = f_next_best_step(
        state,
        _empty_uncertainty(),
        _make_load(score=5),
        oracle_signaled_end=True,
    )
    assert result.action == "stop"
    assert result.state_snapshot["reason"] == "converged"


def test_stop_snapshot_records_load_breakdown_and_raw():
    state = _make_state(turn_number=10, n_clusters=4)
    load = CognitiveLoad(
        score=5,
        driver="tokens",
        turns_score=2,
        tokens_score=5,
        clusters_score=1,
        turns_used=8,
        tokens_used=16000,
        clusters_count=4,
    )
    result = f_next_best_step(state, _empty_uncertainty(), load)
    assert result.state_snapshot["cognitive_load_breakdown"] == {
        "turns": 2, "tokens": 5, "clusters": 1,
    }
    assert result.state_snapshot["cognitive_load_raw"] == {
        "turns_used": 8, "tokens_used": 16000, "clusters_count": 4,
    }
