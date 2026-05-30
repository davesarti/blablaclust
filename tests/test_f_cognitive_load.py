"""Unit tests for f_cognitive_load — pure function, no LLM calls."""

import os
import pytest

from src.engine.cognitive_load_caps import (
    CLUSTER_BUDGET,
    MAX_TURNS,
    TOKEN_BUDGET,
)
from src.engine.f_cognitive_load import f_cognitive_load
from src.harness import ConversationContext
from src.schemas import ChatSessionState, Cluster, CognitiveLoad


# ── fixtures ───────────────────────────────────────────────────────────────


def _make_state(n_clusters: int, turn_number: int = 1) -> ChatSessionState:
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


def _make_context(n_turns: int, char_total: int = 0) -> ConversationContext:
    """A ConversationContext primed with n_turns oracle turns and approximate
    character total (split evenly across the turns). char_total drives the
    fallback token estimate when not running on Claude."""
    ctx = ConversationContext(session_id="s1")
    # Use a non-claude provider so build_messages returns turns un-trimmed and
    # the function's token-count path uses the deterministic char-based estimate.
    os.environ["LLM_PROVIDER"] = "openrouter"
    per_turn_chars = char_total // max(n_turns, 1) if n_turns else 0
    for i in range(n_turns):
        ctx.add_oracle_turn(
            {"raw_text": "x" * per_turn_chars, "feedback_type": "global"}
        )
        ctx.add_system_turn({"display": {"type": "text", "content": "ok"}})
    return ctx


# ── base case ──────────────────────────────────────────────────────────────


def test_returns_cognitive_load_instance():
    state = _make_state(n_clusters=3)
    ctx = _make_context(n_turns=0)
    result = f_cognitive_load(state, ctx)
    assert isinstance(result, CognitiveLoad)


def test_fresh_session_score_is_one():
    """Zero turns, zero tokens, zero clusters → all per-signal scores = 1."""
    state = _make_state(n_clusters=0)
    ctx = _make_context(n_turns=0)
    result = f_cognitive_load(state, ctx)
    assert result.score == 1
    assert result.turns_score == 1
    assert result.tokens_score == 1
    assert result.clusters_score == 1


# ── single-signal saturation ───────────────────────────────────────────────


def test_turns_alone_saturates_to_five():
    state = _make_state(n_clusters=1)
    ctx = _make_context(n_turns=MAX_TURNS)
    result = f_cognitive_load(state, ctx)
    assert result.turns_score == 5
    assert result.score == 5
    assert result.driver == "turns"


def test_clusters_alone_saturates_to_five():
    state = _make_state(n_clusters=CLUSTER_BUDGET)
    ctx = _make_context(n_turns=0)
    result = f_cognitive_load(state, ctx)
    assert result.clusters_score == 5
    assert result.score == 5
    assert result.driver == "clusters"


def test_tokens_alone_saturates_to_five():
    # 4 chars per estimated token in the fallback path.
    state = _make_state(n_clusters=1)
    ctx = _make_context(n_turns=1, char_total=TOKEN_BUDGET * 4)
    result = f_cognitive_load(state, ctx)
    assert result.tokens_score == 5
    assert result.score == 5
    assert result.driver == "tokens"


# ── tiebreak determinism ───────────────────────────────────────────────────


def test_tiebreak_tokens_over_clusters_over_turns():
    """When multiple signals share the max, priority is tokens > clusters > turns."""
    state = _make_state(n_clusters=CLUSTER_BUDGET)
    ctx = _make_context(n_turns=MAX_TURNS, char_total=TOKEN_BUDGET * 4)
    result = f_cognitive_load(state, ctx)
    assert result.score == 5
    assert result.driver == "tokens"


def test_tiebreak_clusters_over_turns():
    state = _make_state(n_clusters=CLUSTER_BUDGET)
    ctx = _make_context(n_turns=MAX_TURNS)
    result = f_cognitive_load(state, ctx)
    assert result.driver == "clusters"


# ── smooth ladder ──────────────────────────────────────────────────────────


def test_smooth_ladder_for_turns_only():
    """Sweep turns 1..20 with empty state and minimal tokens → ladder
    1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5."""
    expected = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
    state = _make_state(n_clusters=1)
    got = []
    for t in range(1, MAX_TURNS + 1):
        ctx = _make_context(n_turns=t)
        got.append(f_cognitive_load(state, ctx).turns_score)
    assert got == expected


# ── max() aggregation ──────────────────────────────────────────────────────


def test_score_equals_max_of_per_signal_scores():
    """Mixed signals: composite equals the per-signal max."""
    state = _make_state(n_clusters=5)  # clusters_score = ceil(5/25*5) = 1
    ctx = _make_context(n_turns=12)    # turns_score    = ceil(12/20*5) = 3
    result = f_cognitive_load(state, ctx)
    assert result.score == max(
        result.turns_score, result.tokens_score, result.clusters_score
    )


# ── raw values are exposed for the eval report ────────────────────────────


def test_raw_values_match_inputs():
    state = _make_state(n_clusters=7)
    ctx = _make_context(n_turns=4)
    result = f_cognitive_load(state, ctx)
    assert result.turns_used == 4
    assert result.clusters_count == 7
    assert result.tokens_used >= 0
