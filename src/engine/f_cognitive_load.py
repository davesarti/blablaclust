"""Deterministic LLM-side cognitive load score (A3).

Pure function. Reads `ChatSessionState` (for cluster count) and
`ConversationContext` (for turn count and prompt token size) and returns a
`CognitiveLoad` whose 1..5 composite score is the max of three per-signal
scores. The Planner reads `.score`; the eval report reads `.driver` and the
per-signal breakdown.
"""

from __future__ import annotations

import math
import os

from src.engine.cognitive_load_caps import (
    CLUSTER_BUDGET,
    MAX_TURNS,
    TOKEN_BUDGET,
)
from src.harness import ConversationContext, count_tokens
from src.schemas import ChatSessionState, CognitiveLoad

# When multiple signals tie at the composite max, this priority decides the
# `driver`. Tokens wins because it is the most direct proxy for prompt bloat.
_DRIVER_PRIORITY: tuple[str, ...] = ("tokens", "clusters", "turns")


def _score_signal(used: int, cap: int) -> int:
    """Map a raw signal value to a 1..5 sub-score.

    `ceil(fraction * 5)` produces a smooth ladder rather than long flat
    plateaus. `used = 0` gives score 1; `used >= cap` saturates at 5.
    """
    if cap <= 0:
        return 1
    fraction = min(1.0, used / cap)
    return max(1, math.ceil(fraction * 5))


def _estimate_tokens(messages: list[dict[str, str]], model: str | None = None) -> int:
    """Pre-trim token count of the conversation.

    Uses Anthropic's `count_tokens` when the active provider is Claude;
    otherwise falls back to a character-based estimate (~4 chars / token).
    """
    provider = os.environ.get("LLM_PROVIDER", "claude").lower()
    if provider == "claude":
        try:
            return count_tokens(messages, system="", model=model)
        except Exception:
            pass
    char_total = sum(len(m.get("content", "")) for m in messages)
    return char_total // 4


def f_cognitive_load(
    state: ChatSessionState,
    context: ConversationContext,
) -> CognitiveLoad:
    """Compute the deterministic A3 cognitive-load score for the current turn."""
    turns_used = len(context._oracle_turns)
    clusters_count = len(state.clusters)
    # Pre-trim prompt size: pass context.turns directly, bypassing the
    # build_messages() trim loop so we measure how much history we are
    # *trying* to carry.
    tokens_used = _estimate_tokens(context.turns)

    turns_score = _score_signal(turns_used, MAX_TURNS)
    tokens_score = _score_signal(tokens_used, TOKEN_BUDGET)
    clusters_score = _score_signal(clusters_count, CLUSTER_BUDGET)

    per_signal = {
        "turns": turns_score,
        "tokens": tokens_score,
        "clusters": clusters_score,
    }
    composite = max(per_signal.values())
    driver = next(name for name in _DRIVER_PRIORITY if per_signal[name] == composite)

    return CognitiveLoad(
        score=composite,
        driver=driver,
        turns_score=turns_score,
        tokens_score=tokens_score,
        clusters_score=clusters_score,
        turns_used=turns_used,
        tokens_used=tokens_used,
        clusters_count=clusters_count,
    )
