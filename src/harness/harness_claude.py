"""Anthropic Claude provider implementation.

Mirrors the shape of harness_openai.py and harness_openrouter.py so Claude is a
first-class sibling rather than the implicit default embedded in the shared
harness module.

Public surface (same names as the other two providers):
    DEFAULT_MODEL           — str
    call_claude             — sync call
    call_claude_async       — async call
    call_claude_batch       — batch (parallel async)
    count_tokens            — Anthropic-specific token-counting endpoint
    extract_usage           — parse Anthropic Message.usage → dict
    estimate_cost_usd_claude — pricing for Claude models
"""

import asyncio
import os
import random
import time
from typing import Any

from .harness import (
    BASE_DELAY,
    DRY_RUN,
    LLMResponse,
    MAX_DELAY,
    MAX_RETRIES,
    _DRY_RUN_OUTPUT,
)

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Pricing per million tokens (input, output, cache).
_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6":         {"input": 3.0,  "output": 15.0, "cache_read": 0.30,  "cache_write": 3.75},
    "claude-opus-4-7":           {"input": 15.0, "output": 75.0, "cache_read": 1.50,  "cache_write": 18.75},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0,  "cache_read": 0.08,  "cache_write": 1.0},
}


# ---------------------------------------------------------------------------
# Transient-error detection (Anthropic-specific)
# ---------------------------------------------------------------------------

def _is_transient_error(exc: Exception) -> bool:
    from anthropic import APIConnectionError, APIStatusError, RateLimitError
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {429, 500, 502, 503, 504}
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dry_run_response(model: str) -> LLMResponse:
    return LLMResponse(
        text=_DRY_RUN_OUTPUT,
        usage={"input_tokens": 0, "output_tokens": 0,
               "cache_read_tokens": 0, "cache_creation_tokens": 0},
        model=model,
    )


def extract_usage(message: Any) -> dict[str, int]:
    """Parse an Anthropic Message.usage object into a standard usage dict."""
    u = message.usage
    return {
        "input_tokens":          getattr(u, "input_tokens", 0) or 0,
        "output_tokens":         getattr(u, "output_tokens", 0) or 0,
        "cache_read_tokens":     getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


def estimate_cost_usd_claude(
    usage: dict[str, int], model: str = DEFAULT_MODEL
) -> float:
    rates = _PRICING.get(model, _PRICING["claude-sonnet-4-6"])
    per_m = 1_000_000
    return (
        usage.get("input_tokens", 0)          * rates["input"]              / per_m
        + usage.get("output_tokens", 0)        * rates["output"]             / per_m
        + usage.get("cache_read_tokens", 0)    * rates.get("cache_read", 0)  / per_m
        + usage.get("cache_creation_tokens", 0) * rates.get("cache_write", 0) / per_m
    )


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(
    messages: list[dict[str, str]],
    system: str,
    model: str = DEFAULT_MODEL,
) -> int:
    """Call Anthropic's token-counting endpoint. Returns 0 in dry-run mode."""
    if DRY_RUN:
        return 0
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.count_tokens(
        model=model,
        system=system,
        messages=messages,
    )
    return response.input_tokens


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def call_claude(
    messages: list[dict[str, str]],
    system: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> LLMResponse:
    if DRY_RUN:
        return _make_dry_run_response(model)

    import anthropic
    client = anthropic.Anthropic()

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            return LLMResponse(
                text=msg.content[0].text,
                usage=extract_usage(msg),
                model=model,
            )
        except Exception as exc:
            if not _is_transient_error(exc):
                raise
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            jitter = random.uniform(0, BASE_DELAY)
            delay = min(BASE_DELAY * (2 ** attempt) + jitter, MAX_DELAY)
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


async def call_claude_async(
    messages: list[dict[str, str]],
    system: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> LLMResponse:
    if DRY_RUN:
        return _make_dry_run_response(model)

    import anthropic
    client = anthropic.AsyncAnthropic()

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            msg = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            return LLMResponse(
                text=msg.content[0].text,
                usage=extract_usage(msg),
                model=model,
            )
        except Exception as exc:
            if not _is_transient_error(exc):
                raise
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            jitter = random.uniform(0, BASE_DELAY)
            delay = min(BASE_DELAY * (2 ** attempt) + jitter, MAX_DELAY)
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


def call_claude_batch(
    requests: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> list[LLMResponse | Exception]:
    async def _gather() -> list[LLMResponse | Exception]:
        tasks = [
            call_claude_async(
                messages=r["messages"],
                system=r.get("system", ""),
                model=model,
                max_tokens=r.get("max_tokens", 2048),
            )
            for r in requests
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    return asyncio.run(_gather())
