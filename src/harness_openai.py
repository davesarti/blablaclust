import asyncio
import os
from typing import Any

import tiktoken
from openai import AsyncOpenAI, OpenAI
from openai import APIConnectionError, APIStatusError, RateLimitError

from src.harness import (
    BASE_DELAY,
    MAX_DELAY,
    MAX_RETRIES,
    ConversationContext,
    DRY_RUN,
    LLMResponse,
    _DRY_RUN_OUTPUT,
    _retry_async,
    _retry_sync,
    hash_prompt,
    load_prompt,
    render_prompt,
)

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":      {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
}


# ---------------------------------------------------------------------------
# Retry internals
# ---------------------------------------------------------------------------

def _is_transient_error_openai(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {429, 500, 502, 503, 504}
    return False


def _retry_sync_openai(fn, max_retries: int = MAX_RETRIES,
                       base_delay: float = BASE_DELAY,
                       max_delay: float = MAX_DELAY) -> Any:
    import random, time
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if not _is_transient_error_openai(exc):
                raise
            last_exc = exc
            if attempt == max_retries:
                break
            jitter = random.uniform(0, base_delay)
            delay = min(base_delay * (2 ** attempt) + jitter, max_delay)
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


async def _retry_async_openai(fn, max_retries: int = MAX_RETRIES,
                               base_delay: float = BASE_DELAY,
                               max_delay: float = MAX_DELAY) -> Any:
    import random
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            if not _is_transient_error_openai(exc):
                raise
            last_exc = exc
            if attempt == max_retries:
                break
            jitter = random.uniform(0, base_delay)
            delay = min(base_delay * (2 ** attempt) + jitter, max_delay)
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_openai_messages(
    messages: list[dict[str, str]],
    system: str,
) -> list[dict[str, str]]:
    if system:
        return [{"role": "system", "content": system}] + messages
    return list(messages)


def _make_dry_run_response(model: str) -> LLMResponse:
    return LLMResponse(
        text=_DRY_RUN_OUTPUT,
        usage={"input_tokens": 0, "output_tokens": 0,
               "cache_read_tokens": 0, "cache_creation_tokens": 0},
        model=model,
    )


# ---------------------------------------------------------------------------
# Token counting & cost
# ---------------------------------------------------------------------------

def count_tokens_openai(
    messages: list[dict[str, str]],
    system: str,
    model: str = DEFAULT_MODEL,
) -> int:
    if DRY_RUN:
        return 0
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    full = _build_openai_messages(messages, system)
    return sum(len(enc.encode(m["content"])) for m in full)


def extract_usage_openai(completion: Any) -> dict[str, int]:
    u = completion.usage
    return {
        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }


def estimate_cost_usd_openai(usage: dict[str, int], model: str = DEFAULT_MODEL) -> float:
    rates = _PRICING.get(model, _PRICING["gpt-4o"])
    per_m = 1_000_000
    return (
        usage.get("input_tokens", 0) * rates["input"] / per_m
        + usage.get("output_tokens", 0) * rates["output"] / per_m
    )


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def call_gpt(
    messages: list[dict[str, str]],
    system: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> LLMResponse:
    if DRY_RUN:
        return _make_dry_run_response(model)

    import time as _time

    client = OpenAI()
    full_messages = _build_openai_messages(messages, system)

    def _call():
        return client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=full_messages,
        )

    # Retry up to MAX_RETRIES extra times when the API returns empty content
    # (content is None or ""). OpenRouter/gpt-4o-mini occasionally returns an
    # empty body that is not flagged as a transient error by status code.
    last_completion = None
    for attempt in range(MAX_RETRIES + 1):
        last_completion = _retry_sync_openai(_call)
        content = last_completion.choices[0].message.content
        if content:
            return LLMResponse(
                text=content,
                usage=extract_usage_openai(last_completion),
                model=model,
            )
        if attempt < MAX_RETRIES:
            delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
            _time.sleep(delay)

    raise ValueError(
        f"LLM returned empty content after {MAX_RETRIES + 1} attempts "
        f"(finish_reason={last_completion.choices[0].finish_reason!r}, model={model!r})"
    )


async def call_gpt_async(
    messages: list[dict[str, str]],
    system: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> LLMResponse:
    if DRY_RUN:
        return _make_dry_run_response(model)

    client = AsyncOpenAI()
    full_messages = _build_openai_messages(messages, system)

    async def _call():
        return await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=full_messages,
        )

    last_completion = None
    for attempt in range(MAX_RETRIES + 1):
        last_completion = await _retry_async_openai(_call)
        content = last_completion.choices[0].message.content
        if content:
            return LLMResponse(
                text=content,
                usage=extract_usage_openai(last_completion),
                model=model,
            )
        if attempt < MAX_RETRIES:
            await asyncio.sleep(min(BASE_DELAY * (2 ** attempt), MAX_DELAY))

    raise ValueError(
        f"LLM returned empty content after {MAX_RETRIES + 1} attempts "
        f"(finish_reason={last_completion.choices[0].finish_reason!r}, model={model!r})"
    )


def call_gpt_batch(
    requests: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> list[LLMResponse | Exception]:
    async def _gather() -> list[LLMResponse | Exception]:
        tasks = [
            call_gpt_async(
                messages=r["messages"],
                system=r.get("system", ""),
                model=model,
                max_tokens=r.get("max_tokens", 2048),
            )
            for r in requests
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    return asyncio.run(_gather())
