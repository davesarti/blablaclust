"""OpenRouter harness (OpenAI-compatible API, different base URL + key).

Mirrors harness.py (Claude) and harness_openai.py (OpenAI) so OpenRouter is a
first-class provider. Reuses the shared primitives (LLMResponse, DRY_RUN,
retry tuning) from harness.py and the OpenAI-compatible retry/error helpers
from harness_openai.py, since OpenRouter speaks the OpenAI chat-completions
protocol via the `openai` SDK pointed at a custom base_url.
"""

import asyncio
import os
from typing import Any

import tiktoken
from openai import AsyncOpenAI, OpenAI

from src.harness import (
    DRY_RUN,
    LLMResponse,
    _make_dry_run_response,
)
from src.harness_openai import (
    _build_openai_messages,
    _retry_async_openai,
    _retry_sync_openai,
    extract_usage_openai,
)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free"
)
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "http://localhost")

# Pricing per million tokens (input, output). OpenRouter hosts many models with
# wildly different rates, so look up by the fully-qualified model slug. Unknown
# models fall back to free (0.0) — update this table as paid models are adopted.
_PRICING: dict[str, dict[str, float]] = {
    "deepseek/deepseek-v4-flash:free":   {"input": 0.0,  "output": 0.0},
    "deepseek/deepseek-chat":            {"input": 0.14, "output": 0.28},
    "openai/gpt-4o":                     {"input": 2.50, "output": 10.0},
    "anthropic/claude-3.5-sonnet":       {"input": 3.0,  "output": 15.0},
    # Google Gemini via OpenRouter
    "google/gemini-2.5-flash":           {"input": 0.30, "output": 2.50},
    "google/gemini-2.5-flash-lite":      {"input": 0.10, "output": 0.40},
}
_DEFAULT_RATES = {"input": 0.0, "output": 0.0}


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def _client_kwargs() -> dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    base_url = os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)
    site_url = os.environ.get("OPENROUTER_SITE_URL", OPENROUTER_SITE_URL)
    return {
        "api_key": api_key,
        "base_url": base_url,
        "default_headers": {"HTTP-Referer": site_url},
    }


# ---------------------------------------------------------------------------
# Token counting & cost
# ---------------------------------------------------------------------------

def count_tokens_openrouter(
    messages: list[dict[str, str]],
    system: str,
    model: str = OPENROUTER_MODEL,
) -> int:
    """Approximate token count. OpenRouter models rarely have a tiktoken
    encoding, so fall back to cl100k_base — good enough for budget checks."""
    if DRY_RUN:
        return 0
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    full = _build_openai_messages(messages, system)
    return sum(len(enc.encode(m["content"])) for m in full)


def estimate_cost_usd_openrouter(
    usage: dict[str, int], model: str = OPENROUTER_MODEL
) -> float:
    rates = _PRICING.get(model, _DEFAULT_RATES)
    per_m = 1_000_000
    return (
        usage.get("input_tokens", 0) * rates["input"] / per_m
        + usage.get("output_tokens", 0) * rates["output"] / per_m
    )


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def call_openrouter(
    messages: list[dict[str, str]],
    system: str,
    model: str = OPENROUTER_MODEL,
    max_tokens: int = 2048,
) -> LLMResponse:
    if DRY_RUN:
        return _make_dry_run_response(model)

    client = OpenAI(**_client_kwargs())
    full_messages = _build_openai_messages(messages, system)

    def _call():
        return client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=full_messages,
        )

    completion = _retry_sync_openai(_call)
    content = completion.choices[0].message.content
    if content is None:
        raise ValueError(
            f"OpenRouter returned no text content "
            f"(finish_reason={completion.choices[0].finish_reason!r}, model={model!r})"
        )
    return LLMResponse(
        text=content,
        usage=extract_usage_openai(completion),
        model=model,
    )


async def call_openrouter_async(
    messages: list[dict[str, str]],
    system: str,
    model: str = OPENROUTER_MODEL,
    max_tokens: int = 2048,
) -> LLMResponse:
    if DRY_RUN:
        return _make_dry_run_response(model)

    client = AsyncOpenAI(**_client_kwargs())
    full_messages = _build_openai_messages(messages, system)

    async def _call():
        return await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=full_messages,
        )

    completion = await _retry_async_openai(_call)
    content = completion.choices[0].message.content
    if content is None:
        raise ValueError(
            f"OpenRouter returned no text content "
            f"(finish_reason={completion.choices[0].finish_reason!r}, model={model!r})"
        )
    return LLMResponse(
        text=content,
        usage=extract_usage_openai(completion),
        model=model,
    )


def call_openrouter_batch(
    requests: list[dict[str, Any]],
    model: str = OPENROUTER_MODEL,
) -> list[LLMResponse | Exception]:
    async def _gather() -> list[LLMResponse | Exception]:
        tasks = [
            call_openrouter_async(
                messages=r["messages"],
                system=r.get("system", ""),
                model=model,
                max_tokens=r.get("max_tokens", 2048),
            )
            for r in requests
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    return asyncio.run(_gather())
