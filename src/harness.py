import asyncio
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import local
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DRY_RUN = os.environ.get("HARNESS_DRY_RUN", "false").lower() == "true"
MAX_RETRIES = int(os.environ.get("HARNESS_MAX_RETRIES", "4"))
BASE_DELAY = float(os.environ.get("HARNESS_BASE_DELAY", "1.0"))
MAX_DELAY = float(os.environ.get("HARNESS_MAX_DELAY", "60.0"))
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS_PER_TURN", "8000"))

_DRY_RUN_OUTPUT = json.dumps({
    "action": "no_change",
    "operations": [],
    "display": "[DRY RUN] This is a mock response. No API call was made.",
})


@dataclass
class LLMResponse:
    text: str
    usage: dict[str, int]
    model: str


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def hash_prompt(name: str) -> str:
    content = load_prompt(name)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def render_prompt(name: str, **kwargs: Any) -> str:
    template = load_prompt(name)
    return template.format(**kwargs)


def extract_json_text(text: str) -> str:
    """Return the raw JSON string from an LLM response.

    Handles three formats:
    1. Markdown fences: ```json ... ``` or ``` ... ```
    2. Prose-wrapped: "Here is the result: { ... }" — extracts the first {...} block
    3. Plain JSON (no wrapping needed)
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        extracted = match.group(1).strip()
        if extracted:  # skip empty code fences — fall through to {...} search
            return extracted
    # Fall back: extract the first top-level {...} block, ignoring surrounding prose
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        return match.group(1).strip()
    return text


def _escape_unescaped_quotes(s: str) -> str:
    """Escape stray double quotes inside JSON string values.

    LLMs routinely emit raw double quotes inside string values — inch marks
    (15.6"), quoted phrases ("best") — which is invalid JSON and makes json.loads
    raise "Expecting ',' delimiter". We walk the text tracking whether we are
    inside a string, and escape any `"` that is NOT a structural terminator. A
    terminator is a quote followed (after optional whitespace) by one of ``:,}]``
    or end-of-input; anything else is treated as content and escaped.
    """
    out: list[str] = []
    in_string = False
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if not in_string:
            out.append(c)
            if c == '"':
                in_string = True
            i += 1
        elif c == "\\":
            out.append(c)
            if i + 1 < n:
                out.append(s[i + 1])
                i += 2
            else:
                i += 1
        elif c == '"':
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j >= n or s[j] in ":,}]":
                out.append(c)
                in_string = False
            else:
                out.append('\\"')
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def loads_llm_json(text: str) -> Any:
    """Parse JSON from an LLM response, tolerating unescaped double quotes.

    Strict parse first (the common path); only if that fails do we escape stray
    quotes and retry. Raises json.JSONDecodeError if still unrecoverable.
    """
    raw = extract_json_text(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_escape_unescaped_quotes(raw))


# ---------------------------------------------------------------------------
# Retry primitives (provider-agnostic)
# ---------------------------------------------------------------------------

def _retry_sync(
    fn,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    is_transient=None,
) -> Any:
    """Retry *fn* up to *max_retries* extra times on transient errors.

    *is_transient* is an optional ``(exc) -> bool`` predicate. When provided,
    non-transient exceptions are re-raised immediately; when omitted every
    exception is treated as retryable (use with care).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if is_transient is not None and not is_transient(exc):
                raise
            last_exc = exc
            if attempt == max_retries:
                break
            jitter = random.uniform(0, base_delay)
            delay = min(base_delay * (2 ** attempt) + jitter, max_delay)
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


async def _retry_async(
    fn,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    is_transient=None,
) -> Any:
    """Async counterpart of _retry_sync."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            if is_transient is not None and not is_transient(exc):
                raise
            last_exc = exc
            if attempt == max_retries:
                break
            jitter = random.uniform(0, base_delay)
            delay = min(base_delay * (2 ** attempt) + jitter, max_delay)
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Dry-run helper (shared by all providers)
# ---------------------------------------------------------------------------

def _make_dry_run_response(model: str) -> LLMResponse:
    return LLMResponse(
        text=_DRY_RUN_OUTPUT,
        usage={"input_tokens": 0, "output_tokens": 0,
               "cache_read_tokens": 0, "cache_creation_tokens": 0},
        model=model,
    )


# ---------------------------------------------------------------------------
# Per-turn LLM cost accumulator
# ---------------------------------------------------------------------------

_turn_tracking = local()


def begin_turn_tracking() -> None:
    """Reset the per-turn accumulator at the start of a request turn.

    Call this once at the top of create_turn (or any other entry point that
    processes one oracle turn).  Every subsequent call_llm on the same thread
    will add its usage and cost into the accumulator automatically.
    """
    _turn_tracking.input_tokens = 0
    _turn_tracking.output_tokens = 0
    _turn_tracking.cost = 0.0
    _turn_tracking.active = True


def pop_turn_tracking() -> tuple[dict[str, int], float]:
    """Return and reset the accumulated (usage, cost_usd) for this turn.

    Returns a usage dict compatible with SystemTurn.token_usage and the total
    estimated cost in USD, covering *all* call_llm calls made since the last
    begin_turn_tracking() on this thread (f_output, cluster_naming,
    f_update_preferences, f_boundary_repair, semantic_clustering, …).
    """
    usage = {
        "input_tokens":  getattr(_turn_tracking, "input_tokens", 0),
        "output_tokens": getattr(_turn_tracking, "output_tokens", 0),
    }
    cost = getattr(_turn_tracking, "cost", 0.0)
    _turn_tracking.active = False
    _turn_tracking.input_tokens = 0
    _turn_tracking.output_tokens = 0
    _turn_tracking.cost = 0.0
    return usage, cost


def _accumulate_turn(usage: dict[str, int], cost: float) -> None:
    """Internal — add one call's usage+cost into the active accumulator."""
    if not getattr(_turn_tracking, "active", False):
        return
    _turn_tracking.input_tokens  = getattr(_turn_tracking, "input_tokens",  0) + usage.get("input_tokens",  0)
    _turn_tracking.output_tokens = getattr(_turn_tracking, "output_tokens", 0) + usage.get("output_tokens", 0)
    _turn_tracking.cost          = getattr(_turn_tracking, "cost", 0.0)        + cost


# ---------------------------------------------------------------------------
# Cost estimation (provider-agnostic dispatcher)
# ---------------------------------------------------------------------------

def _active_model() -> str:
    """The model slug the active provider will actually call.

    Reads LLM_PROVIDER and the matching model env var. Callers that already
    know the model (e.g. from LLMResponse.model) should pass it explicitly.
    """
    provider = os.environ.get("LLM_PROVIDER", "claude").lower()
    if provider == "openrouter":
        return os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free")
    if provider == "openai":
        return os.environ.get("OPENAI_MODEL", "gpt-4o")
    # default: claude
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def estimate_cost_usd(usage: dict[str, int], model: str | None = None) -> float:
    """Estimate cost in USD for a usage dict, dispatching to the active provider.

    Each provider module owns its own pricing table. Passing *model* explicitly
    overrides the active-model lookup (useful when the model is known from the
    LLMResponse rather than from env vars).
    """
    if model is None:
        model = _active_model()
    provider = os.environ.get("LLM_PROVIDER", "claude").lower()
    if provider == "openai":
        from src.harness_openai import estimate_cost_usd_openai
        return estimate_cost_usd_openai(usage, model)
    if provider == "openrouter":
        from src.harness_openrouter import estimate_cost_usd_openrouter
        return estimate_cost_usd_openrouter(usage, model)
    # default: claude
    from src.harness_claude import estimate_cost_usd_claude
    return estimate_cost_usd_claude(usage, model)


# ---------------------------------------------------------------------------
# Provider-agnostic call dispatcher
# ---------------------------------------------------------------------------

def call_llm(
    messages: list[dict[str, str]],
    system: str,
    max_tokens: int = 8192,
) -> LLMResponse:
    """Call the configured LLM provider. Controlled by LLM_PROVIDER env var.

    After every successful call the response usage is fed into the per-turn
    accumulator (if one was started with begin_turn_tracking).  This means
    ALL LLM calls that happen during a turn — f_output, cluster_naming,
    f_update_preferences, semantic_clustering, boundary repair, … — are
    automatically counted without each caller needing to track costs itself.
    """
    provider = os.environ.get("LLM_PROVIDER", "claude").lower()
    if provider == "openai":
        from src.harness_openai import call_gpt
        response = call_gpt(messages, system, max_tokens=max_tokens)
    elif provider == "openrouter":
        from src.harness_openrouter import call_openrouter
        response = call_openrouter(messages, system, max_tokens=max_tokens)
    else:
        # default: claude
        from src.harness_claude import call_claude
        response = call_claude(messages, system, max_tokens=max_tokens)
    # Accumulate usage into the active per-turn tracker (if any).
    # Pass response.model explicitly so the correct pricing row is used even
    # when the active-model env var differs from what the API actually served.
    _accumulate_turn(response.usage, estimate_cost_usd(response.usage, response.model))
    return response


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

@dataclass
class ConversationContext:
    session_id: str
    turns: list[dict[str, str]] = field(default_factory=list)
    _oracle_turns: list[dict] = field(default_factory=list, repr=False)

    def add_oracle_turn(self, oracle_turn: dict) -> None:
        text = oracle_turn.get("raw_text", json.dumps(oracle_turn))
        self.turns.append({"role": "user", "content": text})
        self._oracle_turns.append(oracle_turn)

    def add_system_turn(self, system_turn: dict) -> None:
        display = system_turn.get("display")
        if isinstance(display, dict):
            text = display.get("content") or json.dumps(system_turn)
        elif isinstance(display, str):
            text = display
        else:
            text = json.dumps(system_turn)
        self.turns.append({"role": "assistant", "content": text})

    def build_messages(self) -> list[dict[str, str]]:
        """Return the conversation as a list of role/content dicts.

        When the active provider is Claude, trims from the front if the token
        count exceeds MAX_INPUT_TOKENS (uses the Anthropic count_tokens endpoint).
        Trimming is skipped for other providers — their context windows are large
        enough that the hard limit is rarely hit in a normal session.
        """
        messages = list(self.turns)
        provider = os.environ.get("LLM_PROVIDER", "claude").lower()
        if not DRY_RUN and provider == "claude" and len(messages) > 2:
            from src.harness_claude import count_tokens, DEFAULT_MODEL
            model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
            while count_tokens(messages, system="", model=model) > MAX_INPUT_TOKENS:
                if len(messages) <= 2:
                    break
                messages = messages[2:]  # drop oldest oracle+system pair
        return messages
