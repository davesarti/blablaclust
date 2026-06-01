import asyncio
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
DRY_RUN = os.environ.get("HARNESS_DRY_RUN", "false").lower() == "true"
MAX_RETRIES = int(os.environ.get("HARNESS_MAX_RETRIES", "4"))
BASE_DELAY = float(os.environ.get("HARNESS_BASE_DELAY", "1.0"))
MAX_DELAY = float(os.environ.get("HARNESS_MAX_DELAY", "60.0"))
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS_PER_TURN", "8000"))

# Pricing per million tokens (input, output) — update when Anthropic changes rates
_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-opus-4-7":   {"input": 15.0, "output": 75.0, "cache_read": 1.50, "cache_write": 18.75},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0, "cache_read": 0.08, "cache_write": 1.0},
}

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

#restituisce come stringa il file di prompt specificato
def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def hash_prompt(name: str) -> str:
    content = load_prompt(name)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# è un sistema di template per i prompt
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
        return match.group(1).strip()
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

    This recovers the common case. It can still misread a quoted phrase that is
    immediately followed by a colon inside a value, so callers must keep treating
    a parse failure as recoverable (best-effort).
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
            # Preserve existing escape sequences verbatim (\" \\ \n ...).
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
                out.append(c)          # structural terminator — keep it
                in_string = False
            else:
                out.append('\\"')      # content quote — escape it
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def loads_llm_json(text: str) -> Any:
    """Parse JSON from an LLM response, tolerating unescaped double quotes.

    Strict parse first (the common path); only if that fails do we escape stray
    quotes and retry. Raises json.JSONDecodeError if still unrecoverable, so
    callers can fall back to placeholders/error handling.
    """
    raw = extract_json_text(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_escape_unescaped_quotes(raw))


# ---------------------------------------------------------------------------
# Retry internals
# ---------------------------------------------------------------------------
'''
Questa sezione gestisce gli errori temporanei dell'API per evitare che l'applicazione si blocchi 
in caso di problemi di rete o limiti di richieste (Rate Limit).
'''

'''
Controlla se un'eccezione lanciata dall'API di Anthropic è "temporanea" (come RateLimitError, APIConnectionError, o errori server come 429, 500, 502, ecc.). 
Se è temporanea, significa che la richiesta può essere riprovata e restituisce True.
'''
def _is_transient_error(exc: Exception) -> bool:
    from anthropic import APIConnectionError, APIStatusError, RateLimitError
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {429, 500, 502, 503, 504}
    return False

'''
Esegue una funzione sincrona fn e, se incontra un errore temporaneo, riprova l'esecuzione implementando un exponential backoff con jitter
Si ferma dopo MAX_RETRIES.
'''
def _retry_sync(fn, max_retries: int = MAX_RETRIES,
                base_delay: float = BASE_DELAY,
                max_delay: float = MAX_DELAY) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if not _is_transient_error(exc):
                raise
            last_exc = exc
            if attempt == max_retries:
                break
            jitter = random.uniform(0, base_delay)
            delay = min(base_delay * (2 ** attempt) + jitter, max_delay)
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]

'''
Uguale a _retry_sync ma per funzioni asincrone (defite con async def).
Utilizza await asyncio.sleep al posto di time.sleep.
PER IL MOMENTO LE TENIAMO ENTRAMBE PERCHÈ NON SO COME SARÀ STRUTTURATO IL RESTO DEL PROGETTO.
'''
async def _retry_async(fn, max_retries: int = MAX_RETRIES,
                       base_delay: float = BASE_DELAY,
                       max_delay: float = MAX_DELAY) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            if not _is_transient_error(exc):
                raise
            last_exc = exc
            if attempt == max_retries:
                break
            jitter = random.uniform(0, base_delay)
            delay = min(base_delay * (2 ** attempt) + jitter, max_delay)
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

'''
Crea una risposta fittizia senza chiamare l'API.
Viene utilizzata quando HARNESS_DRY_RUN=true per simulare una risposta senza spendere crediti API.
'''
def _make_dry_run_response(model: str) -> LLMResponse:
    return LLMResponse(
        text=_DRY_RUN_OUTPUT,
        usage={"input_tokens": 0, "output_tokens": 0,
               "cache_read_tokens": 0, "cache_creation_tokens": 0},
        model=model,
    )

'''
Chiama l'API di Anthropic in modo sincrono (bloccante). 
Utilizza _retry_sync per gestire automaticamente eventuali errori temporanei. 
Restituisce un oggetto Message di Anthropic. 
Se DRY_RUN è attivo restituisce il messaggio fittizio
Questa è la funzione "base" che fa la chiamata effettiva al modello.
'''
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

    def _call() -> anthropic.types.Message:
        return client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )

    msg = _retry_sync(_call)
    return LLMResponse(text=msg.content[0].text, usage=extract_usage(msg), model=model)

'''
Fa essenzialmente la stessa cosa di call_claude, ma in modo asincrono (non bloccante).
Questo permette di fare altre cose mentre si aspetta la risposta del modello.
'''
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

    async def _call() -> anthropic.types.Message:
        return await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )

    msg = await _retry_async(_call)
    return LLMResponse(text=msg.content[0].text, usage=extract_usage(msg), model=model)

'''
Esegue più chiamate a Claude in parallelo in modo sincrono (utilizza asyncio.run per eseguire la logica asincrona). 
È utile per interrogare il modello con vari prompt contemporaneamente in modo efficiente. 
Restituisce una lista di risposte (o Eccezioni se qualcosa va storto).
NON PENSO CI SERVA A MOLTO!!!!!!!!!!!!! PER ORA LA TENIAMO COSÌ.
'''
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


# ---------------------------------------------------------------------------
# Token counting & cost  (Component 9)
# ---------------------------------------------------------------------------

'''
Chiama l'endpoint specifico di Anthropic messages.count_tokens per sapere esattamente quanti "token" (unità di testo) di input sono presenti nella conversazione che si sta per inviare, 
permettendo di gestire i limiti di contesto. 
In modalità dry-run restituisce sempre 0.
'''
def count_tokens(
    messages: list[dict[str, str]],
    system: str,
    model: str = DEFAULT_MODEL,
) -> int:
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

'''
Prende l'oggetto Message di Anthropic e restituisce un dizionario con i conteggi dei token di input/output.
'''
def extract_usage(message: "anthropic.types.Message") -> dict[str, int]:
    u = message.usage
    return {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }

# stima il costo in dollari delle chiamate effettuate al modello
def estimate_cost_usd(usage: dict[str, int], model: str = DEFAULT_MODEL) -> float:
    rates = _PRICING.get(model, _PRICING["claude-sonnet-4-6"])
    per_m = 1_000_000
    return (
        usage.get("input_tokens", 0) * rates["input"] / per_m
        + usage.get("output_tokens", 0) * rates["output"] / per_m
        + usage.get("cache_read_tokens", 0) * rates["cache_read"] / per_m
        + usage.get("cache_creation_tokens", 0) * rates["cache_write"] / per_m
    )


# ---------------------------------------------------------------------------
# Provider-agnostic call
# ---------------------------------------------------------------------------

def call_llm(
    messages: list[dict[str, str]],
    system: str,
    max_tokens: int = 8192,
) -> LLMResponse:
    provider = os.environ.get("LLM_PROVIDER", "claude").lower()
    if provider == "openai":
        from src.harness_openai import call_gpt
        return call_gpt(messages, system, max_tokens=max_tokens)
    if provider == "openrouter":
        from src.harness_openrouter import call_openrouter
        return call_openrouter(messages, system, max_tokens=max_tokens)
    return call_claude(messages, system, max_tokens=max_tokens)


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
        # display can be a string (old format) or a Display dict
        # {"type": ..., "content": ..., "items": ...} (current SystemTurn format).
        # All LLM providers (Anthropic, OpenAI, Groq, OpenRouter) require
        # message content to be a plain string, so we always extract text here.
        display = system_turn.get("display")
        if isinstance(display, dict):
            text = display.get("content") or json.dumps(system_turn)
        elif isinstance(display, str):
            text = display
        else:
            text = json.dumps(system_turn)
        self.turns.append({"role": "assistant", "content": text})

    def build_messages(self, model: str = DEFAULT_MODEL) -> list[dict[str, str]]:
        messages = list(self.turns)
        # Token-based trimming requires the Anthropic SDK's count_tokens endpoint.
        # Skip it for other providers — their context windows are large enough that
        # the hard MAX_INPUT_TOKENS limit is unlikely to be hit in a normal session.
        provider = os.environ.get("LLM_PROVIDER", "claude").lower()
        if not DRY_RUN and provider == "claude" and len(messages) > 2:
            while count_tokens(messages, system="", model=model) > MAX_INPUT_TOKENS:
                if len(messages) <= 2:
                    break
                messages = messages[2:]  # drop oldest oracle+system pair
        return messages
