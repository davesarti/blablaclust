import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

# If STRICT_MODE=1, unexpected deviations raise an exception (useful in CI/tests).
# If STRICT_MODE=0, they are logged as warnings and execution continues (production).
STRICT = os.getenv("STRICT_MODE", "0") == "1"

# All log files are written to the logs/ directory at the project root.
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# One JSONL file for LLM calls — one JSON object per line.
_llm_log_path = LOGS_DIR / "llm_calls.jsonl"

# One JSONL file for clustering runs — one JSON object per line.
_clustering_log_path: Path = LOGS_DIR / "clustering_runs.jsonl"

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)


class UnexpectedDeviation(Exception):
    pass


def deviation(msg: str, **kwargs) -> None:
    #Call instead of print() whenever something unexpected happens in the system.
    if STRICT:
        raise UnexpectedDeviation(f"{msg} | {kwargs}")
    log.warning(msg, extra={"details": kwargs})


def log_llm_call(
    session_id: str,
    prompt_name: str,
    prompt_hash: str,
    usage: dict,
    cost_usd: float,
    *,
    turn_number: int | None = None,
    model: str | None = None,
) -> None:
    #Append one line to logs/llm_calls.jsonl for every LLM call.
    # Best-effort: an audit-log failure (bad usage payload, disk error, …) must
    # never propagate and break the LLM call it is recording.
    try:
        entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "prompt_name": prompt_name,
            # Hash of the prompt file — lets us verify which version of the prompt was used.
            "prompt_hash": prompt_hash,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_tokens", 0),
            "cache_creation_tokens": usage.get("cache_creation_tokens", 0),
            "cost_usd": cost_usd,
        }
        if turn_number is not None:
            entry["turn_number"] = turn_number
        if model is not None:
            entry["model"] = model
        with _llm_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:  # noqa: BLE001 — audit logging is non-critical
        log.warning("log_llm_call: failed to record LLM call (%s)", exc)


def log_clustering_run(
    session_id: str,
    k: int,
    backend: str,
    seed: int,
    n_points: int,
    silhouette: float | None,
    turn_number: int,
) -> None:
    #Append one line to logs/clustering_runs.jsonl for every clustering run.
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "k": k,
        "backend": backend,
        "seed": seed,
        "n_points": n_points,
        "silhouette": silhouette,
        "turn_number": turn_number,
    }
    try:
        with _clustering_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass
