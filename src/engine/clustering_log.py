"""Structured JSONL logging for clustering runs.

One JSON object per line is appended to ``logs/clustering_runs.jsonl`` each
time ``initial_clustering`` produces a clustering — initial run, or one
triggered by a split. Mirrors the shape of ``src.logger.log_llm_call`` so
this helper can be folded into ``src/logger.py`` later (P5 owns that file).

Logging is best-effort: any IO error is swallowed so the clustering run
never aborts because the log file can't be written.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.logger import LOGS_DIR

_path: Path = LOGS_DIR / "clustering_runs.jsonl"


def log_clustering_run(
    session_id: str,
    k: int,
    backend: str,
    seed: int,
    n_points: int,
    silhouette: float | None,
    turn_number: int,
) -> None:
    """Append one line describing a single clustering run.

    Args:
        session_id: ChatSession the clustering belongs to.
        k: Number of clusters produced.
        backend: Algorithm name (e.g. ``"kmeans"``).
        seed: Random seed used by the algorithm — for reproducibility.
        n_points: Number of points actually clustered (after dropping ones
            without embeddings).
        silhouette: Mean silhouette score in [-1, 1], or ``None`` when the
            score is undefined (``k < 2`` or ``k >= n_points``).
        turn_number: Turn the clustering snapshot was recorded at.
    """
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
        with _path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        # Logging must never abort a clustering run. A missing log line is
        # acceptable; a crashed clustering is not.
        pass
