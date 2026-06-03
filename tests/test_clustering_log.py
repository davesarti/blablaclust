"""Unit tests for clustering_log + initial_clustering's log integration.

Logging writes to a JSONL file at module load time; tests redirect that path
to a tmp file so they don't pollute the real ``logs/`` directory.
"""

import json

import numpy as np
import pytest

import src.logger as logger
from src.engine.initial_clustering import (
    KMEANS_BACKEND,
    KMEANS_RANDOM_STATE,
    USE_GMM,
    initial_clustering,
)
from src.models import DataPoint


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    """Redirect logger._clustering_log_path to a tmp file for the test."""
    target = tmp_path / "clustering_runs.jsonl"
    monkeypatch.setattr(logger, "_clustering_log_path", target)
    return target


def _read_lines(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# log_clustering_run direct tests
# ---------------------------------------------------------------------------


def test_log_writes_all_fields(log_path):
    logger.log_clustering_run(
        session_id="sess-1",
        k=4,
        backend="kmeans",
        seed=42,
        n_points=120,
        silhouette=0.37,
        turn_number=0,
    )

    entries = _read_lines(log_path)
    assert len(entries) == 1
    e = entries[0]
    assert e["session_id"] == "sess-1"
    assert e["k"] == 4
    assert e["backend"] == "kmeans"
    assert e["seed"] == 42
    assert e["n_points"] == 120
    assert e["silhouette"] == 0.37
    assert e["turn_number"] == 0
    assert "timestamp" in e


def test_log_appends_one_line_per_call(log_path):
    for i in range(3):
        logger.log_clustering_run(
            session_id=f"sess-{i}",
            k=2,
            backend="kmeans",
            seed=42,
            n_points=10,
            silhouette=None,
            turn_number=0,
        )

    assert len(_read_lines(log_path)) == 3


def test_log_handles_none_silhouette(log_path):
    logger.log_clustering_run(
        session_id="s",
        k=1,
        backend="kmeans",
        seed=42,
        n_points=5,
        silhouette=None,
        turn_number=0,
    )
    assert _read_lines(log_path)[0]["silhouette"] is None


def test_log_swallows_io_errors(monkeypatch):
    """A broken file path must not raise — clustering runs come first."""
    from pathlib import Path
    bad = Path("/no/such/dir/forbidden.jsonl")
    monkeypatch.setattr(logger, "_clustering_log_path", bad)
    # Should NOT raise even though the parent directory does not exist.
    logger.log_clustering_run(
        session_id="s", k=2, backend="kmeans", seed=42,
        n_points=10, silhouette=0.5, turn_number=0,
    )


# ---------------------------------------------------------------------------
# initial_clustering integration
# ---------------------------------------------------------------------------


def _make_points(n: int, dim: int = 4) -> list[DataPoint]:
    """Build n DataPoints with deterministic well-separated embeddings."""
    rng = np.random.default_rng(0)
    pts = []
    for i in range(n):
        # Three loose blobs at integer offsets — enough for silhouette > 0.
        center = (i % 3) * 10.0
        emb = (rng.standard_normal(dim) * 0.1 + center).astype(np.float32).tolist()
        dp = DataPoint()
        dp.id = f"p{i}"
        dp.embedding = emb
        dp.data = {"text": f"point {i}"}
        pts.append(dp)
    return pts


def test_initial_clustering_logs_run(log_path):
    points = _make_points(30)
    initial_clustering(points, k=3, session_id="sess-int", turn_number=0)

    entries = _read_lines(log_path)
    assert len(entries) == 1
    e = entries[0]
    assert e["session_id"] == "sess-int"
    assert e["k"] == 3
    # Backend is "gmm" when USE_GMM=True (default), "kmeans" otherwise.
    expected_backend = "gmm" if USE_GMM else KMEANS_BACKEND
    assert e["backend"] == expected_backend
    assert e["seed"] == KMEANS_RANDOM_STATE
    assert e["n_points"] == 30
    assert e["turn_number"] == 0
    assert isinstance(e["silhouette"], float)
    assert -1.0 <= e["silhouette"] <= 1.0


def test_initial_clustering_logs_none_silhouette_when_k1(log_path):
    points = _make_points(10)
    initial_clustering(points, k=1, session_id="sess-k1", turn_number=0)

    entries = _read_lines(log_path)
    assert len(entries) == 1
    assert entries[0]["silhouette"] is None
    assert entries[0]["k"] == 1


def test_initial_clustering_logs_each_call(log_path):
    points = _make_points(20)
    initial_clustering(points, k=2, session_id="s1", turn_number=0)
    initial_clustering(points, k=3, session_id="s2", turn_number=0)

    entries = _read_lines(log_path)
    assert len(entries) == 2
    assert entries[0]["k"] == 2 and entries[0]["session_id"] == "s1"
    assert entries[1]["k"] == 3 and entries[1]["session_id"] == "s2"
