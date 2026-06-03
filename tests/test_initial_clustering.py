"""Tests for soft-assignment probabilities in initial_clustering.

With GMM (the default backend), probabilities are native EM posteriors — no
softmax temperature hack.  With k-means fallback they use the temperature-scaled
softmax.  The tests here guard:
- probabilities are sharper than uniform (the whole point of soft assignments)
- probabilities sum to 1 per point
- silhouette is returned correctly and logged
- degenerate input doesn't crash
"""

from collections import defaultdict

import numpy as np

from src.engine.initial_clustering import initial_clustering
from src.models import DataPoint


def _marginal_points(n: int = 150, dim: int = 64, k: int = 3, seed: int = 1) -> list[DataPoint]:
    """Unit-norm embeddings with marginal cluster separation.

    Centers sit close together and per-point noise is large, so each point is
    only slightly closer to its own center than the others — the regime where a
    fixed-temperature softmax would be near-uniform.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, dim)) * 0.15
    points: list[DataPoint] = []
    for i in range(n):
        v = centers[i % k] + rng.standard_normal(dim) * 1.0
        v = v / np.linalg.norm(v)
        dp = DataPoint()
        dp.id = f"p{i}"
        dp.embedding = v.astype(np.float32).tolist()
        dp.data = {"text": f"point {i}"}
        points.append(dp)
    return points


def _max_probs(assignments) -> np.ndarray:
    """Per-point maximum soft-assignment probability."""
    by_point: dict[str, list[float]] = defaultdict(list)
    for a in assignments:
        by_point[a.data_point_id].append(a.probability)
    return np.array([max(v) for v in by_point.values()])


def test_soft_assignments_sharper_than_uniform(tmp_path, monkeypatch):
    import src.logger as logger
    monkeypatch.setattr(logger, "_clustering_log_path", tmp_path / "runs.jsonl")

    points = _marginal_points()
    _, assignments, _ = initial_clustering(points, k=3, session_id="s", turn_number=0)

    max_probs = _max_probs(assignments)
    # Uniform would put every point at 1/3 ≈ 0.333.  Both GMM (native posteriors)
    # and the k-means softmax fallback must lift the winning cluster well clear of
    # that baseline.  GMM on this synthetic data often pushes most points above 0.9;
    # k-means softmax does so too after temperature scaling.
    assert max_probs.mean() > 0.5
    # At least some points must be distinguishable from the rest — the
    # distribution must not be a spike at exactly 1.0 for every single point.
    # Both backends produce at least a handful of genuinely-ambiguous boundary
    # points, so the overall minimum must be below the 90th percentile.
    assert max_probs.min() < np.percentile(max_probs, 90)


def test_soft_assignments_backend_logged(tmp_path, monkeypatch):
    """The backend used (gmm or kmeans) must be written to the clustering log."""
    import json
    import src.logger as logger
    log_path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(logger, "_clustering_log_path", log_path)

    points = _marginal_points()
    initial_clustering(points, k=3, session_id="s", turn_number=0)

    logged = json.loads(log_path.read_text().splitlines()[-1])
    assert logged["backend"] in ("gmm", "kmeans")


def test_probabilities_sum_to_one_per_point(tmp_path, monkeypatch):
    import src.logger as logger
    monkeypatch.setattr(logger, "_clustering_log_path", tmp_path / "runs.jsonl")

    points = _marginal_points()
    _, assignments, _ = initial_clustering(points, k=4, session_id="s", turn_number=0)

    by_point: dict[str, float] = defaultdict(float)
    for a in assignments:
        by_point[a.data_point_id] += a.probability
    assert all(abs(total - 1.0) < 1e-5 for total in by_point.values())


def test_returns_silhouette_matching_the_log(tmp_path, monkeypatch):
    """The silhouette returned to the caller must be the same value written to
    the structured log — single source of truth, one k-means run."""
    import json
    import src.logger as logger
    log_path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(logger, "_clustering_log_path", log_path)

    points = _marginal_points()
    _, _, silhouette = initial_clustering(points, k=3, session_id="s", turn_number=0)

    assert isinstance(silhouette, float)
    assert -1.0 <= silhouette <= 1.0

    logged = json.loads(log_path.read_text().splitlines()[-1])["silhouette"]
    assert logged == silhouette


def test_returns_none_silhouette_for_k1(tmp_path, monkeypatch):
    """Silhouette is undefined for k < 2 → the function returns None (no crash)."""
    import src.logger as logger
    monkeypatch.setattr(logger, "_clustering_log_path", tmp_path / "runs.jsonl")

    points = _marginal_points()
    _, _, silhouette = initial_clustering(points, k=1, session_id="s", turn_number=0)
    assert silhouette is None


def test_degenerate_input_does_not_crash_clustering(tmp_path, monkeypatch):
    """All-identical embeddings make silhouette_score raise — but a failed
    diagnostic must never abort the clustering run. Expect: no exception,
    clusters + assignments still produced, silhouette falls back to None."""
    import src.logger as logger
    monkeypatch.setattr(logger, "_clustering_log_path", tmp_path / "runs.jsonl")

    # 20 points, every embedding identical → k-means yields one populated
    # cluster, so silhouette_score raises ValueError ("number of labels is 1").
    points = []
    for i in range(20):
        dp = DataPoint()
        dp.id = f"d{i}"
        dp.embedding = [0.5, 0.5, 0.5, 0.5]
        dp.data = {"text": f"identical {i}"}
        points.append(dp)

    clusters, assignments, silhouette = initial_clustering(
        points, k=3, session_id="s", turn_number=0
    )

    assert len(clusters) == 3            # clustering still ran
    assert len(assignments) == 20 * 3    # full snapshot still written
    assert silhouette is None            # diagnostic degraded gracefully
