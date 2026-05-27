"""Tests for the soft-assignment softmax in initial_clustering.

These guard the temperature scaling: without it, softmax(-d²) over realistic
(unit-norm, marginally separated) embeddings collapses to ~uniform — every point
gets ~1/k and none looks more certain than any other, which makes the
uncertainty signal useless. The temperature is a fraction of the data's own mean
squared distance, so it both sharpens the split and stays scale-invariant.
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
    _, assignments = initial_clustering(points, k=3, session_id="s", turn_number=0)

    max_probs = _max_probs(assignments)
    # Uniform would put every point at 1/3 ≈ 0.333. The temperature must lift the
    # winning cluster well clear of that, otherwise the uncertainty signal is noise.
    assert max_probs.mean() > 0.5
    # And the distribution must actually spread — core vs. boundary points differ.
    assert np.percentile(max_probs, 90) - np.percentile(max_probs, 10) > 0.1


def test_soft_assignments_scale_invariant(tmp_path, monkeypatch):
    import src.logger as logger
    monkeypatch.setattr(logger, "_clustering_log_path", tmp_path / "runs.jsonl")

    points = _marginal_points()
    _, assignments = initial_clustering(points, k=3, session_id="s", turn_number=0)
    baseline = np.sort(_max_probs(assignments))

    # Multiplying every embedding by a large constant scales squared distances by
    # its square. A fixed-temperature softmax would saturate to 1.0; ours divides
    # by the data's own scale, so the probabilities are unchanged.
    scaled_points: list[DataPoint] = []
    for p in points:
        dp = DataPoint()
        dp.id = p.id
        dp.embedding = (np.array(p.embedding) * 1000.0).tolist()
        dp.data = p.data
        scaled_points.append(dp)
    _, scaled_assignments = initial_clustering(scaled_points, k=3, session_id="s", turn_number=0)
    scaled = np.sort(_max_probs(scaled_assignments))

    assert np.allclose(baseline, scaled, atol=1e-3)


def test_probabilities_sum_to_one_per_point(tmp_path, monkeypatch):
    import src.logger as logger
    monkeypatch.setattr(logger, "_clustering_log_path", tmp_path / "runs.jsonl")

    points = _marginal_points()
    _, assignments = initial_clustering(points, k=4, session_id="s", turn_number=0)

    by_point: dict[str, float] = defaultdict(float)
    for a in assignments:
        by_point[a.data_point_id] += a.probability
    assert all(abs(total - 1.0) < 1e-5 for total in by_point.values())
