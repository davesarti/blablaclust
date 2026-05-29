"""Unit tests for the generalization mapping (src/engine/generalization.py).

Pure geometry on synthetic vectors — no embedding model, no DB. Three well-
separated 2-D blobs make the nearest-centroid assignment unambiguous and easy
to reason about.
"""

import numpy as np
import pytest

from src.engine.generalization import (
    assign_nearest,
    build_centroids,
    centroids_from_snapshot,
)


# ---------------------------------------------------------------------------
# build_centroids
# ---------------------------------------------------------------------------


def test_build_centroids_means_per_cluster():
    embeddings = np.array([
        [0.0, 0.0],
        [2.0, 0.0],   # cluster A → mean (1, 0)
        [10.0, 10.0],
        [12.0, 10.0],  # cluster B → mean (11, 10)
    ])
    labels = ["A", "A", "B", "B"]
    ids, centroids = build_centroids(embeddings, labels)

    assert ids == ["A", "B"]                      # sorted, deterministic
    np.testing.assert_allclose(centroids[0], [1.0, 0.0])
    np.testing.assert_allclose(centroids[1], [11.0, 10.0])


def test_build_centroids_single_point_cluster():
    ids, centroids = build_centroids(np.array([[5.0, 5.0]]), ["solo"])
    assert ids == ["solo"]
    np.testing.assert_allclose(centroids[0], [5.0, 5.0])


def test_build_centroids_order_independent():
    """Shuffling the rows must not change the centroids (ids are sorted)."""
    emb = np.array([[0.0, 0.0], [10.0, 0.0], [2.0, 0.0]])
    a = build_centroids(emb, ["x", "y", "x"])
    b = build_centroids(emb[::-1], ["x", "y", "x"][::-1])
    assert a[0] == b[0]
    np.testing.assert_allclose(a[1], b[1])


def test_build_centroids_rejects_empty():
    with pytest.raises(ValueError):
        build_centroids(np.empty((0, 3)), [])


def test_build_centroids_rejects_length_mismatch():
    with pytest.raises(ValueError):
        build_centroids(np.array([[1.0, 2.0]]), ["a", "b"])


# ---------------------------------------------------------------------------
# assign_nearest
# ---------------------------------------------------------------------------


def test_assign_nearest_clean_separation():
    centroid_ids = ["A", "B"]
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    queries = np.array([
        [1.0, 1.0],     # → A
        [9.0, 9.0],     # → B
        [0.5, 0.0],     # → A
        [11.0, 12.0],   # → B
    ])
    assert assign_nearest(queries, centroid_ids, centroids) == ["A", "B", "A", "B"]


def test_assign_nearest_picks_closest_of_many():
    centroid_ids = ["red", "green", "blue"]
    centroids = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    queries = np.array([[4.6, 0.0], [0.0, 4.9], [0.1, 0.1]])
    assert assign_nearest(queries, centroid_ids, centroids) == ["green", "blue", "red"]


def test_assign_nearest_rejects_dim_mismatch():
    with pytest.raises(ValueError):
        assign_nearest(np.array([[1.0, 2.0, 3.0]]), ["A"], np.array([[1.0, 2.0]]))


def test_assign_nearest_rejects_empty_queries():
    with pytest.raises(ValueError):
        assign_nearest(np.empty((0, 2)), ["A"], np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# centroids_from_snapshot
# ---------------------------------------------------------------------------


def test_centroids_from_snapshot_uses_hard_cluster():
    point_embeddings = {
        "p1": [0.0, 0.0],
        "p2": [2.0, 0.0],
        "p3": [10.0, 10.0],
    }
    snapshot = {
        "p1": {"A": 0.9, "B": 0.1},   # hard → A
        "p2": {"A": 0.6, "B": 0.4},   # hard → A
        "p3": {"A": 0.2, "B": 0.8},   # hard → B
    }
    ids, centroids = centroids_from_snapshot(point_embeddings, snapshot)
    assert ids == ["A", "B"]
    np.testing.assert_allclose(centroids[0], [1.0, 0.0])   # mean of p1, p2
    np.testing.assert_allclose(centroids[1], [10.0, 10.0])  # p3 alone


def test_centroids_from_snapshot_skips_points_without_embedding():
    point_embeddings = {"p1": [0.0, 0.0]}            # p2 has no embedding
    snapshot = {
        "p1": {"A": 1.0},
        "p2": {"A": 1.0},
    }
    ids, centroids = centroids_from_snapshot(point_embeddings, snapshot)
    assert ids == ["A"]
    np.testing.assert_allclose(centroids[0], [0.0, 0.0])


def test_centroids_from_snapshot_raises_when_no_usable_points():
    with pytest.raises(ValueError):
        centroids_from_snapshot({}, {"p1": {"A": 1.0}})


# ---------------------------------------------------------------------------
# end-to-end: build then assign (the actual generalization flow)
# ---------------------------------------------------------------------------


def test_round_trip_training_then_holdout():
    """Train centroids on labelled points, then a held-out point lands in the
    cluster whose members it resembles — the whole point of the feature."""
    train = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],      # topic A near origin
        [20.0, 20.0], [21.0, 20.0], [20.0, 21.0],  # topic B far away
    ])
    labels = ["A", "A", "A", "B", "B", "B"]
    ids, centroids = build_centroids(train, labels)

    held_out = np.array([[0.5, 0.5], [20.5, 20.5]])
    assert assign_nearest(held_out, ids, centroids) == ["A", "B"]
