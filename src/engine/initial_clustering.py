"""Initial k-means clustering on stored embeddings.

`initial_clustering` returns DB model objects ready to be persisted — callers
own the transaction. `silhouette_for_k` / `sweep_k` are diagnostics that help
the oracle pick a sensible number of clusters.
"""

import uuid

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.models import Cluster as DbCluster, DataPoint, SoftAssignment as DbSoftAssignment


def _embedding_matrix(data_points: list[DataPoint]) -> tuple[list[DataPoint], np.ndarray]:
    """Return (points, X) restricted to data points that have an embedding."""
    valid = [(dp, dp.embedding) for dp in data_points if dp.embedding is not None]
    if not valid:
        raise ValueError("No data points have embeddings — run generate_embeddings first")
    points, raw = zip(*valid)
    return list(points), np.array(raw, dtype=np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def _fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > len(X):
        raise ValueError(f"k={k} exceeds number of embedded points ({len(X)})")
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    model.fit(X)
    return model


def initial_clustering(
    data_points: list[DataPoint],
    k: int,
    session_id: str,
    turn_number: int = 0,
) -> tuple[list[DbCluster], list[DbSoftAssignment]]:
    """Run k-means on the embedding matrix and compute soft assignments.

    Soft probabilities are derived from negative squared distances to centroids
    passed through softmax, so every point's probabilities sum to 1.

    Args:
        data_points: DataPoint rows that must already have embeddings.
        k: Number of clusters (>= 1 and <= number of embedded points).
        session_id: The ChatSession this clustering belongs to.
        turn_number: Turn at which the clustering is recorded (default 0 —
            the pre-oracle state; oracle turns start at 1).

    Returns:
        (db_clusters, db_assignments) — not yet added to any DB session.

    Raises:
        ValueError: if k < 1, no points have embeddings, or k > number of points.
    """
    points, X = _embedding_matrix(data_points)
    model = _fit_kmeans(X, k)

    cluster_ids = [str(uuid.uuid4()) for _ in range(k)]
    db_clusters = [
        DbCluster(
            id=cluster_ids[i],
            session_id=session_id,
            name=f"Cluster {i + 1}",
            description="",
            created_at_turn=turn_number,
        )
        for i in range(k)
    ]

    # Squared Euclidean distance from each point to each centroid: (n, k)
    centroids = model.cluster_centers_
    diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, k, dim)
    sq_dists = np.sum(diffs ** 2, axis=2)  # (n, k)

    # Softmax of negative distances → probabilities in (0, 1) summing to 1 per point
    probs = _softmax(-sq_dists, axis=1)  # (n, k)

    db_assignments = [
        DbSoftAssignment(
            data_point_id=dp.id,
            cluster_id=cluster_ids[j],
            turn_number=turn_number,
            probability=float(probs[i, j]),
        )
        for i, dp in enumerate(points)
        for j in range(k)
    ]

    return db_clusters, db_assignments


def silhouette_for_k(data_points: list[DataPoint], k: int) -> float:
    """Mean silhouette score for a k-means clustering with k clusters.

    Ranges from -1 (overlapping clusters) to 1 (dense, well-separated clusters).
    Requires 2 <= k < number of embedded points.
    """
    if k < 2:
        raise ValueError("silhouette score requires k >= 2")
    _, X = _embedding_matrix(data_points)
    model = _fit_kmeans(X, k)
    return float(silhouette_score(X, model.labels_))


def sweep_k(
    data_points: list[DataPoint],
    k_min: int = 2,
    k_max: int = 10,
) -> dict[int, float]:
    """Compute the silhouette score for every k in [k_min, k_max].

    Returns a dict {k: score}; the highest-scoring k is a reasonable default.
    Values of k that reach or exceed the number of points are skipped.
    """
    if k_min < 2:
        raise ValueError("k_min must be >= 2")
    _, X = _embedding_matrix(data_points)
    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        if k >= len(X):
            break
        model = _fit_kmeans(X, k)
        scores[k] = float(silhouette_score(X, model.labels_))
    return scores
