"""Initial k-means clustering on stored embeddings.

`initial_clustering` returns DB model objects ready to be persisted — callers
own the transaction. `silhouette_for_k` / `sweep_k` are diagnostics that help
the oracle pick a sensible number of clusters.
"""

import uuid

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.logger import log_clustering_run
from src.models import Cluster as DbCluster, DataPoint, SoftAssignment as DbSoftAssignment

# Random seed used by k-means. Logged with every clustering run for
# reproducibility — change this and runs become non-comparable.
KMEANS_RANDOM_STATE = 42
KMEANS_BACKEND = "kmeans"

# Temperature for the soft-assignment softmax, expressed as a fraction of the
# mean squared distance to centroids. softmax(-d²) with no temperature (i.e. a
# fixed 1.0) is nearly uniform for unit-norm embeddings — every point's distances
# are O(1) and similar, so each cluster gets ~1/k and no point looks more certain
# than any other. Dividing distances by a fraction of their own mean sharpens the
# split and stays scale-invariant if the embedding model changes. 0.1 chosen
# empirically: core points reach ~0.9 max-probability, boundary points ~0.4.
SOFTMAX_TEMPERATURE_FRACTION = 0.1


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
    model = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init="auto")
    model.fit(X)
    return model


def initial_clustering(
    data_points: list[DataPoint],
    k: int,
    session_id: str,
    turn_number: int = 0,
) -> tuple[list[DbCluster], list[DbSoftAssignment], float | None]:
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
        (db_clusters, db_assignments, silhouette) — not yet added to any DB
        session. ``silhouette`` is the mean silhouette score of this exact
        k-means fit, or ``None`` when it is undefined (``k < 2`` or
        ``k >= n_points``). Returning it lets callers reuse the value that was
        already computed for the structured log instead of re-fitting k-means.

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

    # Temperature scaled to the data's own distance spread keeps the softmax
    # sharp regardless of embedding magnitude. Guard against a zero mean (e.g.
    # k == n, where every point sits exactly on its own centroid).
    temperature = max(SOFTMAX_TEMPERATURE_FRACTION * float(sq_dists.mean()), 1e-12)

    # Softmax of negative scaled distances → probabilities in (0, 1) summing to 1 per point
    probs = _softmax(-sq_dists / temperature, axis=1)  # (n, k)

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

    # Structured log of this run. Silhouette is undefined for k < 2 or
    # k >= n_points; we record None in those cases rather than crashing.
    # silhouette_score can also raise ValueError on degenerate input (e.g.
    # all embeddings identical → a single distinct label). The score is a
    # best-effort diagnostic — it must never abort an otherwise-valid
    # clustering run, so we swallow any failure and fall back to None.
    silhouette: float | None = None
    if 2 <= k < len(points):
        try:
            silhouette = float(silhouette_score(X, model.labels_))
        except Exception:
            silhouette = None
    log_clustering_run(
        session_id=session_id,
        k=k,
        backend=KMEANS_BACKEND,
        seed=KMEANS_RANDOM_STATE,
        n_points=len(points),
        silhouette=silhouette,
        turn_number=turn_number,
    )

    return db_clusters, db_assignments, silhouette


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
