"""Initial clustering on stored embeddings using GMM (primary) or k-means (fallback).

`initial_clustering` returns DB model objects ready to be persisted — callers
own the transaction. `silhouette_for_k` / `sweep_k` are diagnostics that help
the oracle pick a sensible number of clusters.
"""

import logging
import uuid
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from src.logger import log_clustering_run
from src.models import Cluster as DbCluster, DataPoint, SoftAssignment as DbSoftAssignment

log = logging.getLogger(__name__)

# Toggle: True → try GMM first, fall back to k-means on convergence failure.
# False → always use k-means (original behaviour).
USE_GMM = True

# Random seed used by both backends. Logged with every run for reproducibility.
KMEANS_RANDOM_STATE = 42

# Backward-compatible alias — some tests / callers import KMEANS_BACKEND.
# The actual backend used per-run is determined at runtime (gmm or kmeans).
KMEANS_BACKEND = "kmeans"

# Temperature for the soft-assignment softmax used in the k-means fallback path.
# See the original module docstring for the design rationale.
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


def _fit_kmeans(X: np.ndarray, k: int, seed: int = KMEANS_RANDOM_STATE) -> KMeans:
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > len(X):
        raise ValueError(f"k={k} exceeds number of embedded points ({len(X)})")
    model = KMeans(n_clusters=k, random_state=seed, n_init=20)
    model.fit(X)
    return model


def _kmeans_probs(X: np.ndarray, model: KMeans, k: int) -> np.ndarray:
    """Compute soft-assignment probabilities from k-means centroids via softmax."""
    centroids = model.cluster_centers_
    diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, k, dim)
    sq_dists = np.sum(diffs ** 2, axis=2)  # (n, k)
    min_sq_dists = sq_dists.min(axis=1)  # (n,) — distance to nearest centroid
    temperature = max(SOFTMAX_TEMPERATURE_FRACTION * float(min_sq_dists.mean()), 1e-12)
    return _softmax(-sq_dists / temperature, axis=1)  # (n, k)


def _fit_gmm(X: np.ndarray, k: int, seed: int = KMEANS_RANDOM_STATE) -> tuple[GaussianMixture, np.ndarray]:
    """Fit a diagonal-covariance GMM and return (model, probs).

    Uses diag covariance: the best balance between expressiveness and numerical
    stability for 384-dimensional embeddings.  Full covariance would require
    ~384² = 147 456 parameters per component — more than any reasonably-sized
    cluster has data points, leading to a singular covariance matrix.

    Raises ConvergenceWarning (re-raised as an exception via warnings filter) if
    the EM algorithm does not converge; the caller catches this and falls back
    to k-means.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > len(X):
        raise ValueError(f"k={k} exceeds number of embedded points ({len(X)})")

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        n_init=5,
        max_iter=200,
        random_state=seed,
        reg_covar=1e-4,  # regularise diagonal to prevent near-zero variances
    )

    # Convert ConvergenceWarning to an exception so the caller can catch it.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UserWarning)
        gmm.fit(X)

    probs = gmm.predict_proba(X)  # (n, k) — native posteriors, sums to 1 per row
    # Guard against NaN/Inf that can arise when a component collapses.
    if not np.all(np.isfinite(probs)):
        raise ValueError("GMM produced non-finite probabilities — falling back")

    return gmm, probs


def initial_clustering(
    data_points: list[DataPoint],
    k: int,
    session_id: str,
    turn_number: int = 0,
    seed: int = KMEANS_RANDOM_STATE,
) -> tuple[list[DbCluster], list[DbSoftAssignment], float | None]:
    """Run GMM (or k-means fallback) on the embedding matrix and compute soft assignments.

    With GMM, soft probabilities are the native posterior probabilities from the
    EM algorithm — no softmax hack needed.  The k-means fallback uses the
    temperature-scaled softmax of negative squared distances.

    Args:
        data_points: DataPoint rows that must already have embeddings.
        k: Number of clusters (>= 1 and <= number of embedded points).
        session_id: The ChatSession this clustering belongs to.
        turn_number: Turn at which the clustering is recorded (default 0 —
            the pre-oracle state; oracle turns start at 1).
        seed: Random seed (default ``KMEANS_RANDOM_STATE = 42``).

    Returns:
        (db_clusters, db_assignments, silhouette) — not yet added to any DB
        session.  ``silhouette`` is computed from hard-assignment labels.

    Raises:
        ValueError: if k < 1, no points have embeddings, or k > number of points.
    """
    points, X = _embedding_matrix(data_points)

    probs: np.ndarray
    hard_labels: np.ndarray
    backend: str

    if USE_GMM and k >= 2:
        try:
            gmm_model, probs = _fit_gmm(X, k, seed=seed)
            hard_labels = gmm_model.predict(X)
            backend = "gmm"
            log.info("GMM fit succeeded (k=%d, n=%d)", k, len(points))
        except Exception as exc:
            log.warning(
                "GMM fit failed (k=%d, n=%d, reason=%s) — falling back to k-means",
                k, len(points), exc,
            )
            km = _fit_kmeans(X, k, seed=seed)
            probs = _kmeans_probs(X, km, k)
            hard_labels = km.labels_
            backend = "kmeans"
    else:
        # k=1 or GMM disabled: always use k-means.
        km = _fit_kmeans(X, k, seed=seed)
        if k == 1:
            probs = np.ones((len(points), 1), dtype=np.float64)
        else:
            probs = _kmeans_probs(X, km, k)
        hard_labels = km.labels_
        backend = "kmeans"

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

    silhouette: float | None = None
    if 2 <= k < len(points):
        try:
            silhouette = float(silhouette_score(X, hard_labels))
        except Exception:
            silhouette = None

    log_clustering_run(
        session_id=session_id,
        k=k,
        backend=backend,
        seed=seed,
        n_points=len(points),
        silhouette=silhouette,
        turn_number=turn_number,
    )

    return db_clusters, db_assignments, silhouette


def silhouette_for_k(data_points: list[DataPoint], k: int) -> float:
    """Mean silhouette score for a clustering with k clusters.

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
