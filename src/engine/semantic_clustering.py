"""Re-clustering in a semantically oriented hybrid embedding space.

Called at Turn 1 when the oracle provides an axis_hint. Dissolves all existing
active clusters and creates k fresh ones whose centroid geometry reflects both
the original embeddings (weight alpha) and the oracle's semantic axis (weight
beta). Subsequent oracle turns then operate on a geometry that already aligns
with the semantic intent expressed at Turn 1.

Design decisions vs. alternatives:
- We dissolve ALL existing clusters and create new ones, rather than reusing
  old cluster IDs with new probabilities. This is intentional: after a geometry
  change the old names no longer describe the new groupings correctly, so fresh
  clusters with LLM-generated names starting from the re-embedded data are more
  useful to the oracle.
- k is selected automatically via silhouette score (k=2..K_AUTO_MAX) when not
  provided explicitly. This lets the data geometry decide how many natural bins
  the axis has rather than hard-capping at 3.
- The caller owns the DB transaction — no commit is made here.
"""

import uuid

import numpy as np

from src.engine.cluster_naming import name_clusters
from src.engine.f_semantic_reembed import reembed_for_axis
from src.engine.initial_clustering import (
    KMEANS_RANDOM_STATE,
    _fit_kmeans,
    _softmax,
)
from src.engine.turn_builder import TurnBuilder
from src.logger import log_clustering_run
from src.models import (
    Cluster as DbCluster,
    DataPoint,
    SoftAssignment as DbSoftAssignment,
)

SEMANTIC_BACKEND = "semantic_reembed"

K_AUTO_MIN = 2
K_AUTO_MAX = 5


def _auto_select_k(X: np.ndarray, k_min: int, k_max: int) -> int:
    """Pick k in [k_min, k_max] that maximises silhouette score on X.

    Fits k-means for each candidate k and returns the one with the highest
    average silhouette score. If k_max < k_min (tiny dataset) returns k_min
    without fitting any model.
    """
    from sklearn.metrics import silhouette_score

    if k_max < k_min:
        return k_min

    best_k, best_sil = k_min, -np.inf
    for candidate_k in range(k_min, k_max + 1):
        m = _fit_kmeans(X, candidate_k)
        sil = float(silhouette_score(X, m.labels_))
        print(
            f"[semantic-clustering] auto-k  k={candidate_k}  silhouette={sil:.3f}",
            flush=True,
        )
        if sil > best_sil:
            best_k, best_sil = candidate_k, sil

    print(
        f"[semantic-clustering] auto-k selected k={best_k}  "
        f"best_silhouette={best_sil:.3f}",
        flush=True,
    )
    return best_k


def semantic_clustering(
    data_points: list[DataPoint],
    axis_hint: str,
    builder: TurnBuilder,
    k: int | None = None,
    axis_weight: float = 0.7,
    auto_name: bool = True,
) -> list[DbCluster]:
    """Re-cluster the dataset in a hybrid embedding space oriented by axis_hint.

    Stages all currently active clusters (in DB and within this turn) as
    dissolved and creates k fresh ones. Soft assignment probabilities are
    derived from softmax over negative squared distances in the hybrid space,
    exactly as in initial_clustering. The builder's snapshot is REPLACED — a
    semantic re-embed is a wholesale re-clustering, not an incremental update.

    Args:
        data_points: All DataPoint rows for the session's dataset. Rows without
            embeddings are silently skipped.
        axis_hint: The semantic axis extracted from the oracle's intent
            (e.g. "angry", "battery life", "positive sentiment").
        builder: The conversation turn's in-memory staging area.
        k: Number of clusters to produce. When None (default), selected
            automatically via silhouette score over k=K_AUTO_MIN..K_AUTO_MAX.
        axis_weight: Fraction [0, 1] of k-means distance signal attributed to
            the semantic axis (default 0.7).
        auto_name: When True (default), new clusters are named by the LLM via
            ``name_clusters``. Naming is best-effort — a failed LLM call
            leaves placeholders and never aborts the clustering.

    Returns:
        The list of new clusters (also added to ``builder.new_clusters``).

    Raises:
        ValueError: no active clusters; k < 1; no embedded points; or
            k > number of embedded points.
    """
    if builder.turn_number <= 0:
        raise ValueError(
            f"semantic_clustering requires turn_number > 0, got {builder.turn_number}"
        )

    existing = builder.active_clusters()
    if not existing:
        raise ValueError(
            f"session '{builder.session_id}' has no active clusters — "
            "run initial clustering first"
        )

    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    valid = [dp for dp in data_points if dp.embedding is not None]
    if not valid:
        raise ValueError("no data points have embeddings")

    print(
        f"[semantic-clustering] session={builder.session_id}  axis='{axis_hint}'  "
        f"n_embedded={len(valid)}  turn={builder.turn_number}",
        flush=True,
    )

    # Compute the hybrid (N, D+1) embedding matrix.
    # Must happen before k selection so silhouette-based auto-k uses real geometry.
    X, _ = reembed_for_axis(valid, axis_hint, axis_weight=axis_weight)

    if k is None:
        k = _auto_select_k(X, K_AUTO_MIN, min(K_AUTO_MAX, len(valid) - 1))

    if k > len(valid):
        raise ValueError(
            f"k={k} exceeds number of embedded points ({len(valid)})"
        )

    print(
        f"[semantic-clustering] k={k}",
        flush=True,
    )

    model = _fit_kmeans(X, k)

    # Stage all existing active clusters as dissolved.
    for cluster in existing:
        builder.dissolve(cluster.id)

    cluster_ids = [str(uuid.uuid4()) for _ in range(k)]
    new_clusters = [
        DbCluster(
            id=cluster_ids[i],
            session_id=builder.session_id,
            name=f"Cluster {i + 1}",
            description="",
            created_at_turn=builder.turn_number,
        )
        for i in range(k)
    ]

    # Soft-assignment probabilities: softmax over negative squared distances
    # to centroids, one row per point, one column per cluster.
    centroids = model.cluster_centers_
    diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (N, k, D+1)
    sq_dists = np.sum(diffs**2, axis=2)  # (N, k)
    probs = _softmax(-sq_dists, axis=1)  # (N, k), sums to 1 per row

    # Replace the builder's snapshot wholesale — every point now has fresh
    # probabilities over the new clusters; old clusters fall away with their
    # dissolution.
    new_snapshot: dict[str, dict[str, float]] = {}
    naming_assignments: list[DbSoftAssignment] = []
    for i, dp in enumerate(valid):
        dist: dict[str, float] = {}
        for j in range(k):
            p = float(probs[i, j])
            dist[cluster_ids[j]] = p
            naming_assignments.append(
                DbSoftAssignment(
                    data_point_id=dp.id,
                    cluster_id=cluster_ids[j],
                    turn_number=builder.turn_number,
                    probability=p,
                )
            )
        new_snapshot[dp.id] = dist
    builder.snapshot = new_snapshot

    for cluster in new_clusters:
        builder.add_cluster(cluster)

    # Log hard-assignment sizes so we can see if k-means split the data sensibly.
    hard_labels = model.labels_
    cluster_sizes = {i: int((hard_labels == i).sum()) for i in range(k)}
    print(
        f"[semantic-clustering] k-means done  "
        + "  ".join(f"C{i+1}={sz}" for i, sz in cluster_sizes.items()),
        flush=True,
    )

    if auto_name:
        name_clusters(new_clusters, naming_assignments, valid, axis_hint=axis_hint)

    silhouette: float | None = None
    if 2 <= k < len(valid):
        from sklearn.metrics import silhouette_score
        silhouette = float(silhouette_score(X, model.labels_))

    log_clustering_run(
        session_id=builder.session_id,
        k=k,
        backend=SEMANTIC_BACKEND,
        seed=KMEANS_RANDOM_STATE,
        n_points=len(valid),
        silhouette=silhouette,
        turn_number=builder.turn_number,
    )

    sil_str = f"{silhouette:.3f}" if silhouette is not None else "N/A (k=1)"
    cluster_names = [c.name for c in new_clusters]
    print(
        f"[semantic-clustering] done  silhouette={sil_str}  "
        f"clusters={cluster_names}",
        flush=True,
    )

    return new_clusters
