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
- k defaults to the number of currently active clusters (preserving the
  oracle's original k choice). The caller can override this.
- The caller owns the DB transaction — no commit is made here.
"""

import uuid

import numpy as np
from sqlalchemy.orm import Session

from src.engine.cluster_naming import name_clusters
from src.engine.f_semantic_reembed import reembed_for_axis
from src.engine.initial_clustering import (
    KMEANS_RANDOM_STATE,
    _fit_kmeans,
    _softmax,
)
from src.logger import log_clustering_run
from src.models import (
    Cluster as DbCluster,
    DataPoint,
    SoftAssignment as DbSoftAssignment,
)

SEMANTIC_BACKEND = "semantic_reembed"


def semantic_clustering(
    data_points: list[DataPoint],
    axis_hint: str,
    session_id: str,
    turn_number: int,
    db: Session,
    k: int | None = None,
    alpha: float = 0.7,
    beta: float = 0.3,
    auto_name: bool = True,
) -> tuple[list[DbCluster], list[DbSoftAssignment]]:
    """Re-cluster the dataset in a hybrid embedding space oriented by axis_hint.

    Dissolves all currently active clusters for the session and creates k fresh
    ones. Soft assignment probabilities are derived from softmax over negative
    squared distances in the hybrid space, exactly as in initial_clustering.

    Args:
        data_points: All DataPoint rows for the session's dataset. Rows without
            embeddings are silently skipped.
        axis_hint: The semantic axis extracted from the oracle's intent
            (e.g. "angry", "battery life", "positive sentiment").
        session_id: The ChatSession to re-cluster.
        turn_number: Turn at which the new snapshot is written. Must be > 0
            (turn 0 is reserved for the pre-oracle initial clustering).
        db: SQLAlchemy session. Changes are staged but not committed.
        k: Number of clusters to produce. Defaults to the current number of
            active clusters in the session (preserving the oracle's original k).
        alpha: Weight for the row-normalised original embedding (default 0.7).
        beta: Weight for the normalised axis score (default 0.3).
        auto_name: When True (default), new clusters are named by the LLM via
            name_clusters. When False the generic "Cluster N" placeholders are
            kept. Naming is best-effort — a failed LLM call leaves the
            placeholder and never aborts the clustering.

    Returns:
        (new_clusters, new_assignments) — not yet staged on the DB session.
        The caller must db.add() each object and then commit.

    Raises:
        ValueError: no active clusters; k < 1; no embedded points; k > number
            of embedded points; or turn_number <= 0.
    """
    if turn_number <= 0:
        raise ValueError(
            f"semantic_clustering requires turn_number > 0, got {turn_number}"
        )

    # Load existing active clusters — we need their count for the default k
    # and we will dissolve them as part of this operation.
    existing = (
        db.query(DbCluster)
        .filter(
            DbCluster.session_id == session_id,
            DbCluster.dissolved_at_turn.is_(None),
        )
        .all()
    )
    if not existing:
        raise ValueError(
            f"session '{session_id}' has no active clusters — "
            "run initial clustering first"
        )

    if k is None:
        k = len(existing)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    valid = [dp for dp in data_points if dp.embedding is not None]
    if not valid:
        raise ValueError("no data points have embeddings")
    if k > len(valid):
        raise ValueError(
            f"k={k} exceeds number of embedded points ({len(valid)})"
        )

    # Compute the hybrid (N, D+1) embedding matrix for the full dataset.
    X = reembed_for_axis(valid, axis_hint, alpha=alpha, beta=beta)

    # Run k-means in the hybrid space.
    model = _fit_kmeans(X, k)

    # Dissolve all existing active clusters as of this turn.
    for cluster in existing:
        cluster.dissolved_at_turn = turn_number

    # Build k fresh cluster objects.
    cluster_ids = [str(uuid.uuid4()) for _ in range(k)]
    new_clusters = [
        DbCluster(
            id=cluster_ids[i],
            session_id=session_id,
            name=f"Cluster {i + 1}",
            description="",
            created_at_turn=turn_number,
        )
        for i in range(k)
    ]

    # Soft-assignment probabilities: softmax over negative squared distances
    # to centroids, one row per point, one column per cluster.
    centroids = model.cluster_centers_
    diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (N, k, D+1)
    sq_dists = np.sum(diffs**2, axis=2)  # (N, k)
    probs = _softmax(-sq_dists, axis=1)  # (N, k), sums to 1 per row

    new_assignments = [
        DbSoftAssignment(
            data_point_id=valid[i].id,
            cluster_id=cluster_ids[j],
            turn_number=turn_number,
            probability=float(probs[i, j]),
        )
        for i in range(len(valid))
        for j in range(k)
    ]

    # Name the new clusters via LLM — best-effort, a failure leaves placeholders.
    if auto_name:
        name_clusters(new_clusters, new_assignments, valid)

    # Log the run so it appears in clustering_runs.jsonl alongside all other runs.
    silhouette: float | None = None
    if 2 <= k < len(valid):
        from sklearn.metrics import silhouette_score
        silhouette = float(silhouette_score(X, model.labels_))

    log_clustering_run(
        session_id=session_id,
        k=k,
        backend=SEMANTIC_BACKEND,
        seed=KMEANS_RANDOM_STATE,
        n_points=len(valid),
        silhouette=silhouette,
        turn_number=turn_number,
    )

    return new_clusters, new_assignments
