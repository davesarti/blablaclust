"""Generalization: codify a finished clustering into a reusable assignment
function and apply it to new (held-out) items.

The brief asks whether, once the oracle is happy, the system can produce a
function that assigns *new* items consistently. We answer with the simplest
defensible mapping: **nearest-centroid over the sentence-transformer
embeddings**. A cluster's centroid is the mean embedding of the points
hard-assigned to it in the final clustering; a new item is embedded with the
same model and assigned to the closest centroid.

This mirrors how the clusters were formed: ``initial_clustering`` runs k-means
(Euclidean) over the raw embeddings, so a held-out item is assigned exactly as
if it had been part of the original fit. The functions here are deliberately
free of any DB or LLM coupling — they operate on plain numpy arrays and a
snapshot dict — so they are cheap to unit-test and reusable by both the eval
harness and a live API endpoint.
"""

from __future__ import annotations

import numpy as np


def build_centroids(
    embeddings: np.ndarray,
    hard_labels: list[str],
) -> tuple[list[str], np.ndarray]:
    """Mean embedding per cluster.

    Args:
        embeddings: ``(n, d)`` matrix, one row per point.
        hard_labels: length-``n`` list of the cluster id each point belongs to.

    Returns:
        ``(centroid_ids, centroids)`` where ``centroid_ids`` is the sorted list
        of distinct cluster ids and ``centroids`` is the matching ``(k, d)``
        matrix of mean embeddings. Sorting makes the output deterministic
        regardless of point order.

    Raises:
        ValueError: empty input, or a length mismatch between rows and labels.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty (n, d) matrix")
    if len(hard_labels) != embeddings.shape[0]:
        raise ValueError(
            f"hard_labels length ({len(hard_labels)}) != number of "
            f"embeddings ({embeddings.shape[0]})"
        )

    centroid_ids = sorted(set(hard_labels))
    centroids = np.empty((len(centroid_ids), embeddings.shape[1]), dtype=np.float64)
    for i, cid in enumerate(centroid_ids):
        mask = np.array([lbl == cid for lbl in hard_labels])
        centroids[i] = embeddings[mask].mean(axis=0)
    return centroid_ids, centroids


def assign_nearest(
    embeddings: np.ndarray,
    centroid_ids: list[str],
    centroids: np.ndarray,
) -> list[str]:
    """Assign each row to the nearest centroid by Euclidean distance.

    Args:
        embeddings: ``(m, d)`` matrix of items to classify.
        centroid_ids: length-``k`` cluster ids matching ``centroids`` rows.
        centroids: ``(k, d)`` centroid matrix from :func:`build_centroids`.

    Returns:
        Length-``m`` list of the assigned cluster id for each row.

    Raises:
        ValueError: empty input or a dimensionality mismatch.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty (m, d) matrix")
    if centroids.ndim != 2 or centroids.shape[0] == 0:
        raise ValueError("centroids must be a non-empty (k, d) matrix")
    if embeddings.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"embedding dim ({embeddings.shape[1]}) != centroid dim "
            f"({centroids.shape[1]})"
        )
    if len(centroid_ids) != centroids.shape[0]:
        raise ValueError(
            f"centroid_ids length ({len(centroid_ids)}) != number of "
            f"centroids ({centroids.shape[0]})"
        )

    # Squared Euclidean distance (m, k); argmin is identical to using the root.
    diffs = embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    nearest = np.argmin(sq_dists, axis=1)
    return [centroid_ids[j] for j in nearest]


def centroids_from_snapshot(
    point_embeddings: dict[str, list[float] | np.ndarray],
    snapshot: dict[str, dict[str, float]],
) -> tuple[list[str], np.ndarray]:
    """Build centroids from a finished clustering snapshot.

    A "snapshot" is the per-point distribution over clusters produced by the
    engine (``data_point_id -> {cluster_id: probability}``), e.g. the output of
    ``cluster_operations._load_latest_snapshot``. Each point is folded into its
    hard (argmax) cluster and the centroid is the mean of those embeddings.

    Args:
        point_embeddings: ``data_point_id -> embedding`` for every point that
            has an embedding. Points missing here are skipped (a point with no
            embedding cannot contribute to a centroid).
        snapshot: the final clustering distribution per point.

    Returns:
        ``(centroid_ids, centroids)`` as in :func:`build_centroids`.

    Raises:
        ValueError: no point in the snapshot has a usable embedding.
    """
    vecs: list[np.ndarray] = []
    labels: list[str] = []
    for point_id, distribution in snapshot.items():
        emb = point_embeddings.get(point_id)
        if emb is None or not distribution:
            continue
        hard_cluster = max(distribution, key=distribution.get)
        vecs.append(np.asarray(emb, dtype=np.float64))
        labels.append(hard_cluster)

    if not vecs:
        raise ValueError(
            "no point in the snapshot has an embedding — cannot build centroids"
        )
    return build_centroids(np.vstack(vecs), labels)
