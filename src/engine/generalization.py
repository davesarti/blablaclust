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
if it had been part of the original fit. The three assignment functions here
(``build_centroids``, ``assign_nearest``, ``centroids_from_snapshot``) are
deliberately free of any DB or LLM coupling — they operate on plain numpy arrays
and a snapshot dict — so they are cheap to unit-test and reusable by both the
eval harness and a live API endpoint.

``ingest_points`` (online generalization eval) is the one DB-coupled entry
point: it admits *new* data into an already-converged session by nearest-centroid
assignment against the **frozen** convergence centroids, written as a fresh full
snapshot at ``turn + 1``. Pre-existing points are carried forward verbatim — never
re-clustered — so it is a read-only assignment over the existing geometry. See its
docstring for the policy.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.engine.initial_clustering import SOFTMAX_TEMPERATURE_FRACTION
from src.models import (
    Cluster as DbCluster,
    DataPoint,
    SoftAssignment as DbSoftAssignment,
)


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


# ---------------------------------------------------------------------------
# Live-session ingestion (DB-coupled)
# ---------------------------------------------------------------------------
# Everything above is pure numpy. ``ingest_points`` is the single DB-coupled
# entry point — it composes the pure functions above with the soft-assignment
# schema to admit new data into a running session.


def ingest_points(
    session_id: str,
    new_points: list[DataPoint],
    centroid_ids: list[str],
    centroids: np.ndarray,
    db: Session,
    *,
    temperature_fraction: float = SOFTMAX_TEMPERATURE_FRACTION,
) -> tuple[int, list[DbSoftAssignment]]:
    """Admit new, already-embedded points into a converged session (read-only).

    This is the operational meaning of *generalization*: once the oracle is
    happy, new data arrives into the running system and must be placed without
    disturbing what the oracle accepted. We do the simplest defensible thing —
    **nearest-centroid assignment against the frozen convergence centroids** —
    and record it as a fresh full snapshot at ``turn + 1``.

    Read-only-assignment policy
    ---------------------------
    * Centroids are **frozen**: the caller passes the convergence centroids
      (built once with :func:`centroids_from_snapshot`); they are **not**
      recomputed here and do **not** drift as successive batches arrive.
    * Pre-existing points are **carried forward verbatim** — their soft
      assignments from the latest snapshot are copied unchanged to the new turn.
      They are never re-clustered or re-scored, so old-point assignment stability
      is 100% *by construction* (which is exactly why a separate "stability"
      metric would be vacuous — see ``docs/quality_specs.md``).
    * Only the new points get fresh rows. Their probabilities are the softmax of
      negative squared distance to each frozen centroid (the same construction as
      :func:`src.engine.initial_clustering.initial_clustering`), so a new point
      that sits far from every centroid earns a low max-probability and surfaces
      in the coherence judge's bottom-2 stress sample.

    Args:
        session_id: the converged session to ingest into.
        new_points: NEW :class:`~src.models.DataPoint` objects that already carry
            an ``embedding`` (embedding is the caller's job, as in
            ``initial_clustering``). They are added to ``db`` here.
        centroid_ids: cluster ids matching ``centroids`` rows (frozen).
        centroids: ``(k, d)`` frozen convergence centroids.
        db: SQLAlchemy session. Rows are **staged, not committed** — the caller
            owns the transaction.
        temperature_fraction: softmax temperature as a fraction of the new
            batch's mean squared distance (defaults to the engine's value so the
            calibration of new points matches the initial clustering).

    Returns:
        ``(new_turn, new_assignments)`` — the turn the snapshot was written at and
        every :class:`~src.models.SoftAssignment` row created (carried-forward +
        new).

    Raises:
        ValueError: no new point has an embedding; the session has no existing
            clustering; or an embedding/centroid dimensionality mismatch.
    """
    embedded = [dp for dp in new_points if dp.embedding is not None]
    if not embedded:
        raise ValueError("no new point has an embedding — embed before ingesting")
    X = np.asarray([dp.embedding for dp in embedded], dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)

    # Reuse the public nearest-centroid assignment — it also validates the
    # embedding/centroid dimensions and raises on a mismatch.
    nearest = assign_nearest(X, centroid_ids, centroids)

    # Soft probabilities over the FROZEN centroids: softmax(-d² / temperature),
    # mirroring initial_clustering so new points are calibrated on the same scale.
    diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)  # (m, k)
    temperature = max(temperature_fraction * float(sq_dists.mean()), 1e-12)
    shifted = -sq_dists / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)  # (m, k)

    # softmax(-d²) is monotone in -distance, so the argmax must equal the
    # nearest-centroid hard label. Guard the invariant rather than assume it.
    if [centroid_ids[j] for j in probs.argmax(axis=1)] != nearest:
        raise RuntimeError(
            "soft assignment disagrees with nearest centroid — numerical bug"
        )

    # Latest snapshot turn for this session (the convergence turn, or the last
    # ingestion turn when building a multi-batch drift curve).
    session_cluster_ids = [
        cid
        for (cid,) in db.query(DbCluster.id).filter(DbCluster.session_id == session_id)
    ]
    if not session_cluster_ids:
        raise ValueError(
            f"session '{session_id}' has no clusters — nothing to ingest into"
        )
    prev_turn = (
        db.query(func.max(DbSoftAssignment.turn_number))
        .filter(DbSoftAssignment.cluster_id.in_(session_cluster_ids))
        .scalar()
    )
    if prev_turn is None:
        raise ValueError(
            f"session '{session_id}' has no soft assignments — "
            "run initial clustering first"
        )
    new_turn = prev_turn + 1

    new_rows: list[DbSoftAssignment] = []

    # 1) Carry the existing snapshot forward verbatim (the read-only part).
    prev_rows = (
        db.query(DbSoftAssignment)
        .filter(
            DbSoftAssignment.cluster_id.in_(session_cluster_ids),
            DbSoftAssignment.turn_number == prev_turn,
        )
        .all()
    )
    for r in prev_rows:
        new_rows.append(
            DbSoftAssignment(
                data_point_id=r.data_point_id,
                cluster_id=r.cluster_id,
                turn_number=new_turn,
                probability=r.probability,
            )
        )

    # 2) Persist the new points, then their soft assignments at the new turn.
    #    The DataPoints must exist before their FK soft-assignment rows are added.
    for dp in embedded:
        db.add(dp)
    db.flush()
    for i, dp in enumerate(embedded):
        for j, cid in enumerate(centroid_ids):
            new_rows.append(
                DbSoftAssignment(
                    data_point_id=dp.id,
                    cluster_id=cid,
                    turn_number=new_turn,
                    probability=float(probs[i, j]),
                )
            )

    for row in new_rows:
        db.add(row)
    return new_turn, new_rows
