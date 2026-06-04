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


def gmm_params_from_snapshot(
    point_embeddings: dict[str, list[float] | np.ndarray],
    snapshot: dict[str, dict[str, float]],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Build full GMM parameters from a finished clustering snapshot.

    Extends :func:`centroids_from_snapshot` with the per-cluster diagonal
    covariance and log mixing weights needed for GMM-posterior assignment of
    new arrivals. All quantities are recovered from the snapshot's hard-label
    assignments (argmax per point), making them independent of the fitted model
    object and consistent whether the run used GMM or k-means fallback.

    Args:
        point_embeddings: ``data_point_id -> embedding``. Points without an
            embedding are skipped.
        snapshot: ``data_point_id -> {cluster_id: probability}``.

    Returns:
        ``(centroid_ids, centroids, diag_vars, log_weights)`` where:

        - ``centroid_ids``: sorted list of ``k`` cluster ids.
        - ``centroids``: ``(k, d)`` mean embedding per cluster.
        - ``diag_vars``: ``(k, d)`` per-dim sample variance (ddof=0) per cluster.
        - ``log_weights``: ``(k,)`` log(n_c / n_total) mixing weight per cluster.

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
            "no point in the snapshot has an embedding — cannot build GMM params"
        )

    X = np.vstack(vecs)
    centroid_ids, centroids = build_centroids(X, labels)

    labels_arr = np.array(labels)
    n_total = len(labels)
    diag_vars = np.zeros((len(centroid_ids), X.shape[1]), dtype=np.float64)
    log_weights = np.empty(len(centroid_ids), dtype=np.float64)
    for i, cid in enumerate(centroid_ids):
        mask = labels_arr == cid
        cluster_emb = X[mask]
        diag_vars[i] = cluster_emb.var(axis=0, ddof=0)
        log_weights[i] = np.log(mask.sum() / n_total)

    return centroid_ids, centroids, diag_vars, log_weights


def assign_gmm_posterior(
    embeddings: np.ndarray,
    centroid_ids: list[str],
    centroids: np.ndarray,
    diag_vars: np.ndarray,
    log_weights: np.ndarray,
) -> list[str]:
    """Assign each row to the cluster with the highest GMM log-posterior.

    The log-posterior (up to a shared normalising constant) for point ``x``
    under cluster ``c`` with diagonal covariance is:

    ``log π_c − ½ d²_M(x,c) − ½ Σ_j log(σ²_{c,j} + ε)``

    where ``d²_M(x,c) = Σ_j (x_j − μ_{c,j})² / (σ²_{c,j} + ε)`` and
    ``ε = MAHALANOBIS_REG_COVAR``. The hard label is the argmax over ``c``.
    Unlike :func:`assign_nearest`, this accounts for per-cluster spread: a
    wide cluster can win over a tight one at the same Euclidean distance.

    Args:
        embeddings: ``(m, d)`` matrix of points to classify.
        centroid_ids: length-``k`` cluster ids matching ``centroids`` rows.
        centroids: ``(k, d)`` frozen mean matrix.
        diag_vars: ``(k, d)`` per-cluster diagonal variance (e.g. from
            :func:`gmm_params_from_snapshot`).
        log_weights: ``(k,)`` log mixing weight per cluster.

    Returns:
        Length-``m`` list of the assigned cluster id for each row.

    Raises:
        ValueError: empty input or a dimensionality mismatch.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    diag_vars = np.asarray(diag_vars, dtype=np.float64)
    log_weights = np.asarray(log_weights, dtype=np.float64)

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty (m, d) matrix")
    if centroids.ndim != 2 or centroids.shape[0] == 0:
        raise ValueError("centroids must be a non-empty (k, d) matrix")
    if embeddings.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"embedding dim ({embeddings.shape[1]}) != centroid dim "
            f"({centroids.shape[1]})"
        )
    if diag_vars.shape != centroids.shape:
        raise ValueError(
            f"diag_vars shape {diag_vars.shape} != centroids shape {centroids.shape}"
        )
    if log_weights.shape != (centroids.shape[0],):
        raise ValueError(
            f"log_weights length ({log_weights.shape[0]}) != k ({centroids.shape[0]})"
        )

    reg = diag_vars + MAHALANOBIS_REG_COVAR  # (k, d)
    # Mahalanobis squared distance: (m, k)
    diffs = embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    mah_sq = ((diffs ** 2) / reg[np.newaxis, :, :]).sum(axis=2)
    # Log cluster constant: log π_c − ½ Σ_j log(σ²_c,j + ε)  shape (k,)
    log_det_term = 0.5 * np.log(reg).sum(axis=1)
    log_scores = log_weights[np.newaxis, :] - 0.5 * mah_sq - log_det_term[np.newaxis, :]
    best = np.argmax(log_scores, axis=1)
    return [centroid_ids[j] for j in best]


# ---------------------------------------------------------------------------
# A4 — Mahalanobis-distance OOD reference (pure numpy)
# ---------------------------------------------------------------------------
# A1 (silhouette) and B2 (LLM coherence) measure geometric structure and
# semantic coherence. A4 asks whether new arrivals are *in-distribution* under
# the frozen cluster geometry. With a GMM engine (diagonal covariance), the
# natural metric is Mahalanobis squared distance per cluster — it scales each
# dimension by that cluster's own variance, unlike Euclidean d² which assumes
# equal spherical spread. Both helpers are temperature-free and comparable
# across batches without recalibration.

OOD_PERCENTILE = 95.0
# Matches GaussianMixture(reg_covar=1e-4) — prevents inf Mahalanobis when a
# dimension has near-zero sample variance (singleton or degenerate cluster).
MAHALANOBIS_REG_COVAR = 1e-4


def calibrate_distance_reference(
    embeddings: np.ndarray,
    hard_labels: list[str],
    centroid_ids: list[str],
    centroids: np.ndarray,
) -> dict[str, dict]:
    """Per-cluster Mahalanobis-d² calibration from base in-cluster points.

    For each cluster ``c`` we collect the per-dim sample variance of its base
    points and then compute the Mahalanobis squared distance of every base
    point to the cluster mean:
    ``d²_M(x, c) = Σ_j (x_j − μ_{c,j})² / (σ²_{c,j} + MAHALANOBIS_REG_COVAR)``

    The 95th percentile of the base ``d²_M`` distribution becomes the OOD
    threshold for that cluster; the mean and std calibrate the z-score.

    Args:
        embeddings: ``(n, d)`` matrix of base (converged) points.
        hard_labels: length-``n`` cluster id per base point (argmax of the
            snapshot). Every label must appear in ``centroid_ids``.
        centroid_ids: length-``k`` cluster ids matching ``centroids`` rows.
        centroids: ``(k, d)`` frozen centroid matrix.

    Returns:
        ``{cluster_id: {"d2_95": float, "mean": float, "std": float,
        "diag_var": ndarray(d,)}}`` for every cluster with at least one base
        point. A singleton cluster has ``diag_var = 0`` per dim; the regulariser
        keeps Mahalanobis distances finite at scoring time.

    Raises:
        ValueError: shape mismatch or a label absent from ``centroid_ids``.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty (n, d) matrix")
    if len(hard_labels) != embeddings.shape[0]:
        raise ValueError(
            f"hard_labels length ({len(hard_labels)}) != number of "
            f"embeddings ({embeddings.shape[0]})"
        )
    if len(centroid_ids) != centroids.shape[0]:
        raise ValueError(
            f"centroid_ids length ({len(centroid_ids)}) != number of "
            f"centroids ({centroids.shape[0]})"
        )
    if embeddings.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"embedding dim ({embeddings.shape[1]}) != centroid dim "
            f"({centroids.shape[1]})"
        )
    known = set(centroid_ids)
    unknown = {lbl for lbl in hard_labels if lbl not in known}
    if unknown:
        raise ValueError(
            f"hard_labels contains ids not in centroid_ids: {sorted(unknown)}"
        )

    cid_to_idx = {cid: j for j, cid in enumerate(centroid_ids)}
    labels_arr = np.array(hard_labels)
    calibration: dict[str, dict] = {}
    for cid in centroid_ids:
        mask = labels_arr == cid
        if not mask.any():
            continue
        cluster_emb = embeddings[mask]
        mu = centroids[cid_to_idx[cid]]
        diag_var = cluster_emb.var(axis=0, ddof=0)  # (d,) sample variance
        diffs = cluster_emb - mu
        mah_sq = ((diffs ** 2) / (diag_var + MAHALANOBIS_REG_COVAR)).sum(axis=1)
        calibration[cid] = {
            "d2_95": float(np.percentile(mah_sq, OOD_PERCENTILE)),
            "mean": float(mah_sq.mean()),
            "std": float(mah_sq.std(ddof=0)),
            "diag_var": diag_var,
        }
    return calibration


def assignment_ood(
    embeddings: np.ndarray,
    assigned_clusters: list[str],
    centroid_ids: list[str],
    centroids: np.ndarray,
    calibration: dict[str, dict],
) -> dict:
    """Score new arrivals against the per-cluster Mahalanobis calibration.

    For each new point ``i`` assigned to cluster ``c_i``, compute the
    Mahalanobis squared distance using the frozen calibration's ``diag_var``
    and compare it to ``d2_95(c_i)``. With a well-calibrated base and
    in-distribution new arrivals, ``ood_rate`` sits near
    ``1 − OOD_PERCENTILE / 100`` (≈ 5%).

    Returns:
        ``{"z": ndarray(m,), "is_ood": ndarray(m, bool), "ood_rate": float,
        "mean_z": float, "per_cluster": {cid: {"n", "ood_rate", "mean_z"}}}``

    Raises:
        ValueError: shape mismatch or an assigned cluster absent from
            ``calibration``.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty (m, d) matrix")
    if len(assigned_clusters) != embeddings.shape[0]:
        raise ValueError(
            f"assigned_clusters length ({len(assigned_clusters)}) != number of "
            f"embeddings ({embeddings.shape[0]})"
        )
    if len(centroid_ids) != centroids.shape[0]:
        raise ValueError(
            f"centroid_ids length ({len(centroid_ids)}) != number of "
            f"centroids ({centroids.shape[0]})"
        )
    if embeddings.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"embedding dim ({embeddings.shape[1]}) != centroid dim "
            f"({centroids.shape[1]})"
        )
    missing = {cid for cid in assigned_clusters if cid not in calibration}
    if missing:
        raise ValueError(
            f"assigned_clusters references uncalibrated ids: {sorted(missing)}"
        )

    cid_to_idx = {cid: j for j, cid in enumerate(centroid_ids)}
    m = embeddings.shape[0]
    mah = np.empty(m, dtype=np.float64)
    z = np.empty(m, dtype=np.float64)
    is_ood = np.empty(m, dtype=bool)
    for i, cid in enumerate(assigned_clusters):
        entry = calibration[cid]
        diff = embeddings[i] - centroids[cid_to_idx[cid]]
        mah[i] = float(((diff ** 2) / (entry["diag_var"] + MAHALANOBIS_REG_COVAR)).sum())
        is_ood[i] = mah[i] > entry["d2_95"]
        sigma = entry["std"]
        if sigma > 0.0:
            z[i] = (mah[i] - entry["mean"]) / sigma
        else:
            z[i] = 0.0 if mah[i] == entry["mean"] else np.inf

    per_cluster: dict[str, dict] = {}
    for cid in set(assigned_clusters):
        mask = np.array([c == cid for c in assigned_clusters])
        per_cluster[cid] = {
            "n": int(mask.sum()),
            "ood_rate": float(is_ood[mask].mean()),
            "mean_z": float(z[mask].mean()),
        }

    return {
        "z": z,
        "is_ood": is_ood,
        "ood_rate": float(is_ood.mean()),
        "mean_z": float(z.mean()),
        "per_cluster": per_cluster,
    }


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
    diag_vars: np.ndarray,
    log_weights: np.ndarray,
    db: Session,
) -> tuple[int, list[DbSoftAssignment]]:
    """Admit new, already-embedded points into a converged session (read-only).

    This is the operational meaning of *generalization*: once the oracle is
    happy, new data arrives into the running system and must be placed without
    disturbing what the oracle accepted. New points are assigned via
    **GMM-posterior argmax** against the frozen convergence parameters — the
    same decision rule the GMM engine used at training time — and their soft
    probabilities are the normalised GMM posteriors (no temperature hack).

    Read-only-assignment policy
    ---------------------------
    * All GMM parameters are **frozen**: the caller passes ``(centroid_ids,
      centroids, diag_vars, log_weights)`` built once with
      :func:`gmm_params_from_snapshot`; they are **not** recomputed here and
      do **not** drift as successive batches arrive.
    * Pre-existing points are **carried forward verbatim** — their soft
      assignments from the latest snapshot are copied unchanged to the new turn.
    * Only the new points get fresh rows. Their soft probabilities are the
      normalised GMM posteriors, so a new point far from all cluster means (or
      landing in a very tight cluster) earns a low max-probability and surfaces
      in the coherence judge's bottom-2 stress sample.

    Args:
        session_id: the converged session to ingest into.
        new_points: NEW :class:`~src.models.DataPoint` objects that already
            carry an ``embedding``. They are added to ``db`` here.
        centroid_ids: cluster ids matching ``centroids`` rows (frozen).
        centroids: ``(k, d)`` frozen convergence means.
        diag_vars: ``(k, d)`` frozen per-cluster diagonal variance.
        log_weights: ``(k,)`` frozen log mixing weights.
        db: SQLAlchemy session. Rows are **staged, not committed** — the caller
            owns the transaction.

    Returns:
        ``(new_turn, new_assignments)`` — the turn the snapshot was written at
        and every :class:`~src.models.SoftAssignment` row created
        (carried-forward + new).

    Raises:
        ValueError: no new point has an embedding; the session has no existing
            clustering; or an embedding/centroid dimensionality mismatch.
    """
    embedded = [dp for dp in new_points if dp.embedding is not None]
    if not embedded:
        raise ValueError("no new point has an embedding — embed before ingesting")
    X = np.asarray([dp.embedding for dp in embedded], dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    diag_vars = np.asarray(diag_vars, dtype=np.float64)
    log_weights = np.asarray(log_weights, dtype=np.float64)

    # GMM-posterior assignment — consistent with how the engine assigned base points.
    nearest = assign_gmm_posterior(X, centroid_ids, centroids, diag_vars, log_weights)

    # Soft probabilities: normalised GMM posteriors (no temperature parameter).
    reg = diag_vars + MAHALANOBIS_REG_COVAR  # (k, d)
    diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (m, k, d)
    mah_sq = ((diffs ** 2) / reg[np.newaxis, :, :]).sum(axis=2)  # (m, k)
    log_det_term = 0.5 * np.log(reg).sum(axis=1)  # (k,)
    log_scores = log_weights[np.newaxis, :] - 0.5 * mah_sq - log_det_term[np.newaxis, :]
    log_scores -= log_scores.max(axis=1, keepdims=True)
    exp = np.exp(log_scores)
    probs = exp / exp.sum(axis=1, keepdims=True)  # (m, k)

    if [centroid_ids[j] for j in probs.argmax(axis=1)] != nearest:
        raise RuntimeError(
            "GMM soft assignment disagrees with hard posterior argmax — numerical bug"
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
