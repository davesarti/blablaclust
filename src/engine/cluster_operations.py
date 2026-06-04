"""Cluster operations that execute oracle-requested changes: merge / split / rename.

When the oracle gives feedback ("merge A and B", "split C", "call this cluster
X"), P3's ``f_apply_operations`` decides *which* operation to run; the functions
here perform the actual data-point reassignment with k-means and stage the
result on a :class:`~src.engine.turn_builder.TurnBuilder`. The executor half
of the conversational loop, on the clustering side.

Builder model
-------------
Every function in this module mutates the passed ``builder`` rather than the
DB session: snapshot updates go into ``builder.snapshot``, new clusters into
``builder.new_clusters``, dissolutions into ``builder.dissolved_ids``. The
caller (``f_apply_operations``) commits the builder ONCE at the end of the
conversation turn. That keeps ``Cluster.created_at_turn``,
``Cluster.dissolved_at_turn``, ``SoftAssignment.turn_number``, and
``Turn.turn_number`` in lockstep — no more snapshot-axis vs conv-axis drift.

Snapshot model
--------------
``builder.snapshot`` represents the *complete* clustering at this turn: every
data point has a probability for each cluster it belongs to. The builder is
preloaded with the previous turn's snapshot and each op mutates it forward.
Ops that touch a subset of points (merge, split, batch_move_points) update
only that subset and leave the rest carried-forward; ``semantic_clustering``
replaces the snapshot wholesale.
"""

import uuid

import numpy as np

from src.engine.cluster_naming import name_clusters
from src.engine.initial_clustering import _fit_gmm, _fit_kmeans, _softmax, initial_clustering
from src.engine.turn_builder import TurnBuilder
from src.models import Cluster as DbCluster, DataPoint, SoftAssignment as DbSoftAssignment


def _hard_cluster(distribution: dict[str, float]) -> str:
    """The cluster a point belongs to — the one with the highest probability."""
    return max(distribution, key=distribution.get)


def _renormalize(distribution: dict[str, float]) -> dict[str, float]:
    total = sum(distribution.values())
    if total < 1e-12:
        return distribution
    return {cid: p / total for cid, p in distribution.items()}


def merge_clusters(
    cluster_ids: list[str],
    builder: TurnBuilder,
    auto_name: bool = True,
    axis_hint: str | None = None,
) -> DbCluster:
    """Merge two or more clusters into one.

    Pools every data point belonging to ``cluster_ids``, stages the source
    clusters as dissolved, adds a single new cluster to the builder, and
    updates the in-memory snapshot: pooled points are assigned to the new
    cluster with probability 1.0, every other point is carried forward (its
    probability mass on the merged clusters is dropped).

    Args:
        cluster_ids: IDs of the clusters to merge (at least 2 distinct).
        builder: The conversation turn's in-memory staging area.
        auto_name: When True (default), the merged cluster is labelled by the
            LLM from its pooled points. Naming is best-effort — a failed LLM
            call leaves the placeholder and never aborts the merge.
        axis_hint: When provided, forwarded to ``name_clusters`` so the merged
            cluster reflects the session's semantic axis.

    Returns:
        The new cluster (also added to ``builder.new_clusters``).

    Raises:
        ValueError: fewer than 2 distinct clusters; an unknown or
            already-dissolved cluster; or an empty prior snapshot.
    """
    merge_ids = list(dict.fromkeys(cluster_ids))  # dedupe, preserve order
    if len(merge_ids) < 2:
        raise ValueError("merge_clusters needs at least 2 distinct clusters")

    by_id: dict[str, DbCluster] = {}
    missing: list[str] = []
    for cid in merge_ids:
        cluster = builder.get_cluster(cid)
        if cluster is None:
            missing.append(cid)
        else:
            by_id[cid] = cluster
    if missing:
        raise ValueError(
            f"clusters not found in session '{builder.session_id}': {missing}"
        )
    dissolved = [cid for cid in merge_ids if builder.is_dissolved(cid)]
    if dissolved:
        raise ValueError(f"cannot merge already-dissolved clusters: {dissolved}")

    if not builder.snapshot:
        raise ValueError(
            f"session '{builder.session_id}' has no soft assignments — "
            "run initial clustering first"
        )

    merge_set = set(merge_ids)
    new_cluster = DbCluster(
        id=str(uuid.uuid4()),
        session_id=builder.session_id,
        name=("Merge of " + " + ".join(by_id[cid].name for cid in merge_ids))[:255],
        description="",
        created_at_turn=builder.turn_number,
    )

    # Update the in-memory snapshot.
    #
    # We deliberately DROP the probability mass that the merged clusters used
    # to hold for unpooled points. Folding it into the new cluster
    # (`merged_mass = sum(p[cid] for cid in merge_set)`) sounds symmetric but
    # breaks the hard partition: soft assignments in high-dim sentence-
    # transformer space are very flat (e.g. 5 clusters → ~0.20 each), so the
    # sum of two merged probabilities routinely exceeds the un-merged argmax
    # and the entire dataset would collapse into the merged cluster after a
    # single merge op. Dropping the mass preserves each un-pooled point's
    # original hard cluster. Probabilities for these points no longer sum to
    # 1, but the snapshot stays internally consistent (pooled points are 1.0
    # on the new cluster) and the argmax is what downstream UI reads.
    for point_id, distribution in list(builder.snapshot.items()):
        if _hard_cluster(distribution) in merge_set:
            builder.snapshot[point_id] = {new_cluster.id: 1.0}
        else:
            remaining = {cid: p for cid, p in distribution.items() if cid not in merge_set}
            builder.snapshot[point_id] = _renormalize(remaining)

    for cid in merge_ids:
        builder.dissolve(cid)
    builder.add_cluster(new_cluster)

    if auto_name:
        merged_point_ids = [
            pid
            for pid, dist in builder.snapshot.items()
            if _hard_cluster(dist) == new_cluster.id
        ]
        merged_points = (
            builder.db.query(DataPoint).filter(DataPoint.id.in_(merged_point_ids)).all()
        )
        # name_clusters reads .probability / .data_point_id / .cluster_id off
        # SoftAssignment objects; transient instances work as data carriers
        # without being added to the DB session.
        naming_assignments = [
            DbSoftAssignment(
                data_point_id=pid,
                cluster_id=new_cluster.id,
                turn_number=builder.turn_number,
                probability=1.0,
            )
            for pid in merged_point_ids
        ]
        name_clusters(
            [new_cluster], naming_assignments, merged_points, axis_hint=axis_hint
        )

    return new_cluster


def split_cluster(
    cluster_id: str,
    builder: TurnBuilder,
    k: int = 2,
    auto_name: bool = True,
    axis_hint: str | None = None,
) -> list[DbCluster]:
    """Split one cluster into ``k`` sub-clusters using GMM (k-means fallback).

    Takes the data points whose hard assignment is ``cluster_id``, stages the
    cluster as dissolved, runs GMM with the requested ``k`` on the subset via
    ``initial_clustering``, and updates the in-memory snapshot: subset points
    get fresh soft probabilities, every other point is carried forward
    (re-normalised after dropping the dissolved cluster).

    Raises:
        ValueError: ``k`` is less than 2; unknown or already-dissolved
            cluster; fewer than ``k`` points assigned to it; or an empty
            prior snapshot.
    """
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    cluster = builder.get_cluster(cluster_id)
    if cluster is None:
        raise ValueError(
            f"cluster '{cluster_id}' not found in session '{builder.session_id}'"
        )
    if builder.is_dissolved(cluster_id):
        raise ValueError(f"cannot split already-dissolved cluster '{cluster_id}'")
    if not builder.snapshot:
        raise ValueError(
            f"session '{builder.session_id}' has no soft assignments — "
            "run initial clustering first"
        )

    subset_ids = [
        pid
        for pid, dist in builder.snapshot.items()
        if _hard_cluster(dist) == cluster_id
    ]
    if len(subset_ids) < k:
        raise ValueError(
            f"cannot split cluster '{cluster_id}': {len(subset_ids)} point(s) "
            f"assigned to it (need at least {k})"
        )

    subset_points = (
        builder.db.query(DataPoint).filter(DataPoint.id.in_(subset_ids)).all()
    )

    # Real k-means on the subset. initial_clustering builds the k new clusters
    # and their soft assignments at builder.turn_number; we then merge those
    # subset assignments into our in-memory snapshot.
    new_clusters, subset_assignments, _ = initial_clustering(
        data_points=subset_points,
        k=k,
        session_id=builder.session_id,
        turn_number=builder.turn_number,
    )
    for index, child in enumerate(new_clusters, start=1):
        child.name = f"{cluster.name} - part {index}"[:255]

    if auto_name:
        name_clusters(new_clusters, subset_assignments, subset_points, axis_hint=axis_hint)

    # Subset distributions from k-means
    subset_dist: dict[str, dict[str, float]] = {}
    for a in subset_assignments:
        subset_dist.setdefault(a.data_point_id, {})[a.cluster_id] = a.probability

    # Update the snapshot: subset points get fresh distributions; others have
    # any residual mass on the dissolved parent dropped.
    for point_id, distribution in list(builder.snapshot.items()):
        if point_id in subset_dist:
            builder.snapshot[point_id] = subset_dist[point_id]
        else:
            remaining = {cid: p for cid, p in distribution.items() if cid != cluster_id}
            builder.snapshot[point_id] = _renormalize(remaining)

    builder.dissolve(cluster_id)
    for child in new_clusters:
        builder.add_cluster(child)

    return new_clusters


def batch_move_points(
    moves: list[tuple[str, str]],
    builder: TurnBuilder,
) -> None:
    """Reassign specific data points to clusters in a single batch.

    Handles every form of point-level reassignment — oracle feedback ("this
    review belongs in A, not B"), boundary-repair corrections, uncertainty
    resolution. Each moved point collapses to ``{target: 1.0}`` in the
    in-memory snapshot (its previous mass is dropped), every other point is
    carried forward unchanged, and any active cluster left without probability
    mass is staged as dissolved.

    If the same point appears in multiple pairs the last one wins.

    Raises:
        ValueError: empty ``moves``; an unknown or already-dissolved target
            cluster; a point id not present in the current snapshot; or an
            empty prior snapshot.
    """
    if not moves:
        raise ValueError("batch_move_points needs at least 1 move")
    if not builder.snapshot:
        raise ValueError(
            f"session '{builder.session_id}' has no soft assignments — "
            "run initial clustering first"
        )

    target_ids = {target for _, target in moves}
    missing: list[str] = []
    dissolved: list[str] = []
    for tid in target_ids:
        cluster = builder.get_cluster(tid)
        if cluster is None:
            missing.append(tid)
        elif builder.is_dissolved(tid):
            dissolved.append(tid)
    if missing:
        raise ValueError(
            f"target cluster(s) not found in session '{builder.session_id}': {missing}"
        )
    if dissolved:
        raise ValueError(
            f"cannot move points into dissolved cluster(s): {dissolved}"
        )

    move_map: dict[str, str] = {pid: target for pid, target in moves}
    not_in_snapshot = [pid for pid in move_map if pid not in builder.snapshot]
    if not_in_snapshot:
        raise ValueError(f"points not found in current snapshot: {not_in_snapshot}")

    for point_id, target in move_map.items():
        builder.snapshot[point_id] = {target: 1.0}

    # Dissolve any active cluster that holds no mass after the move. The
    # target always retains mass (the moved points), so it never dissolves
    # itself.
    clusters_with_mass = {
        cid for dist in builder.snapshot.values() for cid in dist
    }
    for cluster in builder.active_clusters():
        if cluster.id not in clusters_with_mass:
            builder.dissolve(cluster.id)


def rename_cluster(
    cluster_id: str,
    new_name: str,
    new_description: str,
    builder: TurnBuilder,
) -> DbCluster:
    """Rename a cluster — updates name and description only.

    No k-means, no snapshot changes. Mutates the cluster object directly
    (whether it's already in DB or staged in the builder); the builder commit
    flushes the mutation when the rest of the turn does.

    Raises:
        ValueError: the cluster does not exist.
    """
    cluster = builder.get_cluster(cluster_id)
    if cluster is None:
        raise ValueError(f"cluster '{cluster_id}' not found")
    cluster.name = new_name[:255]
    cluster.description = new_description
    return cluster


def auto_name_cluster(
    cluster_id: str,
    builder: TurnBuilder,
    axis_hint: str | None = None,
) -> DbCluster:
    """Re-run the naming LLM on a single existing cluster.

    Used when the oracle issues a bare rename (no name, no description) and
    really means "give this cluster a better name from its current contents".

    Raises:
        ValueError: the cluster does not exist.
    """
    cluster = builder.get_cluster(cluster_id)
    if cluster is None:
        raise ValueError(f"cluster '{cluster_id}' not found")

    member_ids = [
        pid for pid, dist in builder.snapshot.items()
        if _hard_cluster(dist) == cluster_id
    ]
    if not member_ids:
        return cluster

    points = (
        builder.db.query(DataPoint).filter(DataPoint.id.in_(member_ids)).all()
    )
    naming_assignments = [
        DbSoftAssignment(
            data_point_id=pid,
            cluster_id=cluster_id,
            turn_number=builder.turn_number,
            probability=1.0,
        )
        for pid in member_ids
    ]
    name_clusters([cluster], naming_assignments, points, axis_hint=axis_hint)
    return cluster


def semantic_reembed_cluster(
    cluster_id: str,
    axis_hint: str,
    builder: TurnBuilder,
    k: int = 2,
    auto_name: bool = True,
) -> list[DbCluster]:
    """Re-embed a single cluster along a semantic axis and split into k sub-clusters.

    Unlike split_cluster (plain k-means geometry), this projects the subset into
    a hybrid embedding space oriented by axis_hint before clustering, so the
    resulting sub-clusters reflect the semantic axis rather than raw distance.
    All other clusters and their soft assignments are unaffected (non-subset
    points are renormalized after the parent is dissolved).

    Raises:
        ValueError: unknown/dissolved cluster; fewer than k points; empty snapshot.
        AxisNotDiscriminativeError: the axis doesn't vary enough in the subset.
    """
    from src.engine.f_semantic_reembed import reembed_for_axis

    cluster = builder.get_cluster(cluster_id)
    if cluster is None:
        raise ValueError(
            f"cluster '{cluster_id}' not found in session '{builder.session_id}'"
        )
    if builder.is_dissolved(cluster_id):
        raise ValueError(f"cannot re-embed already-dissolved cluster '{cluster_id}'")
    if not builder.snapshot:
        raise ValueError(
            f"session '{builder.session_id}' has no soft assignments — "
            "run initial clustering first"
        )

    subset_ids = [
        pid for pid, dist in builder.snapshot.items()
        if _hard_cluster(dist) == cluster_id
    ]
    if len(subset_ids) < k:
        raise ValueError(
            f"cannot split cluster '{cluster_id}': {len(subset_ids)} point(s) "
            f"assigned to it (need at least {k})"
        )

    subset_points = (
        builder.db.query(DataPoint).filter(DataPoint.id.in_(subset_ids)).all()
    )

    X, _ = reembed_for_axis(subset_points, axis_hint)

    try:
        _, probs = _fit_gmm(X, k)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "GMM failed in semantic_reembed_cluster (k=%d, n=%d, reason=%s) — falling back to k-means",
            k, len(subset_points), exc,
        )
        model = _fit_kmeans(X, k)
        centroids = model.cluster_centers_
        diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        probs = _softmax(-sq_dists, axis=1)

    new_cluster_ids = [str(uuid.uuid4()) for _ in range(k)]
    new_clusters = [
        DbCluster(
            id=new_cluster_ids[i],
            session_id=builder.session_id,
            name=f"{cluster.name} - part {i + 1}"[:255],
            description="",
            created_at_turn=builder.turn_number,
        )
        for i in range(k)
    ]

    subset_dist: dict[str, dict[str, float]] = {}
    naming_assignments: list[DbSoftAssignment] = []
    for i, dp in enumerate(subset_points):
        dist = {new_cluster_ids[j]: float(probs[i, j]) for j in range(k)}
        subset_dist[dp.id] = dist
        for j in range(k):
            naming_assignments.append(DbSoftAssignment(
                data_point_id=dp.id,
                cluster_id=new_cluster_ids[j],
                turn_number=builder.turn_number,
                probability=float(probs[i, j]),
            ))

    for point_id, distribution in list(builder.snapshot.items()):
        if point_id in subset_dist:
            builder.snapshot[point_id] = subset_dist[point_id]
        else:
            remaining = {cid: p for cid, p in distribution.items() if cid != cluster_id}
            builder.snapshot[point_id] = _renormalize(remaining)

    builder.dissolve(cluster_id)
    for child in new_clusters:
        builder.add_cluster(child)

    if auto_name:
        name_clusters(new_clusters, naming_assignments, subset_points, axis_hint=axis_hint)

    return new_clusters
