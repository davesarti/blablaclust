"""Cluster operations that execute oracle-requested changes: merge / split / rename.

When the oracle gives feedback ("merge A and B", "split C", "call this cluster
X"), P3's ``f_apply_operations`` decides *which* operation to run; the functions
here perform the actual data-point reassignment with k-means and persist the
result. They are the executor half of the conversational loop, on the
clustering side.

Transaction ownership
---------------------
Every function stages its changes on the passed ``db`` session (adds new rows,
mutates existing clusters) but never calls ``commit()`` — the caller owns the
transaction, so a turn that applies several operations stays atomic. This
mirrors ``initial_clustering``: the engine builds the objects, the caller
commits. Validation always runs before any mutation, so a failed operation
leaves the session untouched.

Snapshot model
--------------
``SoftAssignment.turn_number`` identifies a *complete* snapshot of the
clustering: at any turn every data point has a probability for each cluster it
belongs to, and ``f_uncertainty`` / ``hard_cluster_stats`` read
``max(turn_number)`` expecting it to be complete. An operation at turn N
therefore writes a full snapshot at N — the points it reassigns get fresh
probabilities, every other point is carried forward from the previous snapshot.
Because of this the target ``turn_number`` must be strictly greater than the
latest existing snapshot turn, otherwise carried-forward rows would collide on
the ``(data_point_id, cluster_id, turn_number)`` primary key.
"""

import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.engine.cluster_naming import name_clusters
from src.engine.initial_clustering import initial_clustering
from src.models import Cluster as DbCluster, DataPoint, SoftAssignment as DbSoftAssignment


def _load_latest_snapshot(
    session_id: str, db: Session
) -> tuple[int, dict[str, dict[str, float]]]:
    """Return ``(turn, snapshot)`` for the session's most recent clustering.

    ``snapshot`` maps ``data_point_id -> {cluster_id: probability}``.

    Raises:
        ValueError: the session has no clusters or no soft assignments yet.
    """
    cluster_ids = [
        cid
        for (cid,) in db.query(DbCluster.id)
        .filter(DbCluster.session_id == session_id)
        .all()
    ]
    if not cluster_ids:
        raise ValueError(
            f"session '{session_id}' has no clusters — run initial clustering first"
        )

    latest_turn = (
        db.query(func.max(DbSoftAssignment.turn_number))
        .filter(DbSoftAssignment.cluster_id.in_(cluster_ids))
        .scalar()
    )
    if latest_turn is None:
        raise ValueError(
            f"session '{session_id}' has no soft assignments — "
            "run initial clustering first"
        )

    rows = (
        db.query(DbSoftAssignment)
        .filter(
            DbSoftAssignment.cluster_id.in_(cluster_ids),
            DbSoftAssignment.turn_number == latest_turn,
        )
        .all()
    )
    snapshot: dict[str, dict[str, float]] = {}
    for row in rows:
        snapshot.setdefault(row.data_point_id, {})[row.cluster_id] = row.probability
    return latest_turn, snapshot


def _hard_cluster(distribution: dict[str, float]) -> str:
    """The cluster a point belongs to — the one with the highest probability."""
    return max(distribution, key=distribution.get)


def merge_clusters(
    cluster_ids: list[str],
    session_id: str,
    turn_number: int,
    db: Session,
    auto_name: bool = True,
) -> DbCluster:
    """Merge two or more clusters into one.

    Pools every data point belonging to ``cluster_ids``, dissolves those
    clusters, creates a single new cluster, and writes a fresh soft-assignment
    snapshot at ``turn_number``: pooled points are assigned to the new cluster
    with probability 1.0, every other point is carried forward (its probability
    mass on the merged clusters is folded into the new cluster).

    Args:
        cluster_ids: IDs of the clusters to merge (at least 2 distinct).
        session_id: Session that owns the clusters.
        turn_number: Turn at which the merge is recorded. Must be strictly
            greater than the latest existing snapshot turn.
        db: SQLAlchemy session (changes staged but not committed).
        auto_name: When True (default), the merged cluster is labelled by the
            LLM (via ``name_clusters``) from its pooled points, so it is named
            at the moment of creation. When False the cluster keeps the generic
            ``"Merge of A + B"`` placeholder. Naming is best-effort — a failed
            LLM call leaves the placeholder and never aborts the merge.

    Returns:
        The new cluster.

    Raises:
        ValueError: fewer than 2 distinct clusters; an unknown or
            already-dissolved cluster; no existing clustering; or a
            ``turn_number`` that is not strictly after the latest snapshot.
    """
    merge_ids = list(dict.fromkeys(cluster_ids))  # dedupe, preserve order
    if len(merge_ids) < 2:
        raise ValueError("merge_clusters needs at least 2 distinct clusters")

    clusters = (
        db.query(DbCluster)
        .filter(DbCluster.id.in_(merge_ids), DbCluster.session_id == session_id)
        .all()
    )
    by_id = {c.id: c for c in clusters}
    missing = [cid for cid in merge_ids if cid not in by_id]
    if missing:
        raise ValueError(f"clusters not found in session '{session_id}': {missing}")
    dissolved = [cid for cid in merge_ids if by_id[cid].dissolved_at_turn is not None]
    if dissolved:
        raise ValueError(f"cannot merge already-dissolved clusters: {dissolved}")

    prev_turn, snapshot = _load_latest_snapshot(session_id, db)
    if turn_number <= prev_turn:
        raise ValueError(
            f"turn_number ({turn_number}) must be greater than the latest "
            f"snapshot turn ({prev_turn})"
        )

    merge_set = set(merge_ids)
    new_cluster = DbCluster(
        id=str(uuid.uuid4()),
        session_id=session_id,
        name=("Merge of " + " + ".join(by_id[cid].name for cid in merge_ids))[:255],
        description="",
        created_at_turn=turn_number,
    )

    # Dissolve the merged clusters as of this turn.
    for cid in merge_ids:
        by_id[cid].dissolved_at_turn = turn_number

    # Write the full snapshot at turn_number.
    new_assignments: list[DbSoftAssignment] = []
    for point_id, distribution in snapshot.items():
        if _hard_cluster(distribution) in merge_set:
            # Pooled point — now belongs entirely to the merged cluster.
            new_assignments.append(
                DbSoftAssignment(
                    data_point_id=point_id,
                    cluster_id=new_cluster.id,
                    turn_number=turn_number,
                    probability=1.0,
                )
            )
        else:
            # Untouched point — carry forward only the non-merged-cluster mass.
            #
            # We deliberately DROP the probability mass that the merged clusters
            # used to hold for this point. Folding it into the new cluster
            # (`merged_mass = sum(prob[cid] for cid in merge_set)`) sounds
            # symmetric but breaks the hard partition: k-means soft assignments
            # in high-dim sentence-transformer space are very flat (e.g. 5
            # clusters → ~0.20 each), so the sum of two merged probabilities
            # routinely exceeds the un-merged argmax and the entire dataset
            # collapses into the merged cluster after a single merge op.
            # Dropping the mass preserves each un-pooled point's original hard
            # cluster. Probabilities for these points no longer sum to 1, but
            # the snapshot stays internally consistent (pooled points are 1.0
            # on the new cluster) and the argmax is what downstream UI reads.
            for cid, prob in distribution.items():
                if cid in merge_set:
                    continue
                new_assignments.append(
                    DbSoftAssignment(
                        data_point_id=point_id,
                        cluster_id=cid,
                        turn_number=turn_number,
                        probability=prob,
                    )
                )

    db.add(new_cluster)
    for assignment in new_assignments:
        db.add(assignment)

    # Name the merged cluster at creation time so naming is propagated by the
    # operation itself, not bolted on afterwards by the caller.
    if auto_name:
        merged_point_ids = [
            a.data_point_id for a in new_assignments if a.cluster_id == new_cluster.id
        ]
        merged_points = (
            db.query(DataPoint).filter(DataPoint.id.in_(merged_point_ids)).all()
        )
        name_clusters([new_cluster], new_assignments, merged_points)

    return new_cluster


def split_cluster(
    cluster_id: str,
    session_id: str,
    turn_number: int,
    db: Session,
    k: int = 2,
    auto_name: bool = True,
) -> list[DbCluster]:
    """Split one cluster into ``k`` sub-clusters using k-means on its members.

    Takes the data points whose hard assignment is ``cluster_id``, dissolves
    that cluster, runs real k-means with the requested ``k`` on the subset,
    and writes a fresh soft-assignment snapshot at ``turn_number``: subset
    points get the new k-means probabilities, every other point is carried
    forward (re-normalised after dropping the dissolved cluster).

    Args:
        cluster_id: ID of the cluster to split.
        session_id: Session that owns the cluster.
        turn_number: Turn at which the split is recorded. Must be strictly
            greater than the latest existing snapshot turn.
        db: SQLAlchemy session (changes staged but not committed).
        k: Number of sub-clusters to produce. Must be >= 2 (default: 2).
        auto_name: When True (default), the child clusters are labelled by the
            LLM (via ``name_clusters``) from their representative points, so
            they are named at the moment of creation. When False the children
            keep the generic ``"<parent> - part N"`` placeholder. Naming is
            best-effort — a failed LLM call leaves the placeholders and never
            aborts the split.

    Returns:
        The ``k`` newly created child clusters.

    Raises:
        ValueError: ``k`` is less than 2; unknown or already-dissolved
            cluster; fewer than ``k`` points assigned to it; no existing
            clustering; or a ``turn_number`` that is not strictly after the
            latest snapshot.
    """
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    cluster = (
        db.query(DbCluster)
        .filter(DbCluster.id == cluster_id, DbCluster.session_id == session_id)
        .first()
    )
    if cluster is None:
        raise ValueError(f"cluster '{cluster_id}' not found in session '{session_id}'")
    if cluster.dissolved_at_turn is not None:
        raise ValueError(f"cannot split already-dissolved cluster '{cluster_id}'")

    prev_turn, snapshot = _load_latest_snapshot(session_id, db)
    if turn_number <= prev_turn:
        raise ValueError(
            f"turn_number ({turn_number}) must be greater than the latest "
            f"snapshot turn ({prev_turn})"
        )

    subset_ids = [
        pid for pid, dist in snapshot.items() if _hard_cluster(dist) == cluster_id
    ]
    if len(subset_ids) < k:
        raise ValueError(
            f"cannot split cluster '{cluster_id}': {len(subset_ids)} point(s) "
            f"assigned to it (need at least {k})"
        )

    subset_points = db.query(DataPoint).filter(DataPoint.id.in_(subset_ids)).all()

    # Real k-means on the subset — initial_clustering builds the k new
    # clusters and their soft assignments at turn_number. The split path does
    # not surface a silhouette, so we ignore the third return value.
    new_clusters, subset_assignments, _ = initial_clustering(
        data_points=subset_points,
        k=k,
        session_id=session_id,
        turn_number=turn_number,
    )
    for index, child in enumerate(new_clusters, start=1):
        child.name = f"{cluster.name} - part {index}"[:255]

    # Name the children at creation time so naming is propagated by the
    # operation itself; the "part N" labels above are the fallback if the LLM
    # call fails. subset_assignments / subset_points are already in memory.
    if auto_name:
        name_clusters(new_clusters, subset_assignments, subset_points)

    # Dissolve the parent cluster as of this turn.
    cluster.dissolved_at_turn = turn_number

    # Full snapshot: subset points handled by initial_clustering; carry the rest forward.
    new_assignments: list[DbSoftAssignment] = list(subset_assignments)
    subset_set = set(subset_ids)
    for point_id, distribution in snapshot.items():
        if point_id in subset_set:
            continue
        kept = {cid: p for cid, p in distribution.items() if cid != cluster_id}
        total = sum(kept.values())
        if total <= 0.0:
            continue  # point had mass only on the split cluster (already in subset)
        for cid, prob in kept.items():
            new_assignments.append(
                DbSoftAssignment(
                    data_point_id=point_id,
                    cluster_id=cid,
                    turn_number=turn_number,
                    probability=prob / total,
                )
            )

    for child in new_clusters:
        db.add(child)
    for assignment in new_assignments:
        db.add(assignment)
    return new_clusters


def move_points(
    point_ids: list[str],
    target_cluster_id: str,
    session_id: str,
    turn_number: int,
    db: Session,
) -> DbCluster:
    """Reassign specific data points to an existing cluster.

    Handles point-level oracle feedback ("this review belongs in A, not B") and
    resolves boundary points surfaced by f_uncertainty. Writes a fresh
    soft-assignment snapshot at ``turn_number``: moved points get probability
    1.0 on the target cluster (a hard move — their previous mass is dropped),
    every other point is carried forward unchanged. Any active cluster left
    holding no probability mass after the move is dissolved as of this turn.

    Returns the target cluster. Does not commit — the caller owns the transaction.

    Raises:
        ValueError: empty ``point_ids``; unknown or already-dissolved target
            cluster; a point id not present in the latest snapshot; no existing
            clustering; or a ``turn_number`` not strictly after the latest
            snapshot.
    """
    if not point_ids:
        raise ValueError("move_points needs at least 1 point")

    target = (
        db.query(DbCluster)
        .filter(DbCluster.id == target_cluster_id, DbCluster.session_id == session_id)
        .first()
    )
    if target is None:
        raise ValueError(
            f"target cluster '{target_cluster_id}' not found in session '{session_id}'"
        )
    if target.dissolved_at_turn is not None:
        raise ValueError(
            f"cannot move points into dissolved cluster '{target_cluster_id}'"
        )

    prev_turn, snapshot = _load_latest_snapshot(session_id, db)
    if turn_number <= prev_turn:
        raise ValueError(
            f"turn_number ({turn_number}) must be greater than the latest "
            f"snapshot turn ({prev_turn})"
        )

    move_set = set(point_ids)
    missing = [pid for pid in move_set if pid not in snapshot]
    if missing:
        raise ValueError(f"points not found in current snapshot: {missing}")

    # Build the new full snapshot: moved points collapse to the target cluster,
    # everyone else carries forward unchanged.
    new_snapshot: dict[str, dict[str, float]] = {}
    for point_id, distribution in snapshot.items():
        if point_id in move_set:
            new_snapshot[point_id] = {target_cluster_id: 1.0}
        else:
            new_snapshot[point_id] = dict(distribution)

    for point_id, distribution in new_snapshot.items():
        for cid, prob in distribution.items():
            db.add(
                DbSoftAssignment(
                    data_point_id=point_id,
                    cluster_id=cid,
                    turn_number=turn_number,
                    probability=prob,
                )
            )

    # Dissolve any active cluster that holds no probability mass after the move.
    # A hard move drops the source entries entirely, so an emptied cluster
    # leaves no dangling assignments to carry forward. The target always retains
    # mass (the moved points), so it never dissolves itself.
    clusters_with_mass = {cid for dist in new_snapshot.values() for cid in dist}
    active_clusters = (
        db.query(DbCluster)
        .filter(
            DbCluster.session_id == session_id,
            DbCluster.dissolved_at_turn.is_(None),
        )
        .all()
    )
    for cluster in active_clusters:
        if cluster.id not in clusters_with_mass:
            cluster.dissolved_at_turn = turn_number

    return target


def rename_cluster(
    cluster_id: str,
    new_name: str,
    new_description: str,
    db: Session,
) -> DbCluster:
    """Rename a cluster — updates name and description only.

    No k-means, no soft-assignment changes. Does not commit — the caller owns
    the transaction.

    Raises:
        ValueError: the cluster does not exist.
    """
    cluster = db.query(DbCluster).filter(DbCluster.id == cluster_id).first()
    if cluster is None:
        raise ValueError(f"cluster '{cluster_id}' not found")
    cluster.name = new_name[:255]
    cluster.description = new_description
    return cluster
