"""Unit tests for cluster operations: merge / split / rename.

These run entirely against an in-memory SQLite database — no LLM calls — so the
clustering maths (k-means via initial_clustering) and the DB persistence are
both exercised for real.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.engine.cluster_operations import merge_clusters, rename_cluster, split_cluster
from src.engine.initial_clustering import initial_clustering
from src.models import Base, ChatSession, Cluster, DataPoint, SoftAssignment

SESSION_ID = "sess-ops"

# Six 2-D points in three well-separated groups → k-means k=3 finds them cleanly,
# two points per cluster.
_POINTS = {
    "a1": [0.0, 0.0],
    "a2": [0.2, 0.1],
    "b1": [9.0, 9.0],
    "b2": [9.1, 8.8],
    "c1": [0.0, 9.0],
    "c2": [0.1, 9.2],
}


@pytest.fixture
def db():
    """In-memory DB seeded with one session, six points, and a k=3 clustering at turn 0."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()

    session.add(
        ChatSession(
            id=SESSION_ID,
            dataset_name="ds",
            embedding_model="default",
            status="active",
        )
    )
    data_points = []
    for point_id, embedding in _POINTS.items():
        dp = DataPoint(
            id=point_id, dataset_name="ds", data={"text": point_id}, embedding=embedding
        )
        data_points.append(dp)
        session.add(dp)

    clusters, assignments = initial_clustering(
        data_points=data_points, k=3, session_id=SESSION_ID, turn_number=0
    )
    for cluster in clusters:
        session.add(cluster)
    for assignment in assignments:
        session.add(assignment)
    session.commit()

    yield session
    session.close()


def _hard_clusters(db, turn: int) -> dict[str, str]:
    """Map point_id -> hard cluster_id from the soft assignments at `turn`."""
    rows = db.query(SoftAssignment).filter(SoftAssignment.turn_number == turn).all()
    distributions: dict[str, dict[str, float]] = {}
    for row in rows:
        distributions.setdefault(row.data_point_id, {})[row.cluster_id] = row.probability
    return {pid: max(d, key=d.get) for pid, d in distributions.items()}


def _active_clusters(db) -> list[Cluster]:
    return (
        db.query(Cluster)
        .filter(Cluster.session_id == SESSION_ID, Cluster.dissolved_at_turn.is_(None))
        .all()
    )


# --- merge_clusters ---------------------------------------------------------


def test_merge_dissolves_old_and_creates_new(db):
    cluster_ids = [c.id for c in _active_clusters(db)]
    assert len(cluster_ids) == 3
    merge_ids = cluster_ids[:2]

    new_cluster = merge_clusters(merge_ids, SESSION_ID, turn_number=1, db=db)
    db.commit()

    # the two merged clusters are dissolved as of the operation's turn
    for cid in merge_ids:
        merged = db.query(Cluster).filter(Cluster.id == cid).first()
        assert merged.dissolved_at_turn == 1

    # exactly one fresh active cluster was created at turn 1
    assert new_cluster.created_at_turn == 1
    assert new_cluster.dissolved_at_turn is None
    active = _active_clusters(db)
    assert new_cluster.id in {c.id for c in active}
    assert len(active) == 2  # 3 original - 2 dissolved + 1 new


def test_merge_creates_full_snapshot_at_turn(db):
    cluster_ids = [c.id for c in _active_clusters(db)]
    merge_ids = cluster_ids[:2]
    hard_before = _hard_clusters(db, turn=0)

    new_cluster = merge_clusters(merge_ids, SESSION_ID, turn_number=1, db=db)
    db.commit()

    # every point present at turn 0 still has assignments at turn 1 (complete snapshot)
    assert set(_hard_clusters(db, turn=1)) == set(hard_before)

    # pooled points are now 100% in the new cluster
    pooled = [pid for pid, cid in hard_before.items() if cid in set(merge_ids)]
    assert pooled  # sanity: the merge actually pooled points
    for point_id in pooled:
        rows = (
            db.query(SoftAssignment)
            .filter(
                SoftAssignment.data_point_id == point_id,
                SoftAssignment.turn_number == 1,
            )
            .all()
        )
        assert len(rows) == 1
        assert rows[0].cluster_id == new_cluster.id
        assert rows[0].probability == 1.0

    # no turn-1 assignment references a dissolved cluster
    turn1 = db.query(SoftAssignment).filter(SoftAssignment.turn_number == 1).all()
    assert all(a.cluster_id not in set(merge_ids) for a in turn1)


def test_merge_rejects_single_cluster(db):
    cluster_ids = [c.id for c in _active_clusters(db)]
    with pytest.raises(ValueError):
        merge_clusters(cluster_ids[:1], SESSION_ID, turn_number=1, db=db)


def test_merge_rejects_unknown_cluster(db):
    cluster_ids = [c.id for c in _active_clusters(db)]
    with pytest.raises(ValueError):
        merge_clusters(
            [cluster_ids[0], "does-not-exist"], SESSION_ID, turn_number=1, db=db
        )


def test_merge_rejects_stale_turn_number(db):
    cluster_ids = [c.id for c in _active_clusters(db)]
    # a snapshot already exists at turn 0 → turn_number must be > 0
    with pytest.raises(ValueError):
        merge_clusters(cluster_ids[:2], SESSION_ID, turn_number=0, db=db)


# --- split_cluster ----------------------------------------------------------


def _target_with_two_points(db) -> tuple[str, set[str]]:
    """Pick a cluster that has >= 2 points; return (cluster_id, its point ids)."""
    hard0 = _hard_clusters(db, turn=0)
    for cid in set(hard0.values()):
        members = {pid for pid, c in hard0.items() if c == cid}
        if len(members) >= 2:
            return cid, members
    raise AssertionError("fixture should have a cluster with >= 2 points")


def test_split_creates_two_clusters_with_real_kmeans(db):
    target, subset = _target_with_two_points(db)

    children = split_cluster(target, SESSION_ID, turn_number=1, db=db)
    db.commit()

    assert len(children) == 2
    parent = db.query(Cluster).filter(Cluster.id == target).first()
    assert parent.dissolved_at_turn == 1
    for child in children:
        assert child.created_at_turn == 1
        assert child.dissolved_at_turn is None

    # subset points are reassigned to the two children and genuinely partitioned
    hard1 = _hard_clusters(db, turn=1)
    child_ids = {c.id for c in children}
    for point_id in subset:
        assert hard1[point_id] in child_ids
    assert {hard1[pid] for pid in subset} == child_ids  # real k-means uses both


def test_split_carries_other_points_forward(db):
    target, subset = _target_with_two_points(db)
    hard0 = _hard_clusters(db, turn=0)
    others = {pid for pid in hard0 if pid not in subset}

    split_cluster(target, SESSION_ID, turn_number=1, db=db)
    db.commit()

    hard1 = _hard_clusters(db, turn=1)
    # untouched points keep their cluster, snapshot stays complete
    assert set(hard1) == set(hard0)
    for point_id in others:
        assert hard1[point_id] == hard0[point_id]
    # nothing at turn 1 still points at the dissolved cluster
    turn1 = db.query(SoftAssignment).filter(SoftAssignment.turn_number == 1).all()
    assert all(a.cluster_id != target for a in turn1)


def test_split_rejects_unknown_cluster(db):
    with pytest.raises(ValueError):
        split_cluster("does-not-exist", SESSION_ID, turn_number=1, db=db)


def test_split_rejects_cluster_with_one_point(db):
    # a separate session whose single cluster holds just one point
    db.add(
        ChatSession(
            id="solo", dataset_name="ds", embedding_model="default", status="active"
        )
    )
    db.add(
        DataPoint(id="solo-p", dataset_name="ds", data={"text": "x"}, embedding=[1.0, 1.0])
    )
    db.add(
        Cluster(
            id="solo-c", session_id="solo", name="Solo", description="", created_at_turn=0
        )
    )
    db.add(
        SoftAssignment(
            data_point_id="solo-p", cluster_id="solo-c", turn_number=0, probability=1.0
        )
    )
    db.commit()

    with pytest.raises(ValueError):
        split_cluster("solo-c", "solo", turn_number=1, db=db)


# --- rename_cluster ---------------------------------------------------------


def test_rename_updates_name_and_description(db):
    cluster = _active_clusters(db)[0]
    soft_assignments_before = db.query(SoftAssignment).count()
    clusters_before = db.query(Cluster).count()

    renamed = rename_cluster(
        cluster.id, "Positive reviews", "Mostly 5-star feedback", db
    )
    db.commit()

    assert renamed.id == cluster.id
    assert renamed.name == "Positive reviews"
    assert renamed.description == "Mostly 5-star feedback"
    # nothing else changed: not dissolved, no clusters or assignments added/removed
    assert renamed.dissolved_at_turn is None
    assert db.query(SoftAssignment).count() == soft_assignments_before
    assert db.query(Cluster).count() == clusters_before


def test_rename_rejects_unknown_cluster(db):
    with pytest.raises(ValueError):
        rename_cluster("does-not-exist", "X", "Y", db)
