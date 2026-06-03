"""Unit tests for cluster operations: merge / split / batch_move / rename.

These run entirely against an in-memory SQLite database — no LLM calls — so the
clustering maths (k-means via initial_clustering) and the DB persistence are
both exercised for real. Each op is invoked through a TurnBuilder; the builder
is committed once per test so the resulting snapshot lives at the configured
turn number, mirroring how the router orchestrates a real conv turn.
"""

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.engine.cluster_operations import (
    auto_name_cluster,
    batch_move_points,
    merge_clusters,
    rename_cluster,
    split_cluster,
)
from src.engine.initial_clustering import initial_clustering
from src.engine.turn_builder import TurnBuilder
from src.models import Base, ChatSession, Cluster, DataPoint, SoftAssignment

SESSION_ID = "sess-ops"
SESSION_BIG = "sess-big"

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
            dataset_id="ds",
            embedding_model="default",
            status="active",
        )
    )
    data_points = []
    for point_id, embedding in _POINTS.items():
        dp = DataPoint(
            id=point_id, dataset_id="ds", text=point_id, embedding=embedding
        )
        data_points.append(dp)
        session.add(dp)

    clusters, assignments, _ = initial_clustering(
        data_points=data_points, k=3, session_id=SESSION_ID, turn_number=0
    )
    for cluster in clusters:
        session.add(cluster)
    for assignment in assignments:
        session.add(assignment)
    session.commit()

    yield session
    session.close()


@pytest.fixture
def db_big_cluster():
    """DB with one session and one cluster containing all six well-separated points.

    Used for testing split with k > 2: real k-means on these three distinct
    groups should find k=3 sub-clusters cleanly.
    """
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
            id=SESSION_BIG,
            dataset_id="ds",
            embedding_model="default",
            status="active",
        )
    )
    session.add(
        Cluster(
            id="big-c",
            session_id=SESSION_BIG,
            name="Big Cluster",
            description="",
            created_at_turn=0,
        )
    )
    for point_id, embedding in _POINTS.items():
        session.add(
            DataPoint(
                id=f"{point_id}-big",
                dataset_id="ds",
                text=point_id,
                embedding=embedding,
            )
        )
        session.add(
            SoftAssignment(
                data_point_id=f"{point_id}-big",
                cluster_id="big-c",
                turn_number=0,
                probability=1.0,
            )
        )
    session.commit()

    yield session
    session.close()


def _builder(db, turn_number: int, session_id: str = SESSION_ID) -> TurnBuilder:
    """Convenience: load a builder for the given conv turn."""
    return TurnBuilder.load(session_id, turn_number, db)


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

    builder = _builder(db, turn_number=1)
    new_cluster = merge_clusters(merge_ids, builder, auto_name=False)
    builder.commit()
    db.commit()

    for cid in merge_ids:
        merged = db.query(Cluster).filter(Cluster.id == cid).first()
        assert merged.dissolved_at_turn == 1

    assert new_cluster.created_at_turn == 1
    assert new_cluster.dissolved_at_turn is None
    active = _active_clusters(db)
    assert new_cluster.id in {c.id for c in active}
    assert len(active) == 2  # 3 original - 2 dissolved + 1 new


def test_merge_creates_full_snapshot_at_turn(db):
    cluster_ids = [c.id for c in _active_clusters(db)]
    merge_ids = cluster_ids[:2]
    hard_before = _hard_clusters(db, turn=0)

    builder = _builder(db, turn_number=1)
    new_cluster = merge_clusters(merge_ids, builder, auto_name=False)
    builder.commit()
    db.commit()

    assert set(_hard_clusters(db, turn=1)) == set(hard_before)

    pooled = [pid for pid, cid in hard_before.items() if cid in set(merge_ids)]
    assert pooled
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

    turn1 = db.query(SoftAssignment).filter(SoftAssignment.turn_number == 1).all()
    assert all(a.cluster_id not in set(merge_ids) for a in turn1)


def test_merge_rejects_single_cluster(db):
    cluster_ids = [c.id for c in _active_clusters(db)]
    builder = _builder(db, turn_number=1)
    with pytest.raises(ValueError):
        merge_clusters(cluster_ids[:1], builder)


def test_merge_rejects_unknown_cluster(db):
    cluster_ids = [c.id for c in _active_clusters(db)]
    builder = _builder(db, turn_number=1)
    with pytest.raises(ValueError):
        merge_clusters([cluster_ids[0], "does-not-exist"], builder)


def test_merge_rejects_stale_turn_number(db):
    # a snapshot already exists at turn 0 → builder must load at turn > 0
    with pytest.raises(ValueError):
        _builder(db, turn_number=0)


def test_merge_preserves_argmax_on_flat_soft_assignments():
    """Regression: a merge over flat soft assignments must NOT pull every
    un-pooled point into the new cluster.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = Session()

    sid = "sess-flat"
    db.add(ChatSession(id=sid, dataset_id="ds", embedding_model="d", status="active"))
    for i, cid in enumerate(["c1", "c2", "c3", "c4", "c5"]):
        db.add(Cluster(id=cid, session_id=sid, name=f"C{i+1}", description="",
                       created_at_turn=0))
    flat = {"c1": 0.27, "c2": 0.20, "c3": 0.18, "c4": 0.18, "c5": 0.17}
    db.add(DataPoint(id="p1", dataset_id="ds", text="p", embedding=[0.0]))
    for cid, prob in flat.items():
        db.add(SoftAssignment(data_point_id="p1", cluster_id=cid,
                              turn_number=0, probability=prob))
    db.commit()

    builder = TurnBuilder.load(sid, 1, db)
    merge_clusters(["c4", "c5"], builder, auto_name=False)
    builder.commit()
    db.commit()

    hard = _hard_clusters(db, turn=1)
    assert hard["p1"] == "c1", (
        f"p1 should stay in c1; got {hard['p1']} — merge folded mass back, bug regressed."
    )
    db.close()


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

    builder = _builder(db, turn_number=1)
    children = split_cluster(target, builder, auto_name=False)
    builder.commit()
    db.commit()

    assert len(children) == 2
    parent = db.query(Cluster).filter(Cluster.id == target).first()
    assert parent.dissolved_at_turn == 1
    for child in children:
        assert child.created_at_turn == 1
        assert child.dissolved_at_turn is None

    hard1 = _hard_clusters(db, turn=1)
    child_ids = {c.id for c in children}
    for point_id in subset:
        assert hard1[point_id] in child_ids
    assert {hard1[pid] for pid in subset} == child_ids


def test_split_carries_other_points_forward(db):
    target, subset = _target_with_two_points(db)
    hard0 = _hard_clusters(db, turn=0)
    others = {pid for pid in hard0 if pid not in subset}

    builder = _builder(db, turn_number=1)
    split_cluster(target, builder, auto_name=False)
    builder.commit()
    db.commit()

    hard1 = _hard_clusters(db, turn=1)
    assert set(hard1) == set(hard0)
    for point_id in others:
        assert hard1[point_id] == hard0[point_id]
    turn1 = db.query(SoftAssignment).filter(SoftAssignment.turn_number == 1).all()
    assert all(a.cluster_id != target for a in turn1)


def test_split_rejects_unknown_cluster(db):
    builder = _builder(db, turn_number=1)
    with pytest.raises(ValueError):
        split_cluster("does-not-exist", builder)


def test_split_rejects_cluster_with_one_point(db):
    db.add(
        ChatSession(
            id="solo", dataset_id="ds", embedding_model="default", status="active"
        )
    )
    db.add(
        DataPoint(id="solo-p", dataset_id="ds", text="x", embedding=[1.0, 1.0])
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

    builder = TurnBuilder.load("solo", 1, db)
    with pytest.raises(ValueError):
        split_cluster("solo-c", builder)


def test_split_rejects_k_less_than_2(db):
    cluster_id = _active_clusters(db)[0].id
    builder = _builder(db, turn_number=1)
    with pytest.raises(ValueError, match="k must be at least 2"):
        split_cluster(cluster_id, builder, k=1)


def test_split_k3_creates_three_children(db_big_cluster):
    db = db_big_cluster
    builder = TurnBuilder.load(SESSION_BIG, 1, db)
    children = split_cluster("big-c", builder, k=3, auto_name=False)
    builder.commit()
    db.commit()

    assert len(children) == 3
    parent = db.query(Cluster).filter(Cluster.id == "big-c").first()
    assert parent.dissolved_at_turn == 1
    for child in children:
        assert child.created_at_turn == 1
        assert child.dissolved_at_turn is None

    hard1 = _hard_clusters(db, turn=1)
    child_ids = {c.id for c in children}
    assert set(hard1.values()) == child_ids


def test_split_rejects_k_exceeds_point_count(db):
    target, _ = _target_with_two_points(db)
    builder = _builder(db, turn_number=1)
    with pytest.raises(ValueError, match="need at least 3"):
        split_cluster(target, builder, k=3)


# --- naming propagation -----------------------------------------------------


def test_merge_names_cluster_by_default(db):
    cluster_ids = [c.id for c in _active_clusters(db)]

    def fake_name(clusters, assignments, data_points, **kwargs):
        for c in clusters:
            c.name = "Named by LLM"
            c.description = "LLM description"
        return clusters

    builder = _builder(db, turn_number=1)
    with patch(
        "src.engine.cluster_operations.name_clusters", side_effect=fake_name
    ) as mock_name:
        new_cluster = merge_clusters(cluster_ids[:2], builder)

    mock_name.assert_called_once()
    named = mock_name.call_args.args[0]
    assert len(named) == 1 and named[0] is new_cluster
    assert new_cluster.name == "Named by LLM"
    assert new_cluster.description == "LLM description"


def test_merge_skips_naming_when_auto_name_false(db):
    cluster_ids = [c.id for c in _active_clusters(db)]
    builder = _builder(db, turn_number=1)
    with patch("src.engine.cluster_operations.name_clusters") as mock_name:
        new_cluster = merge_clusters(cluster_ids[:2], builder, auto_name=False)
    mock_name.assert_not_called()
    assert new_cluster.name.startswith("Merge of")


def test_split_names_children_by_default(db):
    target, _ = _target_with_two_points(db)

    def fake_name(clusters, assignments, data_points, **kwargs):
        for i, c in enumerate(clusters):
            c.name = f"Named child {i}"
        return clusters

    builder = _builder(db, turn_number=1)
    with patch(
        "src.engine.cluster_operations.name_clusters", side_effect=fake_name
    ) as mock_name:
        children = split_cluster(target, builder)

    mock_name.assert_called_once()
    assert mock_name.call_args.args[0] is children
    assert {c.name for c in children} == {"Named child 0", "Named child 1"}


def test_split_skips_naming_when_auto_name_false(db):
    target, _ = _target_with_two_points(db)
    builder = _builder(db, turn_number=1)
    with patch("src.engine.cluster_operations.name_clusters") as mock_name:
        children = split_cluster(target, builder, auto_name=False)
    mock_name.assert_not_called()
    assert all(" - part " in c.name for c in children)


# --- batch_move_points ------------------------------------------------------


def test_batch_move_keeps_source_with_residual_soft_mass(db):
    """Soft assignments leave residual probability on every cluster, so a
    source that loses all its hard members keeps mass and must NOT dissolve."""
    hard0 = _hard_clusters(db, turn=0)
    source = next(iter(set(hard0.values())))
    members = [pid for pid, cid in hard0.items() if cid == source]
    target = next(cid for cid in set(hard0.values()) if cid != source)

    builder = _builder(db, turn_number=1)
    batch_move_points([(pid, target) for pid in members], builder)
    builder.commit()
    db.commit()

    survivor = db.query(Cluster).filter(Cluster.id == source).first()
    assert survivor.dissolved_at_turn is None
    assert source in {c.id for c in _active_clusters(db)}
    turn1 = db.query(SoftAssignment).filter(SoftAssignment.turn_number == 1).all()
    assert any(a.cluster_id == source for a in turn1)
    hard1 = _hard_clusters(db, turn=1)
    assert all(cid != source for cid in hard1.values())
    assert target in {c.id for c in _active_clusters(db)}
    for pid in members:
        assert hard1[pid] == target


def test_batch_move_applies_all_in_one_snapshot(db):
    """N moves with different targets write exactly ONE snapshot turn."""
    hard0 = _hard_clusters(db, turn=0)
    cluster_ids = list(set(hard0.values()))
    assert len(cluster_ids) >= 2

    moves: list[tuple[str, str]] = []
    seen_points: set[str] = set()
    for cid in cluster_ids:
        for pid, src in hard0.items():
            if src == cid and pid not in seen_points:
                target = next(c for c in cluster_ids if c != cid)
                moves.append((pid, target))
                seen_points.add(pid)
                break

    builder = _builder(db, turn_number=1)
    batch_move_points(moves, builder)
    builder.commit()
    db.commit()

    snapshot_turns = {
        r.turn_number for r in db.query(SoftAssignment).all()
    }
    assert snapshot_turns == {0, 1}

    hard1 = _hard_clusters(db, turn=1)
    for pid, target in moves:
        assert hard1[pid] == target


def test_batch_move_carries_other_points_forward(db):
    hard0 = _hard_clusters(db, turn=0)
    cluster_ids = list(set(hard0.values()))
    point_id, source = next(iter(hard0.items()))
    target = next(cid for cid in cluster_ids if cid != source)
    others = {pid for pid in hard0 if pid != point_id}

    builder = _builder(db, turn_number=1)
    batch_move_points([(point_id, target)], builder)
    builder.commit()
    db.commit()

    hard1 = _hard_clusters(db, turn=1)
    assert set(hard1) == set(hard0)
    for pid in others:
        assert hard1[pid] == hard0[pid]


def test_batch_move_rejects_empty_list(db):
    builder = _builder(db, turn_number=1)
    with pytest.raises(ValueError):
        batch_move_points([], builder)


def test_batch_move_rejects_unknown_target(db):
    point_id = next(iter(_hard_clusters(db, turn=0)))
    builder = _builder(db, turn_number=1)
    with pytest.raises(ValueError):
        batch_move_points([(point_id, "does-not-exist")], builder)


def test_batch_move_rejects_unknown_point(db):
    target = _active_clusters(db)[0].id
    builder = _builder(db, turn_number=1)
    with pytest.raises(ValueError):
        batch_move_points([("does-not-exist", target)], builder)


def test_batch_move_dedupes_duplicate_points(db):
    """A duplicate point_id with conflicting targets: last write wins."""
    hard0 = _hard_clusters(db, turn=0)
    point_id, source = next(iter(hard0.items()))
    others = [cid for cid in set(hard0.values()) if cid != source]
    first_target, second_target = others[0], others[-1]

    builder = _builder(db, turn_number=1)
    batch_move_points(
        [(point_id, first_target), (point_id, second_target)],
        builder,
    )
    builder.commit()
    db.commit()
    hard1 = _hard_clusters(db, turn=1)
    assert hard1[point_id] == second_target


# --- rename_cluster ---------------------------------------------------------


def test_rename_updates_name_and_description(db):
    cluster = _active_clusters(db)[0]
    soft_assignments_before = db.query(SoftAssignment).count()
    clusters_before = db.query(Cluster).count()

    builder = _builder(db, turn_number=1)
    renamed = rename_cluster(
        cluster.id, "Positive reviews", "Mostly 5-star feedback", builder
    )
    # rename does not require a snapshot write; do NOT commit the builder
    # (that would write a duplicate snapshot at turn 1). The name mutation is
    # on the existing DB row and flushes with the next db.commit().
    db.commit()

    assert renamed.id == cluster.id
    assert renamed.name == "Positive reviews"
    assert renamed.description == "Mostly 5-star feedback"
    assert renamed.dissolved_at_turn is None
    assert db.query(SoftAssignment).count() == soft_assignments_before
    assert db.query(Cluster).count() == clusters_before


def test_rename_rejects_unknown_cluster(db):
    builder = _builder(db, turn_number=1)
    with pytest.raises(ValueError):
        rename_cluster("does-not-exist", "X", "Y", builder)


# --- auto_name_cluster ------------------------------------------------------


def test_auto_name_cluster_invokes_naming_llm_with_member_points(db):
    cluster = _active_clusters(db)[0]
    builder = _builder(db, turn_number=1)
    expected_member_ids = {
        pid for pid, dist in builder.snapshot.items()
        if max(dist, key=dist.get) == cluster.id
    }
    assert expected_member_ids, "fixture must hard-assign at least one point"

    with patch("src.engine.cluster_operations.name_clusters") as mock_name:
        auto_name_cluster(cluster.id, builder, axis_hint="tone")

    mock_name.assert_called_once()
    args, kwargs = mock_name.call_args
    clusters_arg, assignments_arg, points_arg = args
    assert clusters_arg == [cluster]
    assert kwargs == {"axis_hint": "tone"}
    assert {p.id for p in points_arg} == expected_member_ids
    assert {a.data_point_id for a in assignments_arg} == expected_member_ids
    assert all(a.cluster_id == cluster.id for a in assignments_arg)
    assert all(a.probability == 1.0 for a in assignments_arg)


def test_auto_name_cluster_rejects_unknown_cluster(db):
    builder = _builder(db, turn_number=1)
    with pytest.raises(ValueError):
        auto_name_cluster("does-not-exist", builder)
