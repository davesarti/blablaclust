"""Unit tests for the generalization mapping (src/engine/generalization.py).

Pure geometry on synthetic vectors — no embedding model, no DB. Three well-
separated 2-D blobs make the nearest-centroid assignment unambiguous and easy
to reason about.
"""

import numpy as np
import pytest
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.engine.generalization import (
    assign_nearest,
    build_centroids,
    centroids_from_snapshot,
    ingest_points,
)
from src.engine.initial_clustering import initial_clustering
from src.models import Base, ChatSession, DataPoint, SoftAssignment


# ---------------------------------------------------------------------------
# build_centroids
# ---------------------------------------------------------------------------


def test_build_centroids_means_per_cluster():
    embeddings = np.array([
        [0.0, 0.0],
        [2.0, 0.0],   # cluster A → mean (1, 0)
        [10.0, 10.0],
        [12.0, 10.0],  # cluster B → mean (11, 10)
    ])
    labels = ["A", "A", "B", "B"]
    ids, centroids = build_centroids(embeddings, labels)

    assert ids == ["A", "B"]                      # sorted, deterministic
    np.testing.assert_allclose(centroids[0], [1.0, 0.0])
    np.testing.assert_allclose(centroids[1], [11.0, 10.0])


def test_build_centroids_single_point_cluster():
    ids, centroids = build_centroids(np.array([[5.0, 5.0]]), ["solo"])
    assert ids == ["solo"]
    np.testing.assert_allclose(centroids[0], [5.0, 5.0])


def test_build_centroids_order_independent():
    """Shuffling the rows must not change the centroids (ids are sorted)."""
    emb = np.array([[0.0, 0.0], [10.0, 0.0], [2.0, 0.0]])
    a = build_centroids(emb, ["x", "y", "x"])
    b = build_centroids(emb[::-1], ["x", "y", "x"][::-1])
    assert a[0] == b[0]
    np.testing.assert_allclose(a[1], b[1])


def test_build_centroids_rejects_empty():
    with pytest.raises(ValueError):
        build_centroids(np.empty((0, 3)), [])


def test_build_centroids_rejects_length_mismatch():
    with pytest.raises(ValueError):
        build_centroids(np.array([[1.0, 2.0]]), ["a", "b"])


# ---------------------------------------------------------------------------
# assign_nearest
# ---------------------------------------------------------------------------


def test_assign_nearest_clean_separation():
    centroid_ids = ["A", "B"]
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    queries = np.array([
        [1.0, 1.0],     # → A
        [9.0, 9.0],     # → B
        [0.5, 0.0],     # → A
        [11.0, 12.0],   # → B
    ])
    assert assign_nearest(queries, centroid_ids, centroids) == ["A", "B", "A", "B"]


def test_assign_nearest_picks_closest_of_many():
    centroid_ids = ["red", "green", "blue"]
    centroids = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    queries = np.array([[4.6, 0.0], [0.0, 4.9], [0.1, 0.1]])
    assert assign_nearest(queries, centroid_ids, centroids) == ["green", "blue", "red"]


def test_assign_nearest_rejects_dim_mismatch():
    with pytest.raises(ValueError):
        assign_nearest(np.array([[1.0, 2.0, 3.0]]), ["A"], np.array([[1.0, 2.0]]))


def test_assign_nearest_rejects_empty_queries():
    with pytest.raises(ValueError):
        assign_nearest(np.empty((0, 2)), ["A"], np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# centroids_from_snapshot
# ---------------------------------------------------------------------------


def test_centroids_from_snapshot_uses_hard_cluster():
    point_embeddings = {
        "p1": [0.0, 0.0],
        "p2": [2.0, 0.0],
        "p3": [10.0, 10.0],
    }
    snapshot = {
        "p1": {"A": 0.9, "B": 0.1},   # hard → A
        "p2": {"A": 0.6, "B": 0.4},   # hard → A
        "p3": {"A": 0.2, "B": 0.8},   # hard → B
    }
    ids, centroids = centroids_from_snapshot(point_embeddings, snapshot)
    assert ids == ["A", "B"]
    np.testing.assert_allclose(centroids[0], [1.0, 0.0])   # mean of p1, p2
    np.testing.assert_allclose(centroids[1], [10.0, 10.0])  # p3 alone


def test_centroids_from_snapshot_skips_points_without_embedding():
    point_embeddings = {"p1": [0.0, 0.0]}            # p2 has no embedding
    snapshot = {
        "p1": {"A": 1.0},
        "p2": {"A": 1.0},
    }
    ids, centroids = centroids_from_snapshot(point_embeddings, snapshot)
    assert ids == ["A"]
    np.testing.assert_allclose(centroids[0], [0.0, 0.0])


def test_centroids_from_snapshot_raises_when_no_usable_points():
    with pytest.raises(ValueError):
        centroids_from_snapshot({}, {"p1": {"A": 1.0}})


# ---------------------------------------------------------------------------
# end-to-end: build then assign (the actual generalization flow)
# ---------------------------------------------------------------------------


def test_round_trip_training_then_holdout():
    """Train centroids on labelled points, then a held-out point lands in the
    cluster whose members it resembles — the whole point of the feature."""
    train = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],      # topic A near origin
        [20.0, 20.0], [21.0, 20.0], [20.0, 21.0],  # topic B far away
    ])
    labels = ["A", "A", "A", "B", "B", "B"]
    ids, centroids = build_centroids(train, labels)

    held_out = np.array([[0.5, 0.5], [20.5, 20.5]])
    assert assign_nearest(held_out, ids, centroids) == ["A", "B"]


# ---------------------------------------------------------------------------
# ingest_points — online generalization into a live (converged) session
# ---------------------------------------------------------------------------

_INGEST_SESSION = "gen-ingest-test"

# Three well-separated 2-D groups → k-means k=3 finds them cleanly.
_GROUPS = {
    "a1": [0.0, 0.0], "a2": [0.3, 0.2],
    "b1": [10.0, 10.0], "b2": [10.2, 9.8],
    "c1": [0.0, 10.0], "c2": [0.2, 10.1],
}


@pytest.fixture
def converged_db():
    """In-memory DB with a converged k=3 clustering of six points at turn 0."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    db.add(ChatSession(id=_INGEST_SESSION, dataset_id="ds",
                       embedding_model="default", status="converged"))
    points = []
    for pid, emb in _GROUPS.items():
        dp = DataPoint(id=pid, dataset_id="ds", text=pid, embedding=emb)
        db.add(dp)
        points.append(dp)
    clusters, assignments, _ = initial_clustering(
        points, k=3, session_id=_INGEST_SESSION, turn_number=0
    )
    for c in clusters:
        db.add(c)
    for a in assignments:
        db.add(a)
    db.commit()
    yield db
    db.close()


def _snapshot(db, turn):
    rows = db.query(SoftAssignment).filter(SoftAssignment.turn_number == turn).all()
    snap = {}
    for r in rows:
        snap.setdefault(r.data_point_id, {})[r.cluster_id] = r.probability
    return snap


def _frozen_centroids(db):
    snap0 = _snapshot(db, 0)
    emb = {pid: _GROUPS[pid] for pid in snap0}
    return centroids_from_snapshot(emb, snap0)


def test_ingest_writes_new_snapshot_at_next_turn(converged_db):
    cids, centroids = _frozen_centroids(converged_db)
    new = [DataPoint(id="n_a", dataset_id="ds", text="n_a", embedding=[0.1, 0.1])]

    new_turn, rows = ingest_points(_INGEST_SESSION, new, cids, centroids, converged_db)
    converged_db.commit()

    assert new_turn == 1
    assert (
        converged_db.query(func.max(SoftAssignment.turn_number)).scalar() == 1
    )


def test_ingest_carries_existing_points_forward_verbatim(converged_db):
    """Pre-existing points are copied to the new turn unchanged (read-only)."""
    snap0 = _snapshot(converged_db, 0)
    cids, centroids = _frozen_centroids(converged_db)
    new = [DataPoint(id="n_a", dataset_id="ds", text="n_a", embedding=[0.1, 0.1])]

    ingest_points(_INGEST_SESSION, new, cids, centroids, converged_db)
    converged_db.commit()

    snap1 = _snapshot(converged_db, 1)
    for pid, dist0 in snap0.items():
        assert pid in snap1
        assert snap1[pid] == pytest.approx(dist0)  # verbatim, same probabilities


def test_ingest_does_not_mutate_the_converged_snapshot(converged_db):
    """Turn 0 (the converged snapshot) is never touched."""
    before = _snapshot(converged_db, 0)
    cids, centroids = _frozen_centroids(converged_db)
    new = [DataPoint(id="n_a", dataset_id="ds", text="n_a", embedding=[0.1, 0.1])]

    ingest_points(_INGEST_SESSION, new, cids, centroids, converged_db)
    converged_db.commit()

    assert _snapshot(converged_db, 0) == before


def test_ingest_assigns_new_point_to_nearest_centroid(converged_db):
    """A new point near group A lands in the same cluster as a1/a2, matching
    assign_nearest (the persisted soft argmax)."""
    snap0 = _snapshot(converged_db, 0)
    a_cluster = max(snap0["a1"], key=snap0["a1"].get)
    cids, centroids = _frozen_centroids(converged_db)

    new = [DataPoint(id="n_a", dataset_id="ds", text="n_a", embedding=[0.05, 0.05])]
    expected = assign_nearest(np.array([[0.05, 0.05]]), cids, centroids)[0]

    ingest_points(_INGEST_SESSION, new, cids, centroids, converged_db)
    converged_db.commit()

    dist = _snapshot(converged_db, 1)["n_a"]
    hard = max(dist, key=dist.get)
    assert hard == expected == a_cluster
    assert sum(dist.values()) == pytest.approx(1.0)  # full soft distribution


def test_ingest_calibrates_boundary_point_lower(converged_db):
    """A point far from every centroid gets a LOWER max-probability than a point
    sitting on a centroid — this is what lets B2's bottom-2 sample catch bad new
    members."""
    cids, centroids = _frozen_centroids(converged_db)
    central = DataPoint(id="n_c", dataset_id="ds", text="c", embedding=[0.0, 0.0])
    boundary = DataPoint(id="n_b", dataset_id="ds", text="b", embedding=[5.0, 5.0])

    ingest_points(_INGEST_SESSION, [central, boundary], cids, centroids, converged_db)
    converged_db.commit()

    snap1 = _snapshot(converged_db, 1)
    assert max(snap1["n_c"].values()) > max(snap1["n_b"].values())


def test_ingest_freezes_centroids_across_batches(converged_db):
    """Two successive ingestions reuse the SAME frozen centroids and advance the
    turn each time; batch-1 points are carried into batch-2's snapshot."""
    cids, centroids = _frozen_centroids(converged_db)

    t1, _ = ingest_points(
        _INGEST_SESSION,
        [DataPoint(id="n1", dataset_id="ds", text="n1", embedding=[0.1, 0.1])],
        cids, centroids, converged_db,
    )
    converged_db.commit()
    t2, _ = ingest_points(
        _INGEST_SESSION,
        [DataPoint(id="n2", dataset_id="ds", text="n2", embedding=[10.1, 10.1])],
        cids, centroids, converged_db,
    )
    converged_db.commit()

    assert (t1, t2) == (1, 2)
    snap2 = _snapshot(converged_db, 2)
    assert "n1" in snap2 and "n2" in snap2          # batch-1 carried forward
    assert len(snap2) == len(_GROUPS) + 2           # 6 original + 2 ingested


def test_ingest_rejects_points_without_embedding(converged_db):
    cids, centroids = _frozen_centroids(converged_db)
    new = [DataPoint(id="n_x", dataset_id="ds", text="x", embedding=None)]
    with pytest.raises(ValueError):
        ingest_points(_INGEST_SESSION, new, cids, centroids, converged_db)


def test_ingest_rejects_session_without_clustering(converged_db):
    cids, centroids = _frozen_centroids(converged_db)
    new = [DataPoint(id="n_y", dataset_id="ds", text="y", embedding=[0.1, 0.1])]
    with pytest.raises(ValueError):
        ingest_points("no-such-session", new, cids, centroids, converged_db)
