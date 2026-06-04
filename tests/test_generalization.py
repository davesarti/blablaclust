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
    assign_gmm_posterior,
    assign_nearest,
    assignment_ood,
    build_centroids,
    calibrate_distance_reference,
    centroids_from_snapshot,
    gmm_params_from_snapshot,
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
# gmm_params_from_snapshot — extends centroids_from_snapshot with the per-
# cluster diagonal covariance and log mixing weights needed for GMM-posterior
# assignment of new arrivals.
# ---------------------------------------------------------------------------


def test_gmm_params_returns_same_centroids_as_centroids_from_snapshot():
    point_embeddings = {
        "p1": [0.0, 0.0], "p2": [2.0, 0.0], "p3": [0.0, 2.0],
        "p4": [10.0, 10.0], "p5": [12.0, 10.0],
    }
    snapshot = {
        "p1": {"A": 0.9, "B": 0.1}, "p2": {"A": 0.8, "B": 0.2},
        "p3": {"A": 0.7, "B": 0.3},
        "p4": {"A": 0.1, "B": 0.9}, "p5": {"A": 0.2, "B": 0.8},
    }
    cids_c, cents_c = centroids_from_snapshot(point_embeddings, snapshot)
    cids_g, cents_g, diag_vars, log_weights = gmm_params_from_snapshot(
        point_embeddings, snapshot
    )
    assert cids_c == cids_g
    np.testing.assert_allclose(cents_c, cents_g)


def test_gmm_params_shapes():
    point_embeddings = {
        "p1": [0.0, 0.0], "p2": [2.0, 0.0],
        "p3": [10.0, 10.0],
    }
    snapshot = {
        "p1": {"A": 1.0}, "p2": {"A": 1.0}, "p3": {"B": 1.0},
    }
    cids, cents, diag_vars, log_weights = gmm_params_from_snapshot(
        point_embeddings, snapshot
    )
    d = 2
    k = 2
    assert diag_vars.shape == (k, d)
    assert log_weights.shape == (k,)
    assert (diag_vars >= 0).all()


def test_gmm_params_log_weights_reflect_cluster_sizes():
    """log_weights = log(n_c / n_total) — mixing weight proportional to cluster size."""
    point_embeddings = {
        "p1": [0.0, 0.0], "p2": [1.0, 0.0], "p3": [0.0, 1.0],  # 3 in A
        "p4": [10.0, 0.0],                                         # 1 in B
    }
    snapshot = {
        "p1": {"A": 1.0}, "p2": {"A": 1.0}, "p3": {"A": 1.0},
        "p4": {"B": 1.0},
    }
    cids, _, _, log_weights = gmm_params_from_snapshot(point_embeddings, snapshot)
    a_idx = cids.index("A")
    b_idx = cids.index("B")
    np.testing.assert_allclose(np.exp(log_weights[a_idx]), 3 / 4)
    np.testing.assert_allclose(np.exp(log_weights[b_idx]), 1 / 4)


def test_gmm_params_diag_var_matches_hard_label_sample_variance():
    """diag_var[c] = sample variance (ddof=0) per dim of base points in cluster c."""
    point_embeddings = {
        "p1": [0.0, 0.0], "p2": [2.0, 0.0], "p3": [0.0, 4.0],
        "p4": [10.0, 10.0],
    }
    snapshot = {
        "p1": {"A": 1.0}, "p2": {"A": 1.0}, "p3": {"A": 1.0},
        "p4": {"B": 1.0},
    }
    cids, _, diag_vars, _ = gmm_params_from_snapshot(point_embeddings, snapshot)
    a_idx = cids.index("A")
    a_emb = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
    np.testing.assert_allclose(diag_vars[a_idx], a_emb.var(axis=0, ddof=0))


def test_gmm_params_raises_when_no_usable_points():
    with pytest.raises(ValueError):
        gmm_params_from_snapshot({}, {"p1": {"A": 1.0}})


# ---------------------------------------------------------------------------
# assign_gmm_posterior — GMM-posterior hard assignment: prefers the cluster
# whose Gaussian density (weighted by mixing weight) is highest at x. Unlike
# assign_nearest, it accounts for per-cluster covariance so a wide cluster
# can "win" over a tight one at the same Euclidean distance.
# ---------------------------------------------------------------------------


def test_assign_gmm_posterior_prefers_wide_cluster_at_boundary():
    """A point equidistant between a tight and a wide cluster goes to the wide one.

    Tight cluster A at (0,0) with σ²=0.01; wide cluster B at (3,0) with σ²=4.0;
    equal mixing weights. A new point at (1.5, 0) is equidistant in Euclidean
    terms but the GMM posterior strongly favours B because A's Gaussian density
    there is negligible.
    """
    centroid_ids = ["A", "B"]
    centroids = np.array([[0.0, 0.0], [3.0, 0.0]])
    diag_vars = np.array([[0.01, 0.01], [4.0, 4.0]])
    log_weights = np.log([0.5, 0.5])

    result = assign_gmm_posterior(
        np.array([[1.5, 0.0]]), centroid_ids, centroids, diag_vars, log_weights
    )
    assert result == ["B"]


def test_assign_gmm_posterior_agrees_with_nearest_for_equal_covariance():
    """Equal diagonal covariance and equal weights → same result as assign_nearest."""
    centroid_ids = ["A", "B", "C"]
    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    diag_vars = np.ones((3, 2))
    log_weights = np.log([1 / 3, 1 / 3, 1 / 3])

    queries = np.array([[0.5, 0.5], [9.0, 0.5], [0.5, 9.0]])
    gmm_result = assign_gmm_posterior(queries, centroid_ids, centroids, diag_vars, log_weights)
    eucl_result = assign_nearest(queries, centroid_ids, centroids)
    assert gmm_result == eucl_result


def test_assign_gmm_posterior_rejects_dim_mismatch():
    with pytest.raises(ValueError):
        assign_gmm_posterior(
            np.array([[1.0, 2.0, 3.0]]),
            ["A"], np.array([[1.0, 2.0]]),
            np.array([[1.0, 1.0]]), np.log([1.0]),
        )


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
    return gmm_params_from_snapshot(emb, snap0)


def test_ingest_writes_new_snapshot_at_next_turn(converged_db):
    cids, centroids, diag_vars, log_weights = _frozen_centroids(converged_db)
    new = [DataPoint(id="n_a", dataset_id="ds", text="n_a", embedding=[0.1, 0.1])]

    new_turn, rows = ingest_points(
        _INGEST_SESSION, new, cids, centroids, diag_vars, log_weights, converged_db
    )
    converged_db.commit()

    assert new_turn == 1
    assert (
        converged_db.query(func.max(SoftAssignment.turn_number)).scalar() == 1
    )


def test_ingest_carries_existing_points_forward_verbatim(converged_db):
    """Pre-existing points are copied to the new turn unchanged (read-only)."""
    snap0 = _snapshot(converged_db, 0)
    cids, centroids, diag_vars, log_weights = _frozen_centroids(converged_db)
    new = [DataPoint(id="n_a", dataset_id="ds", text="n_a", embedding=[0.1, 0.1])]

    ingest_points(_INGEST_SESSION, new, cids, centroids, diag_vars, log_weights, converged_db)
    converged_db.commit()

    snap1 = _snapshot(converged_db, 1)
    for pid, dist0 in snap0.items():
        assert pid in snap1
        assert snap1[pid] == pytest.approx(dist0)  # verbatim, same probabilities


def test_ingest_does_not_mutate_the_converged_snapshot(converged_db):
    """Turn 0 (the converged snapshot) is never touched."""
    before = _snapshot(converged_db, 0)
    cids, centroids, diag_vars, log_weights = _frozen_centroids(converged_db)
    new = [DataPoint(id="n_a", dataset_id="ds", text="n_a", embedding=[0.1, 0.1])]

    ingest_points(_INGEST_SESSION, new, cids, centroids, diag_vars, log_weights, converged_db)
    converged_db.commit()

    assert _snapshot(converged_db, 0) == before


def test_ingest_assigns_new_point_by_gmm_posterior(converged_db):
    """A new point near group A lands in the same cluster as a1/a2 (GMM posterior)."""
    snap0 = _snapshot(converged_db, 0)
    a_cluster = max(snap0["a1"], key=snap0["a1"].get)
    cids, centroids, diag_vars, log_weights = _frozen_centroids(converged_db)

    new = [DataPoint(id="n_a", dataset_id="ds", text="n_a", embedding=[0.05, 0.05])]
    expected = assign_gmm_posterior(
        np.array([[0.05, 0.05]]), cids, centroids, diag_vars, log_weights
    )[0]

    ingest_points(_INGEST_SESSION, new, cids, centroids, diag_vars, log_weights, converged_db)
    converged_db.commit()

    dist = _snapshot(converged_db, 1)["n_a"]
    hard = max(dist, key=dist.get)
    assert hard == expected == a_cluster
    assert sum(dist.values()) == pytest.approx(1.0)


def test_ingest_calibrates_boundary_point_lower():
    """A point between two clusters gets a LOWER max-probability than a point on
    a centroid — the GMM posterior is more diffuse at the boundary.

    The _GROUPS fixture uses 2-pt clusters with near-zero variance; Mahalanobis
    distances collapse all posteriors to ~1.0. This test uses 6-pt clusters with
    meaningful spread so the posteriors vary smoothly between central and boundary.
    """
    _SID = "spread-calib-test"
    _DID = "spread-ds"
    # 6 points per cluster → per-dim variance ≈ 1.4 — enough for non-degenerate posteriors.
    groups_a = [[0, 0], [2, 0], [-2, 0], [0, 2], [0, -2], [1, 1]]
    groups_b = [[10, 0], [12, 0], [8, 0], [10, 2], [10, -2], [11, 1]]

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    db.add(ChatSession(id=_SID, dataset_id=_DID, embedding_model="default", status="converged"))
    pts = []
    for i, emb in enumerate(groups_a + groups_b):
        dp = DataPoint(id=f"sp{i}", dataset_id=_DID, text=f"sp{i}", embedding=emb)
        db.add(dp)
        pts.append(dp)
    clusters, assignments, _ = initial_clustering(pts, k=2, session_id=_SID, turn_number=0)
    for c in clusters:
        db.add(c)
    for a in assignments:
        db.add(a)
    db.commit()

    snap0 = _snapshot(db, 0)
    emb_dict = {f"sp{i}": (groups_a + groups_b)[i] for i in range(12)}
    cids, centroids, diag_vars, log_weights = gmm_params_from_snapshot(emb_dict, snap0)

    central = DataPoint(id="central", dataset_id=_DID, text="c", embedding=[0, 0])
    boundary = DataPoint(id="boundary", dataset_id=_DID, text="b", embedding=[5, 0])
    ingest_points(_SID, [central, boundary], cids, centroids, diag_vars, log_weights, db)
    db.commit()

    snap1 = _snapshot(db, 1)
    assert max(snap1["central"].values()) > max(snap1["boundary"].values())
    db.close()


def test_ingest_freezes_gmm_params_across_batches(converged_db):
    """Two successive ingestions reuse the SAME frozen GMM params and advance the
    turn each time; batch-1 points are carried into batch-2's snapshot."""
    cids, centroids, diag_vars, log_weights = _frozen_centroids(converged_db)

    t1, _ = ingest_points(
        _INGEST_SESSION,
        [DataPoint(id="n1", dataset_id="ds", text="n1", embedding=[0.1, 0.1])],
        cids, centroids, diag_vars, log_weights, converged_db,
    )
    converged_db.commit()
    t2, _ = ingest_points(
        _INGEST_SESSION,
        [DataPoint(id="n2", dataset_id="ds", text="n2", embedding=[10.1, 10.1])],
        cids, centroids, diag_vars, log_weights, converged_db,
    )
    converged_db.commit()

    assert (t1, t2) == (1, 2)
    snap2 = _snapshot(converged_db, 2)
    assert "n1" in snap2 and "n2" in snap2
    assert len(snap2) == len(_GROUPS) + 2


def test_ingest_rejects_points_without_embedding(converged_db):
    cids, centroids, diag_vars, log_weights = _frozen_centroids(converged_db)
    new = [DataPoint(id="n_x", dataset_id="ds", text="x", embedding=None)]
    with pytest.raises(ValueError):
        ingest_points(_INGEST_SESSION, new, cids, centroids, diag_vars, log_weights, converged_db)


def test_ingest_rejects_session_without_clustering(converged_db):
    cids, centroids, diag_vars, log_weights = _frozen_centroids(converged_db)
    new = [DataPoint(id="n_y", dataset_id="ds", text="y", embedding=[0.1, 0.1])]
    with pytest.raises(ValueError):
        ingest_points("no-such-session", new, cids, centroids, diag_vars, log_weights, converged_db)


# ---------------------------------------------------------------------------
# A4 — calibrate_distance_reference: per-cluster Mahalanobis-d² reference
# over base points. Returns dict-of-dicts with keys d2_95, mean, std, diag_var.
# ---------------------------------------------------------------------------

# Matches GMM reg_covar — prevents inf Mahalanobis when a dimension has near-zero variance.
_REG_COVAR = 1e-4


def _mahalanobis_sq(emb: np.ndarray, centroid: np.ndarray, diag_var: np.ndarray) -> np.ndarray:
    """Per-point Mahalanobis squared distance: sum((x - mu)^2 / (var + reg)) per dim."""
    diffs = emb - centroid
    return ((diffs ** 2) / (diag_var + _REG_COVAR)).sum(axis=1)


def test_calibrate_returns_one_entry_per_cluster():
    emb = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [10.0, 0.0], [11.0, 0.0], [10.0, 1.0],
    ])
    labels = ["A", "A", "A", "B", "B", "B"]
    cids, centroids = build_centroids(emb, labels)

    cal = calibrate_distance_reference(emb, labels, cids, centroids)

    assert set(cal.keys()) == {"A", "B"}
    for cid in cids:
        entry = cal[cid]
        assert {"d2_95", "mean", "std", "diag_var"} == set(entry.keys())
        assert entry["d2_95"] >= entry["mean"] >= 0.0
        assert entry["std"] >= 0.0
        assert entry["diag_var"].shape == (emb.shape[1],)
        assert (entry["diag_var"] >= 0.0).all()


def test_calibrate_diag_var_is_sample_variance():
    """diag_var is the per-dim sample variance (ddof=0) of the base points in each cluster."""
    emb = np.array([
        [0.0, 0.0], [2.0, 0.0], [0.0, 4.0],   # cluster A: var_x=8/9·... — just check via numpy
        [10.0, 10.0], [12.0, 10.0],
    ])
    labels = ["A", "A", "A", "B", "B"]
    cids, centroids = build_centroids(emb, labels)
    cal = calibrate_distance_reference(emb, labels, cids, centroids)

    a_points = emb[:3]
    expected_var = a_points.var(axis=0, ddof=0)
    np.testing.assert_allclose(cal["A"]["diag_var"], expected_var)

    b_points = emb[3:]
    np.testing.assert_allclose(cal["B"]["diag_var"], b_points.var(axis=0, ddof=0))


def test_calibrate_matches_mahalanobis_percentile_and_moments():
    """d2_95/mean/std are statistics of the per-point Mahalanobis d² under the
    cluster's own diagonal covariance, not Euclidean d²."""
    emb = np.array([[float(i), float(i) * 2.0] for i in range(100)])
    labels = ["A"] * 100
    cids, centroids = build_centroids(emb, labels)
    cal = calibrate_distance_reference(emb, labels, cids, centroids)

    diag_var = emb.var(axis=0, ddof=0)
    mah_sq = _mahalanobis_sq(emb, centroids[0], diag_var)

    assert cal["A"]["d2_95"] == pytest.approx(float(np.percentile(mah_sq, 95)))
    assert cal["A"]["mean"] == pytest.approx(float(mah_sq.mean()))
    assert cal["A"]["std"] == pytest.approx(float(mah_sq.std(ddof=0)))


def test_calibrate_singleton_cluster_has_zero_spread():
    """A cluster with one base point sits on its own centroid — zero Mahalanobis d².
    diag_var is all zeros (sample var of one point); scoring will regularise with _REG_COVAR."""
    emb = np.array([[5.0, 5.0], [0.0, 0.0], [1.0, 0.0]])
    labels = ["solo", "A", "A"]
    cids, centroids = build_centroids(emb, labels)
    cal = calibrate_distance_reference(emb, labels, cids, centroids)

    entry = cal["solo"]
    assert entry["d2_95"] == pytest.approx(0.0)
    assert entry["mean"] == pytest.approx(0.0)
    assert entry["std"] == pytest.approx(0.0)
    np.testing.assert_allclose(entry["diag_var"], [0.0, 0.0])


def test_calibrate_rejects_unknown_label():
    """A label that isn't in centroid_ids means the calibration set is inconsistent
    with the centroids — refuse rather than silently drop it."""
    emb = np.array([[0.0, 0.0], [1.0, 0.0]])
    cids, centroids = build_centroids(np.array([[0.0, 0.0]]), ["A"])
    with pytest.raises(ValueError):
        calibrate_distance_reference(emb, ["A", "ghost"], cids, centroids)


def test_calibrate_rejects_length_mismatch():
    emb = np.array([[0.0, 0.0], [1.0, 0.0]])
    cids, centroids = build_centroids(emb, ["A", "A"])
    with pytest.raises(ValueError):
        calibrate_distance_reference(emb, ["A"], cids, centroids)


# ---------------------------------------------------------------------------
# A4 — assignment_ood: per-point Mahalanobis z-score and is_ood flag against
# the calibrated reference, plus pooled and per-cluster aggregates.
# ---------------------------------------------------------------------------


def _make_calibration(emb: np.ndarray, labels: list[str]):
    cids, centroids = build_centroids(emb, labels)
    cal = calibrate_distance_reference(emb, labels, cids, centroids)
    return cids, centroids, cal


def test_ood_rate_matches_5pct_when_new_equals_base():
    """If new arrivals have the exact same in-cluster d² distribution as base,
    the OOD rate must match the calibration's complement-of-95th — i.e. 5%."""
    emb = np.array([[float(i), 0.0] for i in range(1000)])
    labels = ["A"] * 1000
    cids, centroids, cal = _make_calibration(emb, labels)

    out = assignment_ood(emb, labels, cids, centroids, cal)

    # By construction d²_95 = 95th percentile of these same d² values, so the
    # rate of "strictly greater than d²_95" is ~5% modulo interpolation/ties.
    assert 0.04 <= out["ood_rate"] <= 0.06


def test_ood_rate_high_for_shifted_distribution():
    """New arrivals drawn far outside base distribution → OOD rate ≈ 1.0."""
    base = np.array([[i * 0.01, 0.0] for i in range(100)])   # tightly packed
    labels = ["A"] * 100
    cids, centroids, cal = _make_calibration(base, labels)

    shifted = np.array([[100.0 + i, 0.0] for i in range(50)])  # all far away
    out = assignment_ood(shifted, ["A"] * 50, cids, centroids, cal)

    assert out["ood_rate"] == pytest.approx(1.0)


def test_ood_per_cluster_breakdown_matches_assignment():
    """Aggregates split by the assigned cluster, not by a re-derivation."""
    base = np.array([
        [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
        [10.0, 10.0], [11.0, 10.0], [9.0, 10.0], [10.0, 11.0], [10.0, 9.0],
    ])
    labels = ["A"] * 5 + ["B"] * 5
    cids, centroids, cal = _make_calibration(base, labels)

    new_emb = np.array([
        [0.0, 0.0], [0.5, 0.0],                              # → A, in-distribution
        [100.0, 100.0], [200.0, 200.0], [300.0, 300.0],      # → B, far OOD
    ])
    assigned = ["A", "A", "B", "B", "B"]
    out = assignment_ood(new_emb, assigned, cids, centroids, cal)

    assert out["per_cluster"]["A"]["n"] == 2
    assert out["per_cluster"]["A"]["ood_rate"] == 0.0
    assert out["per_cluster"]["B"]["n"] == 3
    assert out["per_cluster"]["B"]["ood_rate"] == 1.0


def test_ood_returns_per_point_arrays_aligned_with_input():
    base = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    labels = ["A"] * 4
    cids, centroids, cal = _make_calibration(base, labels)

    new = np.array([[0.5, 0.0], [100.0, 100.0]])
    out = assignment_ood(new, ["A", "A"], cids, centroids, cal)

    assert out["z"].shape == (2,)
    assert out["is_ood"].shape == (2,)
    assert bool(out["is_ood"][0]) is False
    assert bool(out["is_ood"][1]) is True


def test_ood_pooled_mean_z_signals_drift():
    """Pooled mean z is the headline summary alongside ood_rate; it must rise
    when the new batch sits systematically farther from centroids than base."""
    base = np.array([[float(i), 0.0] for i in range(100)])
    labels = ["A"] * 100
    cids, centroids, cal = _make_calibration(base, labels)

    # Same shape, shifted outward in d² → mean z should be clearly positive.
    shifted = np.array([[float(i) + 200.0, 0.0] for i in range(100)])
    out = assignment_ood(shifted, ["A"] * 100, cids, centroids, cal)

    assert out["mean_z"] > 1.0


def test_ood_rejects_assignment_to_unknown_cluster():
    base = np.array([[0.0, 0.0], [1.0, 0.0]])
    labels = ["A", "A"]
    cids, centroids, cal = _make_calibration(base, labels)
    with pytest.raises(ValueError):
        assignment_ood(np.array([[0.1, 0.1]]), ["ghost"], cids, centroids, cal)


def test_ood_rejects_length_mismatch():
    base = np.array([[0.0, 0.0], [1.0, 0.0]])
    labels = ["A", "A"]
    cids, centroids, cal = _make_calibration(base, labels)
    with pytest.raises(ValueError):
        assignment_ood(np.array([[0.1, 0.1], [0.2, 0.2]]), ["A"], cids, centroids, cal)
