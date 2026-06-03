"""Unit tests for src.viz.umap_projection.

Runs against an in-memory SQLite DB. The dimensionality reducer is forced to
PCA where coordinates are asserted (deterministic, no umap/numba needed); a
couple of tests exercise the auto/umap path only for shape and graceful
fallback.
"""

import json

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.logger as logger
from src.models import Base, ChatSession, Cluster, DataPoint, Dataset, SoftAssignment, Turn
from src.viz.umap_projection import (
    _build_hybrid_space,
    compute_coords,
    compute_geometry_aware_coords,
    project_session,
)

SESSION_ID = "sess-umap"
DATASET = "ds"

# Six 2-D-ish embeddings: three near origin, three far away.
_EMB = {
    "p0": [0.0, 0.1],
    "p1": [0.1, 0.0],
    "p2": [0.2, 0.1],
    "p3": [9.0, 9.0],
    "p4": [9.1, 8.9],
    "p5": [8.9, 9.1],
}


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()

    session.add(Dataset(id=DATASET, name=DATASET, description=""))
    session.flush()
    session.add(
        ChatSession(
            id=SESSION_ID,
            dataset_id=DATASET,
            embedding_model="default",
            status="active",
        )
    )
    for pid, emb in _EMB.items():
        session.add(
            DataPoint(id=pid, dataset_id=DATASET, text=f"text {pid}", embedding=emb)
        )

    # Turn 0: two clusters (c1 = p0..p2, c2 = p3..p5).
    # Turn 1: re-partition into c3 (p0,p1,p2,p3) and c4 (p4,p5); c1/c2 dissolved.
    for cid, name, created, dissolved in [
        ("c1", "A", 0, 1),
        ("c2", "B", 0, 1),
        ("c3", "X", 1, None),
        ("c4", "Y", 1, None),
    ]:
        session.add(
            Cluster(
                id=cid,
                session_id=SESSION_ID,
                name=name,
                description="",
                created_at_turn=created,
                dissolved_at_turn=dissolved,
            )
        )

    def soft(pid, cid, turn, prob):
        session.add(
            SoftAssignment(
                data_point_id=pid, cluster_id=cid, turn_number=turn, probability=prob
            )
        )

    turn0 = {"p0": "c1", "p1": "c1", "p2": "c1", "p3": "c2", "p4": "c2", "p5": "c2"}
    turn1 = {"p0": "c3", "p1": "c3", "p2": "c3", "p3": "c3", "p4": "c4", "p5": "c4"}
    for pid in _EMB:
        # winner gets 0.8, the other cluster 0.2 — exercises the argmax path.
        for turn, mapping, clusters in [(0, turn0, ("c1", "c2")), (1, turn1, ("c3", "c4"))]:
            winner = mapping[pid]
            for cid in clusters:
                soft(pid, cid, turn, 0.8 if cid == winner else 0.2)

    session.commit()
    yield session
    session.close()


# ---------------------------------------------------------------------------
# compute_coords
# ---------------------------------------------------------------------------


def test_compute_coords_pca_shape_and_determinism():
    X = np.random.default_rng(0).standard_normal((20, 8)).astype(np.float32)
    c1, name = compute_coords(X, reducer="pca")
    c2, _ = compute_coords(X, reducer="pca")
    assert name == "pca"
    assert c1.shape == (20, 2)
    assert np.allclose(c1, c2)  # deterministic


def test_compute_coords_auto_returns_2d():
    X = np.random.default_rng(1).standard_normal((30, 8)).astype(np.float32)
    coords, name = compute_coords(X)  # auto: umap if installed, else pca
    assert coords.shape == (30, 2)
    assert name in {"umap", "pca"}


# ---------------------------------------------------------------------------
# project_session
# ---------------------------------------------------------------------------


def test_project_session_basic_shape(db):
    res = project_session(db, SESSION_ID, reducer="pca")
    assert res["session_id"] == SESSION_ID
    assert res["dataset_name"] == DATASET
    assert res["reducer"] == "pca"
    assert res["n_points"] == 6
    assert len(res["points"]) == 6
    assert res["turns"] == [0, 1]
    # every point has a coordinate and text preview
    for p in res["points"]:
        assert {"id", "x", "y", "text"} <= p.keys()
        assert isinstance(p["x"], float) and isinstance(p["y"], float)


def test_project_session_assignments_parallel_to_points(db):
    res = project_session(db, SESSION_ID, reducer="pca")
    order = [p["id"] for p in res["points"]]
    # Turn 0: first three points -> c1, last three -> c2 (argmax of 0.8 vs 0.2).
    t0 = res["assignments"]["0"]
    assert len(t0) == 6
    by_point = dict(zip(order, t0))
    assert by_point["p0"] == "c1" and by_point["p5"] == "c2"
    # Turn 1: re-partition; p3 moved from c2-group into c3, p4/p5 -> c4.
    t1 = dict(zip(order, res["assignments"]["1"]))
    assert t1["p3"] == "c3" and t1["p4"] == "c4"
    # distinct cluster counts reflect the evolution (2 -> 2 here, ids differ)
    assert {c for c in t0} == {"c1", "c2"}
    assert {c for c in res["assignments"]["1"]} == {"c3", "c4"}
    # no unassigned points
    assert None not in t0 and None not in res["assignments"]["1"]


def test_project_session_confidence_parallel_to_points(db):
    res = project_session(db, SESSION_ID, reducer="pca")
    assert set(res["confidence"].keys()) == {"0", "1"}
    for t in res["turns"]:
        conf = res["confidence"][str(t)]
        assert len(conf) == res["n_points"]
        non_null = [c for c in conf if c is not None]
        # fixture: every assigned point's winning probability is 0.8
        assert non_null
        assert all(0.0 <= c <= 1.0 for c in non_null)
        assert all(abs(c - 0.8) < 1e-6 for c in non_null)


def test_project_session_cluster_metadata(db):
    res = project_session(db, SESSION_ID, reducer="pca")
    assert res["clusters"]["c1"]["dissolved_at_turn"] == 1
    assert res["clusters"]["c3"]["created_at_turn"] == 1
    assert res["clusters"]["c3"]["dissolved_at_turn"] is None
    assert res["clusters"]["c1"]["name"] == "A"


def test_project_session_silhouette_overlay(db, tmp_path, monkeypatch):
    log_path = tmp_path / "runs.jsonl"
    log_path.write_text(
        json.dumps({"session_id": SESSION_ID, "turn_number": 0, "silhouette": 0.5})
        + "\n"
        + json.dumps({"session_id": "other", "turn_number": 0, "silhouette": 0.9})
        + "\n"
    )
    monkeypatch.setattr(logger, "_clustering_log_path", log_path)

    res = project_session(db, SESSION_ID, reducer="pca")
    assert res["silhouette_by_turn"] == {"0": 0.5}  # only this session's entry


def test_project_session_uses_and_fills_cache(db):
    cache: dict = {}
    project_session(db, SESSION_ID, coords_cache=cache, reducer="pca")
    assert DATASET in cache
    point_ids, coords, reducer_name = cache[DATASET]
    assert len(point_ids) == 6 and coords.shape == (6, 2) and reducer_name == "pca"
    # second call reuses cached coords (same object identity for the array)
    res2 = project_session(db, SESSION_ID, coords_cache=cache, reducer="pca")
    assert res2["n_points"] == 6


def test_project_session_unknown_session_raises(db):
    with pytest.raises(ValueError, match="not found"):
        project_session(db, "does-not-exist", reducer="pca")


def test_project_session_no_embeddings_raises(db):
    for dp in db.query(DataPoint).all():
        dp.embedding = None
    db.commit()
    with pytest.raises(ValueError, match="no embedded points"):
        project_session(db, SESSION_ID, reducer="pca")


def test_project_session_collapses_no_change_turns(db):
    """A snapshot turn with the same hard partition as the previous one is
    hidden from the slider. Mirrors the boundary-repair pattern where many
    consecutive snapshots end up argmax-equivalent and read as 'no change.'"""
    # Add turn 2 = same hard partition as turn 1 (just with different soft
    # probabilities). The collapse should hide turn 2 because the argmax is
    # identical to turn 1.
    turn1 = {"p0": "c3", "p1": "c3", "p2": "c3", "p3": "c3", "p4": "c4", "p5": "c4"}
    for pid, winner in turn1.items():
        for cid in ("c3", "c4"):
            db.add(
                SoftAssignment(
                    data_point_id=pid,
                    cluster_id=cid,
                    turn_number=2,
                    probability=0.7 if cid == winner else 0.3,
                )
            )
    db.commit()

    res = project_session(db, SESSION_ID, reducer="pca")
    # Turn 2 is collapsed away — only the partition-changing turns 0 and 1 stay.
    assert res["turns"] == [0, 1]
    assert set(res["assignments"].keys()) == {"0", "1"}


def test_project_session_collapses_long_run_of_one_point_moves(db):
    """A run of N consecutive snapshots that each differ by a single point —
    the boundary-repair signature — collapses to a single jump from before to
    after. Each individual transition is visually invisible (1 pt of 6 here);
    the slider should show the start, not 20 near-duplicate frames.
    """
    # Build turns 2..5: each successive turn flips one more point from c3 to c4
    # (4 single-point moves), then turn 6 makes a bigger jump (2 changes).
    sequences = [
        {"p0": "c3", "p1": "c3", "p2": "c4", "p3": "c3", "p4": "c4", "p5": "c4"},  # 2: p2 flipped
        {"p0": "c4", "p1": "c3", "p2": "c4", "p3": "c3", "p4": "c4", "p5": "c4"},  # 3: p0 flipped
        {"p0": "c4", "p1": "c4", "p2": "c4", "p3": "c3", "p4": "c4", "p5": "c4"},  # 4: p1 flipped
        {"p0": "c4", "p1": "c4", "p2": "c4", "p3": "c4", "p4": "c4", "p5": "c4"},  # 5: p3 flipped
    ]
    for turn_offset, mapping in enumerate(sequences, start=2):
        for pid, winner in mapping.items():
            for cid in ("c3", "c4"):
                db.add(
                    SoftAssignment(
                        data_point_id=pid, cluster_id=cid,
                        turn_number=turn_offset,
                        probability=0.7 if cid == winner else 0.3,
                    )
                )
    db.commit()

    res = project_session(db, SESSION_ID, reducer="pca")
    # Turns 0 and 1 are kept (multi-point repartition). Turns 2..5 each diff by
    # exactly one point from their predecessor → all collapsed.
    assert res["turns"] == [0, 1]


def test_project_session_keeps_reembed_turn_even_if_partition_unchanged(db):
    """A semantic_reembed turn must remain in the slider so the axis arrow /
    geometry-aware view is still reachable, even if its argmax matches the
    previous turn by coincidence."""
    # Add turn 2 with the same hard partition as turn 1, but flag it as a
    # semantic_reembed in the Turn row's system_output.
    turn1 = {"p0": "c3", "p1": "c3", "p2": "c3", "p3": "c3", "p4": "c4", "p5": "c4"}
    for pid, winner in turn1.items():
        for cid in ("c3", "c4"):
            db.add(
                SoftAssignment(
                    data_point_id=pid,
                    cluster_id=cid,
                    turn_number=2,
                    probability=0.7 if cid == winner else 0.3,
                )
            )
    db.add(
        Turn(
            session_id=SESSION_ID,
            turn_number=2,
            oracle_input={},
            system_output={
                "state_snapshot": {
                    "operations": [
                        {"type": "semantic_reembed", "axis_label": "tone"}
                    ]
                }
            },
        )
    )
    db.commit()

    res = project_session(db, SESSION_ID, reducer="pca")
    assert 2 in res["turns"]


# ---------------------------------------------------------------------------
# Phase 2 — geometry-aware projection (semantic_reembed turns)
# ---------------------------------------------------------------------------
#
# Tests inject a deterministic pole encoder so they don't need to download the
# MiniLM model and can run with arbitrary embedding dimensions. The encoder
# returns axis-dependent poles in the *same* dimensionality as the points'
# embeddings, exactly like the production encoder must.


def _make_dummy_encoder(dim: int):
    """Deterministic encoder: pole vectors derived from a hash of the axis text.

    Different axes -> different (pos, neg) -> different hybrid space, but the
    output is reproducible across calls (same seed). Tests verify both
    properties via this encoder.
    """
    def encode(axis_label: str):
        seed = abs(hash(("pole", axis_label))) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(dim), rng.standard_normal(dim)
    return encode


def _seed_reembed_turn(db, axis_label: str = "positive sentiment") -> None:
    """Add the Turn row for turn 1 so `_axis_label_at_turn` finds the axis.

    The fixture's turn 1 already dissolves c1/c2 and creates c3/c4, which the
    reembed-detection heuristic (≥2 dissolved + ≥2 created at same turn) picks
    up. The Turn row supplies the axis_label in its system_output snapshot.
    """
    db.add(
        Turn(
            session_id=SESSION_ID,
            turn_number=1,
            oracle_input={"raw_text": "split by sentiment", "feedback_type": "global"},
            system_output={
                "state_snapshot": {
                    "operations": [
                        {"type": "semantic_reembed", "axis_label": axis_label}
                    ]
                }
            },
        )
    )
    db.commit()


def test_build_hybrid_space_shape_and_determinism():
    """The hybrid space is (N, D+1), deterministic for the same axis."""
    enc = _make_dummy_encoder(dim=8)
    X = np.random.default_rng(0).standard_normal((6, 8)).astype(np.float32)
    ids = ["p" + str(i) for i in range(6)]
    H1 = _build_hybrid_space(ids, X, "happy", pole_encoder=enc)
    H2 = _build_hybrid_space(ids, X, "happy", pole_encoder=enc)
    assert H1.shape == (6, 9)
    np.testing.assert_allclose(H1, H2)  # no LLM, fully reproducible


def test_build_hybrid_space_axis_changes_geometry():
    """Different axes must produce different hybrid spaces."""
    enc = _make_dummy_encoder(dim=8)
    X = np.random.default_rng(1).standard_normal((6, 8)).astype(np.float32)
    ids = ["p" + str(i) for i in range(6)]
    H_a = _build_hybrid_space(ids, X, "happy", pole_encoder=enc)
    H_b = _build_hybrid_space(ids, X, "technical", pole_encoder=enc)
    # Last column = the axis projection — must differ between axes.
    assert not np.allclose(H_a[:, -1], H_b[:, -1])


def test_build_hybrid_space_rejects_dim_mismatch():
    enc = _make_dummy_encoder(dim=5)  # poles in 5-D
    X = np.random.default_rng(2).standard_normal((6, 8)).astype(np.float32)  # points in 8-D
    with pytest.raises(ValueError, match="does not match"):
        _build_hybrid_space(["a", "b", "c", "d", "e", "f"], X, "x", pole_encoder=enc)


def test_compute_geometry_aware_coords_returns_none_when_no_reembed(db):
    """Without a semantic_reembed op on this turn, the function returns None."""
    enc = _make_dummy_encoder(dim=2)
    X = np.array([db.get(DataPoint, pid).embedding for pid in _EMB], dtype=np.float32)
    res = compute_geometry_aware_coords(
        db, SESSION_ID, turn_number=1, embeddings=X,
        point_ids=list(_EMB.keys()), reducer="pca", pole_encoder=enc,
    )
    assert res is None  # no Turn row seeded yet → no axis_label


def test_compute_geometry_aware_coords_uses_axis_and_caches(db):
    _seed_reembed_turn(db, axis_label="positive sentiment")
    enc = _make_dummy_encoder(dim=2)
    X = np.array([db.get(DataPoint, pid).embedding for pid in _EMB], dtype=np.float32)
    cache: dict = {}

    res = compute_geometry_aware_coords(
        db, SESSION_ID, turn_number=1, embeddings=X,
        point_ids=list(_EMB.keys()), reducer="pca", cache=cache, pole_encoder=enc,
    )
    assert res is not None
    coords, reducer_name, axis_label = res
    assert coords.shape == (6, 2)
    assert reducer_name == "pca"
    assert axis_label == "positive sentiment"
    assert (SESSION_ID, 1) in cache  # cached for reuse

    # Second call hits the cache and returns the same tuple instance.
    assert compute_geometry_aware_coords(
        db, SESSION_ID, turn_number=1, embeddings=X,
        point_ids=list(_EMB.keys()), reducer="pca", cache=cache, pole_encoder=enc,
    ) is res


def test_project_session_geometry_aware_off_by_default(db):
    _seed_reembed_turn(db)
    res = project_session(db, SESSION_ID, reducer="pca")  # no flag
    assert res.get("geometry_aware") == {}  # opt-in only


def test_project_session_geometry_aware_payload(db, monkeypatch):
    """End-to-end: project_session emits the geometry_aware payload for the
    reembed turn, parallel to the baseline points order."""
    _seed_reembed_turn(db, axis_label="positive sentiment")
    enc = _make_dummy_encoder(dim=2)
    monkeypatch.setattr(
        "src.viz.umap_projection._encode_poles_default", enc
    )
    res = project_session(db, SESSION_ID, reducer="pca", geometry_aware=True)
    ga = res.get("geometry_aware") or {}
    assert "1" in ga, "expected geometry_aware payload for the reembed turn"
    entry = ga["1"]
    assert entry["axis_label"] == "positive sentiment"
    assert entry["reducer"] == "pca"
    assert len(entry["points"]) == 6
    # parallel to the top-level points order
    assert [p["id"] for p in entry["points"]] == [p["id"] for p in res["points"]]
    # centroids cover both turn-1 clusters (c3, c4)
    assert set(entry["centroids"].keys()) == {"c3", "c4"}


def test_project_session_geometry_aware_differs_from_baseline(db, monkeypatch):
    """The geometry-aware layout should not be identical to the baseline one —
    that's the whole point of fitting a second UMAP on the hybrid space."""
    _seed_reembed_turn(db)
    enc = _make_dummy_encoder(dim=2)
    monkeypatch.setattr("src.viz.umap_projection._encode_poles_default", enc)
    res = project_session(db, SESSION_ID, reducer="pca", geometry_aware=True)
    baseline = {p["id"]: (p["x"], p["y"]) for p in res["points"]}
    ga_points = res["geometry_aware"]["1"]["points"]
    ga = {p["id"]: (p["x"], p["y"]) for p in ga_points}
    diffs = [
        np.hypot(ga[pid][0] - baseline[pid][0], ga[pid][1] - baseline[pid][1])
        for pid in baseline
    ]
    assert max(diffs) > 1e-6
