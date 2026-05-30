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
from src.models import Base, ChatSession, Cluster, DataPoint, SoftAssignment
from src.viz.umap_projection import compute_coords, project_session

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

    session.add(
        ChatSession(
            id=SESSION_ID,
            dataset_name=DATASET,
            embedding_model="default",
            status="active",
        )
    )
    for pid, emb in _EMB.items():
        session.add(
            DataPoint(id=pid, dataset_name=DATASET, data={"text": f"text {pid}"}, embedding=emb)
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
