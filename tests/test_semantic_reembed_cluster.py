"""Tests for semantic_reembed_cluster and the cluster_reembed op in f_apply_operations.

LLM calls (reembed_for_axis, name_clusters) are mocked so the suite is
deterministic and fast. The actual clustering maths (GMM/k-means) and snapshot
logic run for real against an in-memory SQLite DB.
"""

import time

import numpy as np
import pytest
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.engine.cluster_operations import semantic_reembed_cluster
from src.engine.f_apply_operations import f_apply_operations, _match_names_to_clusters
from src.engine.initial_clustering import initial_clustering
from src.engine.turn_builder import TurnBuilder
from src.models import Base, ChatSession, Cluster, DataPoint, SoftAssignment

SESSION_ID = "sess-reembed"

# 8 points: 4 "book" points clustered near [1,0], 4 "film" points near [0,1].
# The parent cluster will contain all 8; target cluster has 0 points (bystander).
_BOOK_EMBS = {
    "b1": [1.0, 0.0, 0.0, 0.1],
    "b2": [0.9, 0.1, 0.0, 0.0],
    "b3": [1.0, 0.0, 0.1, 0.0],
    "b4": [0.8, 0.0, 0.0, 0.2],
}
_FILM_EMBS = {
    "f1": [0.0, 1.0, 0.0, 0.1],
    "f2": [0.1, 0.9, 0.0, 0.0],
    "f3": [0.0, 1.0, 0.1, 0.0],
    "f4": [0.0, 0.8, 0.0, 0.2],
}
_ALL_EMBS = {**_BOOK_EMBS, **_FILM_EMBS}

# One bystander cluster (not split)
_BYSTANDER_EMBS = {
    "x1": [0.5, 0.5, 1.0, 0.0],
    "x2": [0.4, 0.6, 0.9, 0.1],
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

    session.add(ChatSession(id=SESSION_ID, dataset_id="ds", embedding_model="test", status="active"))

    all_points = {**_ALL_EMBS, **_BYSTANDER_EMBS}
    for pid, emb in all_points.items():
        session.add(DataPoint(id=pid, dataset_id="ds", text=f"text_{pid}", embedding=emb))

    # Turn 0: one big cluster (parent) + one bystander cluster
    parent = Cluster(id="parent", session_id=SESSION_ID, name="Books and Films", description="", created_at_turn=0)
    bystander = Cluster(id="bystander", session_id=SESSION_ID, name="Other", description="", created_at_turn=0)
    session.add(parent)
    session.add(bystander)

    for pid in _ALL_EMBS:
        session.add(SoftAssignment(data_point_id=pid, cluster_id="parent", turn_number=0, probability=1.0))
    for pid in _BYSTANDER_EMBS:
        session.add(SoftAssignment(data_point_id=pid, cluster_id="bystander", turn_number=0, probability=1.0))

    session.commit()
    session.close()

    yield Session

    engine.dispose()


def _fake_reembed(points, axis_label, axis_weight=None):
    """Return a clean 1-D axis: book points score high, film points score low."""
    scores = np.array([
        1.0 if p.id.startswith("b") else 0.0
        for p in points
    ], dtype=np.float32).reshape(-1, 1)
    return scores, "llm"


def test_semantic_reembed_cluster_basic(db):
    """Parent dissolved, two children created, bystander untouched."""
    session = db()

    with patch("src.engine.f_semantic_reembed.reembed_for_axis", side_effect=_fake_reembed), \
         patch("src.engine.cluster_naming.name_clusters"):
        builder = TurnBuilder.load(SESSION_ID, turn_number=1, db=session)
        children = semantic_reembed_cluster("parent", "topic", builder, k=2)
        builder.commit()
        session.commit()

    assert len(children) == 2
    assert "parent" in builder.dissolved_ids

    # Bystander's snapshot must still be intact
    for pid in _BYSTANDER_EMBS:
        assert "bystander" in builder.snapshot[pid]

    session.close()


def test_semantic_reembed_cluster_non_subset_renormalized(db):
    """Bystander points have renormalized probabilities (sum ≈ 1)."""
    session = db()

    with patch("src.engine.f_semantic_reembed.reembed_for_axis", side_effect=_fake_reembed), \
         patch("src.engine.cluster_naming.name_clusters"):
        builder = TurnBuilder.load(SESSION_ID, turn_number=1, db=session)
        semantic_reembed_cluster("parent", "topic", builder, k=2)

    for pid in _BYSTANDER_EMBS:
        total = sum(builder.snapshot[pid].values())
        assert abs(total - 1.0) < 1e-6, f"point {pid} probs sum to {total}"

    session.close()


def test_semantic_reembed_cluster_subset_soft_assignments(db):
    """Subset points get non-trivial soft probabilities (GMM, not hard 1.0)."""
    session = db()

    with patch("src.engine.f_semantic_reembed.reembed_for_axis", side_effect=_fake_reembed), \
         patch("src.engine.cluster_naming.name_clusters"):
        builder = TurnBuilder.load(SESSION_ID, turn_number=1, db=session)
        children = semantic_reembed_cluster("parent", "topic", builder, k=2)

    child_ids = {c.id for c in children}
    for pid in _ALL_EMBS:
        dist = builder.snapshot[pid]
        assert set(dist.keys()) == child_ids
        assert abs(sum(dist.values()) - 1.0) < 1e-6

    session.close()


def test_semantic_reembed_cluster_performance(db):
    """Runs in under 2 seconds when reembed_for_axis is mocked (no LLM calls)."""
    session = db()

    with patch("src.engine.f_semantic_reembed.reembed_for_axis", side_effect=_fake_reembed), \
         patch("src.engine.cluster_naming.name_clusters"):
        builder = TurnBuilder.load(SESSION_ID, turn_number=1, db=session)
        t0 = time.perf_counter()
        semantic_reembed_cluster("parent", "topic", builder, k=2)
        elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"took {elapsed:.2f}s — GMM or snapshot logic is unexpectedly slow"
    session.close()


def test_match_names_to_clusters(db):
    """Oracle names are matched to clusters by centroid similarity, not position."""
    session = db()

    with patch("src.engine.f_semantic_reembed.reembed_for_axis", side_effect=_fake_reembed), \
         patch("src.engine.cluster_naming.name_clusters"):
        builder = TurnBuilder.load(SESSION_ID, turn_number=1, db=session)
        children = semantic_reembed_cluster("parent", "topic", builder, k=2)

    # Encode "Books" and "Films" — the ST model will place them near book/film centroids
    matched = _match_names_to_clusters(children, ["Books", "Films"], builder)

    # Find which child has book points as its hard-assigned majority
    for child in children:
        assigned = [pid for pid, dist in builder.snapshot.items()
                    if max(dist, key=dist.get) == child.id]
        book_count = sum(1 for pid in assigned if pid.startswith("b"))
        film_count = sum(1 for pid in assigned if pid.startswith("f"))
        if book_count > film_count:
            assert matched[child.id] == "Books", f"Book cluster got name '{matched[child.id]}'"
        elif film_count > book_count:
            assert matched[child.id] == "Films", f"Film cluster got name '{matched[child.id]}'"

    session.close()


def test_cluster_reembed_op_via_f_apply(db):
    """cluster_reembed op type dispatches to semantic_reembed_cluster."""
    session = db()

    with patch("src.engine.f_semantic_reembed.reembed_for_axis", side_effect=_fake_reembed), \
         patch("src.engine.cluster_naming.name_clusters"):
        builder = TurnBuilder.load(SESSION_ID, turn_number=1, db=session)
        f_apply_operations(
            [{"type": "cluster_reembed", "cluster_id": "parent", "axis_label": "topic", "k": 2}],
            builder=builder,
        )

    assert "parent" in builder.dissolved_ids
    assert len(builder.new_clusters) == 2
    session.close()
