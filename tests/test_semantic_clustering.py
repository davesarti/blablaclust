"""Integration tests for semantic_clustering.

Run entirely against an in-memory SQLite database.  SentenceTransformer
(via _cosine_axis_scores) and name_clusters are patched so no LLM calls or
model downloads occur, but the real k-means and soft-assignment maths run.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.logger as logger
from src.engine.initial_clustering import initial_clustering
from src.engine.semantic_clustering import (
    K_AUTO_MAX,
    K_AUTO_MIN,
    SEMANTIC_BACKEND,
    _auto_select_k,
    semantic_clustering,
)
from src.models import Base, ChatSession, Cluster, DataPoint, SoftAssignment

SESSION_ID = "sess-semantic"

# Six 2-D points: three in group A (near origin), three in group B (near (9,9)).
# k-means with k=2 reliably separates them — large separation, tight clusters.
_POINTS = {
    "a1": [0.0, 0.0],
    "a2": [0.1, 0.1],
    "a3": [0.2, 0.0],
    "b1": [9.0, 9.0],
    "b2": [9.1, 8.9],
    "b3": [8.9, 9.1],
}


# Mock for _cosine_axis_scores: returns a spread-out array (high variance) so
# the cosine path is taken and _llm_axis_scores is never called.
def _fake_cosine(points, pole_pos_text, pole_neg_text):
    return np.linspace(0.0, 1.0, len(points), dtype=np.float64)


PATCH_POLES = patch(
    "src.engine.f_semantic_reembed._generate_axis_poles",
    return_value=("high pole text", "low pole text"),
)
PATCH_COSINE = patch(
    "src.engine.f_semantic_reembed._cosine_axis_scores", side_effect=_fake_cosine
)
PATCH_NAME = patch("src.engine.semantic_clustering.name_clusters")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path, monkeypatch):
    """In-memory DB seeded with six points and a k=2 initial clustering at turn 0."""
    # Redirect clustering log to a tmp file (conftest.py already does this for
    # _clustering_log_path; we reset it again here for clarity in log tests).
    log_path = tmp_path / "clustering_runs.jsonl"
    monkeypatch.setattr(logger, "_clustering_log_path", log_path)

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
            id=point_id,
            dataset_name="ds",
            data={"text": point_id},
            embedding=embedding,
        )
        data_points.append(dp)
        session.add(dp)

    # Initial k=2 clustering at turn 0 (the pre-oracle state).
    clusters, assignments = initial_clustering(
        data_points=data_points, k=2, session_id=SESSION_ID, turn_number=0
    )
    for c in clusters:
        session.add(c)
    for a in assignments:
        session.add(a)
    session.commit()

    yield session
    session.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _active_clusters(db_session, session_id: str = SESSION_ID) -> list[Cluster]:
    return (
        db_session.query(Cluster)
        .filter(
            Cluster.session_id == session_id,
            Cluster.dissolved_at_turn.is_(None),
        )
        .all()
    )


def _soft_assignments_at(db_session, turn: int) -> list[SoftAssignment]:
    return (
        db_session.query(SoftAssignment)
        .filter(SoftAssignment.turn_number == turn)
        .all()
    )


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_dissolves_all_existing_clusters(db):
    """All active clusters from turn 0 must have dissolved_at_turn == 1 after re-embed."""
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()
    active_before = _active_clusters(db)
    assert len(active_before) == 2

    with PATCH_POLES, PATCH_COSINE, PATCH_NAME:
        semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            auto_name=False,
        )
    db.commit()

    for c in active_before:
        db.refresh(c)
        assert c.dissolved_at_turn == 1


def test_creates_k_new_active_clusters(db):
    """k new undissolved clusters must be created at turn 1."""
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()

    with PATCH_POLES, PATCH_COSINE, PATCH_NAME:
        new_clusters, _ = semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            k=3,
            auto_name=False,
        )
    db.commit()

    assert len(new_clusters) == 3
    for c in new_clusters:
        assert c.created_at_turn == 1
        assert c.dissolved_at_turn is None
        assert c.session_id == SESSION_ID


def test_auto_k_called_when_k_is_none(db):
    """When k is not provided, _auto_select_k is called and its result is used."""
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()

    with (
        PATCH_POLES,
        PATCH_COSINE,
        PATCH_NAME,
        patch(
            "src.engine.semantic_clustering._auto_select_k", return_value=2
        ) as mock_auto,
    ):
        new_clusters, _ = semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            auto_name=False,
        )
    db.commit()

    mock_auto.assert_called_once()
    assert len(new_clusters) == 2


def test_explicit_k_bypasses_auto_selection(db):
    """When k is passed explicitly, _auto_select_k is never called."""
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()

    with (
        PATCH_POLES,
        PATCH_COSINE,
        PATCH_NAME,
        patch("src.engine.semantic_clustering._auto_select_k") as mock_auto,
    ):
        new_clusters, _ = semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            k=3,
            auto_name=False,
        )
    db.commit()

    mock_auto.assert_not_called()
    assert len(new_clusters) == 3


def test_auto_k_result_is_in_valid_range(db):
    """With real silhouette selection, auto-k is always in [K_AUTO_MIN, K_AUTO_MAX]."""
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()

    with PATCH_POLES, PATCH_COSINE, PATCH_NAME:
        new_clusters, _ = semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            auto_name=False,
        )
    db.commit()

    assert K_AUTO_MIN <= len(new_clusters) <= K_AUTO_MAX


def test_writes_full_snapshot_at_turn_1_all_points_covered(db):
    """Every data point must appear in the turn-1 soft assignments."""
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()
    all_point_ids = {dp.id for dp in data_points}

    with PATCH_POLES, PATCH_COSINE, PATCH_NAME:
        _, new_assignments = semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            auto_name=False,
        )

    assigned_ids = {a.data_point_id for a in new_assignments}
    assert assigned_ids == all_point_ids


def test_soft_assignments_sum_to_one_per_point(db):
    """Probabilities at turn 1 must sum to 1.0 for every data point."""
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()

    with PATCH_POLES, PATCH_COSINE, PATCH_NAME:
        _, new_assignments = semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            k=2,
            auto_name=False,
        )

    sums: dict[str, float] = {}
    for a in new_assignments:
        sums[a.data_point_id] = sums.get(a.data_point_id, 0.0) + a.probability

    for pid, total in sums.items():
        assert pytest.approx(total, abs=1e-5) == 1.0, f"point {pid} sums to {total}"


def test_all_assignments_have_correct_turn_number(db):
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()

    with PATCH_POLES, PATCH_COSINE, PATCH_NAME:
        _, new_assignments = semantic_clustering(
            data_points=data_points,
            axis_hint="sentiment",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            auto_name=False,
        )

    assert all(a.turn_number == 1 for a in new_assignments)


def test_probabilities_are_in_0_1(db):
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()

    with PATCH_POLES, PATCH_COSINE, PATCH_NAME:
        _, new_assignments = semantic_clustering(
            data_points=data_points,
            axis_hint="quality",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            k=2,
            auto_name=False,
        )

    for a in new_assignments:
        assert 0.0 <= a.probability <= 1.0


def test_well_separated_clusters_get_distinct_hard_assignments(db):
    """The six well-separated points should end up in two distinct clusters."""
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()

    with PATCH_POLES, PATCH_COSINE, PATCH_NAME:
        new_clusters, new_assignments = semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            k=2,
            auto_name=False,
        )

    cluster_ids = {c.id for c in new_clusters}
    distributions: dict[str, dict[str, float]] = {}
    for a in new_assignments:
        distributions.setdefault(a.data_point_id, {})[a.cluster_id] = a.probability

    hard_assignments = {
        pid: max(dist, key=dist.get) for pid, dist in distributions.items()
    }

    # All six points assigned, two non-empty clusters
    assert len(hard_assignments) == 6
    used_clusters = set(hard_assignments.values())
    assert used_clusters == cluster_ids
    assert len(used_clusters) == 2


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------


def test_raises_on_turn_number_zero(db):
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()
    with pytest.raises(ValueError, match="turn_number > 0"):
        semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=0,
            db=db,
        )


def test_raises_on_negative_turn_number(db):
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()
    with pytest.raises(ValueError, match="turn_number > 0"):
        semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=-1,
            db=db,
        )


def test_raises_on_k_less_than_1(db):
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()
    with PATCH_POLES, PATCH_COSINE, PATCH_NAME:
        with pytest.raises(ValueError, match="k must be"):
            semantic_clustering(
                data_points=data_points,
                axis_hint="angry",
                session_id=SESSION_ID,
                turn_number=1,
                db=db,
                k=0,
            )


def test_raises_on_k_exceeds_point_count(db):
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()
    with PATCH_POLES, PATCH_COSINE:
        with pytest.raises(ValueError, match="exceeds number of embedded points"):
            semantic_clustering(
                data_points=data_points,
                axis_hint="angry",
                session_id=SESSION_ID,
                turn_number=1,
                db=db,
                k=999,
            )


def test_raises_on_no_active_clusters(db):
    for c in _active_clusters(db):
        c.dissolved_at_turn = 0
    db.commit()

    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()
    with pytest.raises(ValueError, match="no active clusters"):
        semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
        )


def test_raises_on_no_embedded_points(db):
    for dp in db.query(DataPoint).all():
        dp.embedding = None
    db.commit()

    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()
    with pytest.raises(ValueError, match="no data points have embeddings"):
        semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
        )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_logs_semantic_backend(db, tmp_path, monkeypatch):
    log_path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(logger, "_clustering_log_path", log_path)

    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()

    with PATCH_POLES, PATCH_COSINE, PATCH_NAME:
        semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            k=2,
            auto_name=False,
        )

    entries = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    # Exactly one entry (from semantic_clustering — the initial_clustering in the
    # fixture wrote to a different log_path before the monkeypatch above took effect).
    assert len(entries) == 1
    e = entries[0]
    assert e["session_id"] == SESSION_ID
    assert e["k"] == 2
    assert e["backend"] == SEMANTIC_BACKEND
    assert e["turn_number"] == 1
    assert e["n_points"] == 6
    assert "timestamp" in e


def test_logs_silhouette_for_k_ge_2(db, tmp_path, monkeypatch):
    log_path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(logger, "_clustering_log_path", log_path)

    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()

    with PATCH_POLES, PATCH_COSINE, PATCH_NAME:
        semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            k=2,
            auto_name=False,
        )

    e = json.loads(log_path.read_text().splitlines()[0])
    assert isinstance(e["silhouette"], float)
    assert -1.0 <= e["silhouette"] <= 1.0


def test_logs_none_silhouette_for_k1(db, tmp_path, monkeypatch):
    log_path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(logger, "_clustering_log_path", log_path)

    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()

    with PATCH_POLES, PATCH_COSINE, PATCH_NAME:
        semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            k=1,
            auto_name=False,
        )

    e = json.loads(log_path.read_text().splitlines()[0])
    assert e["silhouette"] is None


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def test_auto_name_true_calls_name_clusters(db):
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()

    with PATCH_POLES, PATCH_COSINE, PATCH_NAME as mock_name:
        semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            auto_name=True,
        )

    mock_name.assert_called_once()


def test_auto_name_false_skips_naming_and_keeps_placeholders(db):
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()

    with PATCH_POLES, PATCH_COSINE, PATCH_NAME as mock_name:
        new_clusters, _ = semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            auto_name=False,
        )

    mock_name.assert_not_called()
    assert all(c.name.startswith("Cluster ") for c in new_clusters)


def test_auto_name_receives_new_clusters_and_assignments(db):
    data_points = db.query(DataPoint).filter(DataPoint.dataset_name == "ds").all()
    captured = {}

    def fake_name(clusters, assignments, points, **kwargs):
        captured["clusters"] = clusters
        captured["assignments"] = assignments

    with (
        PATCH_POLES,
        PATCH_COSINE,
        patch("src.engine.semantic_clustering.name_clusters", side_effect=fake_name),
    ):
        new_clusters, new_assignments = semantic_clustering(
            data_points=data_points,
            axis_hint="angry",
            session_id=SESSION_ID,
            turn_number=1,
            db=db,
            auto_name=True,
        )

    assert captured["clusters"] is new_clusters
    assert captured["assignments"] is new_assignments


# ---------------------------------------------------------------------------
# _auto_select_k unit tests
# ---------------------------------------------------------------------------


class TestAutoSelectK:
    """Unit tests for the silhouette-based automatic k selector."""

    def test_returns_kmin_when_kmax_less_than_kmin(self):
        """Edge case: k_max < k_min returns k_min without running k-means."""
        X = np.zeros((5, 3), dtype=np.float32)
        assert _auto_select_k(X, k_min=3, k_max=1) == 3

    def test_selects_k2_for_clearly_bimodal_data(self):
        """Two tight well-separated clusters → silhouette peaks at k=2."""
        group_a = np.tile([0.0, 0.0, 0.0], (20, 1))
        group_b = np.tile([100.0, 100.0, 100.0], (20, 1))
        X = np.vstack([group_a, group_b]).astype(np.float32)
        assert _auto_select_k(X, k_min=2, k_max=5) == 2

    def test_result_is_always_in_range(self):
        """Auto-k always returns a value in [k_min, k_max]."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        k = _auto_select_k(X, k_min=2, k_max=4)
        assert 2 <= k <= 4
