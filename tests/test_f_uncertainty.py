"""Tests for f_uncertainty — no real DB, no API calls needed."""

from unittest.mock import MagicMock, patch

import pytest

from src.engine.f_uncertainty import BoundaryPoint, f_uncertainty


def _make_db(cluster_ids, assignments, data_points=None):
    """Build a mock SQLAlchemy Session that returns the given data."""
    db = MagicMock()

    # db.query(DbCluster.id).filter(...).all() → [(cid,), ...]
    cluster_query = MagicMock()
    cluster_query.filter.return_value.all.return_value = [(cid,) for cid in cluster_ids]

    # db.query(func.max(...)).filter(...).scalar() → latest_turn
    latest_turn = max((a.turn_number for a in assignments), default=None)
    max_query = MagicMock()
    max_query.filter.return_value.scalar.return_value = latest_turn

    # db.query(SoftAssignment).filter(...).all() → assignments
    assign_query = MagicMock()
    assign_query.filter.return_value.all.return_value = assignments

    # db.query(DataPoint).filter(...).all() → data_points
    dp_query = MagicMock()
    dp_query.filter.return_value.all.return_value = data_points or []

    call_count = [0]

    def side_effect(target):
        call_count[0] += 1
        n = call_count[0]
        if n == 1:
            return cluster_query   # DbCluster.id query
        if n == 2:
            return max_query       # func.max(turn_number) query
        if n == 3:
            return assign_query    # SoftAssignment query
        return dp_query            # DataPoint query

    db.query.side_effect = side_effect
    return db


def _soft(point_id, cluster_id, probability, turn_number=1):
    a = MagicMock()
    a.data_point_id = point_id
    a.cluster_id = cluster_id
    a.probability = probability
    a.turn_number = turn_number
    return a


def _dp(id_, text=""):
    dp = MagicMock()
    dp.id = id_
    dp.text = text
    return dp


# ── tests ──────────────────────────────────────────────────────────────────

def test_returns_empty_when_no_clusters():
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = []
    result = f_uncertainty("session-1", db)
    assert result == []


def test_returns_empty_when_no_assignments():
    db = _make_db(["c1", "c2"], [])
    # latest_turn is None when no assignments
    result = f_uncertainty("session-1", db)
    assert result == []


def test_uncertainty_score_formula():
    # point A: max prob = 0.9 → uncertainty = 0.1
    # point B: max prob = 0.5 → uncertainty = 0.5
    assignments = [
        _soft("A", "c1", 0.9),
        _soft("A", "c2", 0.1),
        _soft("B", "c1", 0.5),
        _soft("B", "c2", 0.5),
    ]
    db = _make_db(["c1", "c2"], assignments)
    results = f_uncertainty("session-1", db)
    assert len(results) == 2
    b = next(r for r in results if r.point_id == "B")
    a = next(r for r in results if r.point_id == "A")
    assert b.uncertainty_score == pytest.approx(0.5, abs=1e-4)
    assert a.uncertainty_score == pytest.approx(0.1, abs=1e-4)


def test_sorted_descending_by_uncertainty():
    assignments = [
        _soft("A", "c1", 0.9), _soft("A", "c2", 0.1),  # uncertainty 0.1
        _soft("B", "c1", 0.6), _soft("B", "c2", 0.4),  # uncertainty 0.4
        _soft("C", "c1", 0.5), _soft("C", "c2", 0.5),  # uncertainty 0.5
    ]
    db = _make_db(["c1", "c2"], assignments)
    results = f_uncertainty("session-1", db)
    scores = [r.uncertainty_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_n_limits_results():
    assignments = [
        _soft("A", "c1", 0.5), _soft("A", "c2", 0.5),
        _soft("B", "c1", 0.6), _soft("B", "c2", 0.4),
        _soft("C", "c1", 0.7), _soft("C", "c2", 0.3),
    ]
    db = _make_db(["c1", "c2"], assignments)
    results = f_uncertainty("session-1", db, top_n=2)
    assert len(results) == 2


def test_cluster_scores_dict_populated():
    assignments = [
        _soft("A", "c1", 0.7),
        _soft("A", "c2", 0.3),
    ]
    db = _make_db(["c1", "c2"], assignments)
    results = f_uncertainty("session-1", db)
    assert len(results) == 1
    assert results[0].cluster_scores == {"c1": 0.7, "c2": 0.3}


def test_text_preview_populated():
    assignments = [_soft("A", "c1", 0.5), _soft("A", "c2", 0.5)]
    dp_a = _dp("A", text="Hello world")
    db = _make_db(["c1", "c2"], assignments, data_points=[dp_a])
    results = f_uncertainty("session-1", db)
    assert results[0].text_preview == "Hello world"


def test_text_preview_truncated_at_200_chars():
    long_text = "x" * 300
    assignments = [_soft("A", "c1", 0.5), _soft("A", "c2", 0.5)]
    dp_a = _dp("A", text=long_text)
    db = _make_db(["c1", "c2"], assignments, data_points=[dp_a])
    results = f_uncertainty("session-1", db)
    assert len(results[0].text_preview) <= 200


def test_returns_boundary_point_instances():
    assignments = [_soft("A", "c1", 0.6), _soft("A", "c2", 0.4)]
    db = _make_db(["c1", "c2"], assignments)
    results = f_uncertainty("session-1", db)
    assert all(isinstance(r, BoundaryPoint) for r in results)
