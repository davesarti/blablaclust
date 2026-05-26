"""Tests for f_apply_operations."""

from unittest.mock import MagicMock, call, patch

import pytest

from src.engine.f_apply_operations import f_apply_operations

MOD = "src.engine.f_apply_operations"


def _db():
    return MagicMock()


# ── empty / unknown ────────────────────────────────────────────────────────

def test_empty_operations_returns_turn_number_unchanged():
    result = f_apply_operations([], session_id="s1", turn_number=5, db=_db())
    assert result == 5


def test_unknown_operation_type_is_skipped():
    ops = [{"type": "reorder", "cluster_id": "c1"}]
    result = f_apply_operations(ops, session_id="s1", turn_number=5, db=_db())
    assert result == 5  # no snapshot written


def test_missing_type_key_is_skipped():
    ops = [{"cluster_id": "c1"}]
    result = f_apply_operations(ops, session_id="s1", turn_number=5, db=_db())
    assert result == 5


# ── merge ──────────────────────────────────────────────────────────────────

def test_merge_calls_merge_clusters_with_correct_args():
    op = {"type": "merge", "cluster_ids": ["c1", "c2"], "new_name": "Combined"}
    db = _db()
    with patch(f"{MOD}.merge_clusters") as mock_merge:
        f_apply_operations([op], session_id="s1", turn_number=3, db=db)
    mock_merge.assert_called_once_with(
        cluster_ids=["c1", "c2"], session_id="s1", turn_number=3, db=db
    )


def test_merge_increments_turn_number():
    op = {"type": "merge", "cluster_ids": ["c1", "c2"]}
    with patch(f"{MOD}.merge_clusters"):
        result = f_apply_operations([op], session_id="s1", turn_number=3, db=_db())
    assert result == 4


# ── split ──────────────────────────────────────────────────────────────────

def test_split_calls_split_cluster_with_correct_args():
    op = {"type": "split", "cluster_id": "c1"}
    db = _db()
    with patch(f"{MOD}.split_cluster") as mock_split:
        f_apply_operations([op], session_id="s1", turn_number=7, db=db)
    mock_split.assert_called_once_with(
        cluster_id="c1", session_id="s1", turn_number=7, db=db, k=2
    )


def test_split_passes_k_to_split_cluster():
    op = {"type": "split", "cluster_id": "c1", "k": 4}
    db = _db()
    with patch(f"{MOD}.split_cluster") as mock_split:
        f_apply_operations([op], session_id="s1", turn_number=7, db=db)
    mock_split.assert_called_once_with(
        cluster_id="c1", session_id="s1", turn_number=7, db=db, k=4
    )


def test_split_increments_turn_number():
    op = {"type": "split", "cluster_id": "c1"}
    with patch(f"{MOD}.split_cluster"):
        result = f_apply_operations([op], session_id="s1", turn_number=7, db=_db())
    assert result == 8


# ── rename ─────────────────────────────────────────────────────────────────

def test_rename_calls_rename_cluster_with_correct_args():
    op = {"type": "rename", "cluster_id": "c1", "new_name": "Food",
          "new_description": "All food reviews"}
    db = _db()
    with patch(f"{MOD}.rename_cluster") as mock_rename:
        f_apply_operations([op], session_id="s1", turn_number=2, db=db)
    mock_rename.assert_called_once_with(
        cluster_id="c1", new_name="Food",
        new_description="All food reviews", db=db
    )


def test_rename_does_not_increment_turn_number():
    op = {"type": "rename", "cluster_id": "c1", "new_name": "New"}
    with patch(f"{MOD}.rename_cluster"):
        result = f_apply_operations([op], session_id="s1", turn_number=2, db=_db())
    assert result == 2  # rename writes no snapshot


def test_rename_defaults_new_description_to_empty_string():
    op = {"type": "rename", "cluster_id": "c1", "new_name": "New"}
    db = _db()
    with patch(f"{MOD}.rename_cluster") as mock_rename:
        f_apply_operations([op], session_id="s1", turn_number=2, db=db)
    _, kwargs = mock_rename.call_args
    assert kwargs["new_description"] == ""


# ── multiple operations ────────────────────────────────────────────────────

def test_multiple_ops_turn_number_increments_correctly():
    # merge (turn 5→6) + split (turn 6→7) + rename (no increment)
    ops = [
        {"type": "merge", "cluster_ids": ["c1", "c2"]},
        {"type": "split", "cluster_id": "c3"},
        {"type": "rename", "cluster_id": "c4", "new_name": "X"},
    ]
    with patch(f"{MOD}.merge_clusters") as mock_merge, \
         patch(f"{MOD}.split_cluster") as mock_split, \
         patch(f"{MOD}.rename_cluster"):
        result = f_apply_operations(ops, session_id="s1", turn_number=5, db=_db())
    assert result == 7
    mock_merge.assert_called_once_with(
        cluster_ids=["c1", "c2"], session_id="s1", turn_number=5, db=mock_merge.call_args[1]["db"]
    )
    mock_split.assert_called_once_with(
        cluster_id="c3", session_id="s1", turn_number=6, db=mock_split.call_args[1]["db"], k=2
    )


def test_value_error_from_merge_propagates():
    """A bad merge (e.g. already-dissolved cluster) must propagate so the router
    can surface it as HTTP 422.  Silent skipping is forbidden — it would hide
    real prompt/LLM bugs behind apparent success."""
    op = {"type": "merge", "cluster_ids": ["c1", "c2"]}
    with patch(f"{MOD}.merge_clusters", side_effect=ValueError("already dissolved")):
        with pytest.raises(ValueError, match="already dissolved"):
            f_apply_operations([op], session_id="s1", turn_number=3, db=_db())


def test_missing_required_field_raises_key_error():
    """A malformed merge op (missing cluster_ids) must raise KeyError, not pass silently."""
    op = {"type": "merge"}  # cluster_ids missing
    with pytest.raises(KeyError):
        f_apply_operations([op], session_id="s1", turn_number=3, db=_db())
