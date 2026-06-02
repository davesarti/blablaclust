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
        cluster_ids=["c1", "c2"], session_id="s1", turn_number=3, db=db, axis_hint=None
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
        cluster_id="c1", session_id="s1", turn_number=7, db=db, k=2, axis_hint=None
    )


def test_split_passes_k_to_split_cluster():
    op = {"type": "split", "cluster_id": "c1", "k": 4}
    db = _db()
    with patch(f"{MOD}.split_cluster") as mock_split:
        f_apply_operations([op], session_id="s1", turn_number=7, db=db)
    mock_split.assert_called_once_with(
        cluster_id="c1", session_id="s1", turn_number=7, db=db, k=4, axis_hint=None
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


def test_rename_preserves_existing_description_when_oracle_omits_it():
    # Issue #43 fix: when the oracle renames a cluster but provides no
    # new_description, the existing (e.g. auto-generated) description must be
    # preserved — NOT clobbered with an empty string. The handler looks the
    # current cluster up and reuses its description.
    op = {"type": "rename", "cluster_id": "c1", "new_name": "New"}
    db = _db()
    existing = MagicMock()
    existing.name = "Old name"
    existing.description = "Auto-generated description"
    db.query.return_value.filter.return_value.first.return_value = existing
    with patch(f"{MOD}.rename_cluster") as mock_rename:
        f_apply_operations([op], session_id="s1", turn_number=2, db=db)
    _, kwargs = mock_rename.call_args
    assert kwargs["new_name"] == "New"                              # oracle's new name wins
    assert kwargs["new_description"] == "Auto-generated description"  # existing preserved


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
        cluster_ids=["c1", "c2"], session_id="s1", turn_number=5,
        db=mock_merge.call_args[1]["db"], axis_hint=None
    )
    mock_split.assert_called_once_with(
        cluster_id="c3", session_id="s1", turn_number=6,
        db=mock_split.call_args[1]["db"], k=2, axis_hint=None
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


# ── cluster_id transcription repair ─────────────────────────────────────────

from src.engine.f_apply_operations import _resolve_cluster_id, _normalize_cluster_ids

_ACTIVE = [
    "00322caf-296b-4b03-9080-99a5305bc071",
    "32e99fb7-fc54-4e13-ab72-89a911be06bb",
    "7333b026-a001-4532-86e9-fc77b2d0b108",
]


def test_resolve_exact_id_unchanged():
    assert _resolve_cluster_id(_ACTIVE[0], _ACTIVE, set(_ACTIVE)) == _ACTIVE[0]


def test_resolve_single_char_typo_corrected():
    # The real bug: LLM mistyped one hex digit of an otherwise-valid UUID.
    typo = "00322caf-296b-4b03-9080-99a5303bc071"  # 5 -> 3
    assert _resolve_cluster_id(typo, _ACTIVE, set(_ACTIVE)) == _ACTIVE[0]


def test_resolve_unrelated_id_left_alone():
    # A totally different id must NOT be silently snapped to a real cluster —
    # it should pass through and fail loudly downstream.
    bogus = "deadbeef-0000-0000-0000-000000000000"
    assert _resolve_cluster_id(bogus, _ACTIVE, set(_ACTIVE)) == bogus


def test_resolve_empty_and_non_string_pass_through():
    assert _resolve_cluster_id("", _ACTIVE, set(_ACTIVE)) == ""
    assert _resolve_cluster_id(None, _ACTIVE, set(_ACTIVE)) is None


def test_normalize_repairs_merge_ids_in_place():
    typo = "00322caf-296b-4b03-9080-99a5303bc071"  # 5 -> 3
    ops = [{"type": "merge", "cluster_ids": [typo, _ACTIVE[1]]}]
    db = MagicMock()
    rows = [MagicMock(id=i) for i in _ACTIVE]
    db.query.return_value.filter.return_value.all.return_value = rows
    _normalize_cluster_ids(ops, session_id="s1", db=db)
    assert ops[0]["cluster_ids"] == [_ACTIVE[0], _ACTIVE[1]]
