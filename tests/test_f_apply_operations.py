"""Tests for f_apply_operations under the builder-driven contract.

f_apply_operations no longer juggles turn numbers — every op writes through
the same TurnBuilder at a single turn. These tests use a MagicMock builder to
verify the dispatch wiring without spinning up an SQLite DB; the underlying
cluster-op functions are tested for real in test_cluster_operations.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.engine.f_apply_operations import (
    _normalize_cluster_ids,
    _resolve_cluster_id,
    f_apply_operations,
)

MOD = "src.engine.f_apply_operations"


def _builder() -> MagicMock:
    """Mock TurnBuilder. ``active_clusters`` returns []; tests override as needed."""
    b = MagicMock()
    b.session_id = "s1"
    b.turn_number = 3
    b.active_clusters.return_value = []
    b.new_clusters = {}
    b.dissolved_ids = set()
    return b


# ── empty / unknown ────────────────────────────────────────────────────────

def test_empty_operations_is_a_noop():
    builder = _builder()
    f_apply_operations([], builder=builder)
    # Nothing dispatched; builder unchanged.
    builder.add_cluster.assert_not_called()
    builder.dissolve.assert_not_called()


def test_unknown_operation_type_is_skipped():
    ops = [{"type": "reorder", "cluster_id": "c1"}]
    builder = _builder()
    with patch(f"{MOD}.merge_clusters") as mm, \
         patch(f"{MOD}.split_cluster") as ms, \
         patch(f"{MOD}.batch_move_points") as mb, \
         patch(f"{MOD}.rename_cluster") as mr:
        f_apply_operations(ops, builder=builder)
    mm.assert_not_called()
    ms.assert_not_called()
    mb.assert_not_called()
    mr.assert_not_called()


def test_missing_type_key_is_skipped():
    ops = [{"cluster_id": "c1"}]
    with patch(f"{MOD}.merge_clusters") as mm:
        f_apply_operations(ops, builder=_builder())
    mm.assert_not_called()


# ── merge ──────────────────────────────────────────────────────────────────

def test_merge_calls_merge_clusters_with_builder():
    op = {"type": "merge", "cluster_ids": ["c1", "c2"]}
    builder = _builder()
    with patch(f"{MOD}.merge_clusters") as mock_merge:
        f_apply_operations([op], builder=builder)
    mock_merge.assert_called_once_with(
        cluster_ids=["c1", "c2"], builder=builder, axis_hint=None
    )


def test_merge_with_inline_new_name_calls_rename():
    op = {"type": "merge", "cluster_ids": ["c1", "c2"], "new_name": "Combined"}
    builder = _builder()
    new_cluster = MagicMock()
    new_cluster.id = "merged"
    new_cluster.name = "Merge of A + B"
    new_cluster.description = "auto-desc"
    with patch(f"{MOD}.merge_clusters", return_value=new_cluster), \
         patch(f"{MOD}.rename_cluster") as mock_rename:
        f_apply_operations([op], builder=builder)
    mock_rename.assert_called_once_with(
        cluster_id="merged",
        new_name="Combined",
        new_description="auto-desc",
        builder=builder,
    )


# ── split ──────────────────────────────────────────────────────────────────

def test_split_calls_split_cluster_with_builder():
    op = {"type": "split", "cluster_id": "c1"}
    builder = _builder()
    with patch(f"{MOD}.split_cluster", return_value=[]) as mock_split:
        f_apply_operations([op], builder=builder)
    mock_split.assert_called_once_with(
        cluster_id="c1", builder=builder, k=2, axis_hint=None
    )


def test_split_passes_k_to_split_cluster():
    op = {"type": "split", "cluster_id": "c1", "k": 4}
    builder = _builder()
    with patch(f"{MOD}.split_cluster", return_value=[]) as mock_split:
        f_apply_operations([op], builder=builder)
    mock_split.assert_called_once_with(
        cluster_id="c1", builder=builder, k=4, axis_hint=None
    )


def test_split_with_inline_new_names_calls_rename_for_each_child():
    op = {"type": "split", "cluster_id": "c1", "new_names": ["A", "B"]}
    builder = _builder()
    children = [MagicMock(id="x", description=""), MagicMock(id="y", description="")]
    with patch(f"{MOD}.split_cluster", return_value=children), \
         patch(f"{MOD}.rename_cluster") as mock_rename:
        f_apply_operations([op], builder=builder)
    assert mock_rename.call_count == 2


# ── move ───────────────────────────────────────────────────────────────────

def test_move_fans_out_point_ids_into_batch_move_pairs():
    op = {
        "type": "move",
        "point_ids": ["p1", "p2", "p3"],
        "target_cluster_id": "c-target",
    }
    builder = _builder()
    with patch(f"{MOD}.batch_move_points") as mock_batch:
        f_apply_operations([op], builder=builder)
    mock_batch.assert_called_once_with(
        [("p1", "c-target"), ("p2", "c-target"), ("p3", "c-target")],
        builder=builder,
    )


# ── rename ─────────────────────────────────────────────────────────────────

def test_rename_calls_rename_cluster_with_builder():
    op = {
        "type": "rename",
        "cluster_id": "c1",
        "new_name": "Food",
        "new_description": "All food reviews",
    }
    builder = _builder()
    builder.get_cluster.return_value = MagicMock(name="Old", description="old")
    with patch(f"{MOD}.rename_cluster") as mock_rename:
        f_apply_operations([op], builder=builder)
    mock_rename.assert_called_once_with(
        cluster_id="c1",
        new_name="Food",
        new_description="All food reviews",
        builder=builder,
    )


def test_rename_preserves_existing_description_when_oracle_omits_it():
    """When the oracle renames a cluster but provides no new_description, the
    existing description must be preserved — NOT clobbered with an empty
    string."""
    op = {"type": "rename", "cluster_id": "c1", "new_name": "New"}
    builder = _builder()
    existing = MagicMock()
    existing.name = "Old name"
    existing.description = "Auto-generated description"
    builder.get_cluster.return_value = existing
    with patch(f"{MOD}.rename_cluster") as mock_rename:
        f_apply_operations([op], builder=builder)
    _, kwargs = mock_rename.call_args
    assert kwargs["new_name"] == "New"
    assert kwargs["new_description"] == "Auto-generated description"


def test_rename_preserves_existing_name_when_oracle_supplies_only_description():
    """Mirror of the previous test for the (desc given, name missing) case."""
    op = {"type": "rename", "cluster_id": "c1", "new_description": "Refreshed desc"}
    builder = _builder()
    existing = MagicMock()
    existing.name = "Old name"
    existing.description = "old desc"
    builder.get_cluster.return_value = existing
    with patch(f"{MOD}.rename_cluster") as mock_rename, \
         patch(f"{MOD}.auto_name_cluster") as mock_auto:
        f_apply_operations([op], builder=builder)
    mock_auto.assert_not_called()
    _, kwargs = mock_rename.call_args
    assert kwargs["new_name"] == "Old name"
    assert kwargs["new_description"] == "Refreshed desc"


def test_bare_rename_runs_the_naming_prompt():
    """A rename op with neither new_name nor new_description should re-run
    the naming LLM on the cluster instead of falling through to a no-op."""
    op = {"type": "rename", "cluster_id": "c1"}
    builder = _builder()
    with patch(f"{MOD}.rename_cluster") as mock_rename, \
         patch(f"{MOD}.auto_name_cluster") as mock_auto:
        f_apply_operations([op], builder=builder, axis_hint="sentiment")
    mock_rename.assert_not_called()
    mock_auto.assert_called_once_with("c1", builder, axis_hint="sentiment")


# ── multiple operations share one turn ─────────────────────────────────────

def test_multiple_ops_share_same_turn_number():
    """All ops within a conv turn write into the same builder at one turn —
    no per-op turn-number bumping, unlike the old contract."""
    ops = [
        {"type": "merge", "cluster_ids": ["c1", "c2"]},
        {"type": "split", "cluster_id": "c3"},
        {"type": "rename", "cluster_id": "c4", "new_name": "X"},
    ]
    builder = _builder()
    builder.turn_number = 5
    builder.get_cluster.return_value = MagicMock(name="X", description="")
    with patch(f"{MOD}.merge_clusters") as mock_merge, \
         patch(f"{MOD}.split_cluster", return_value=[]) as mock_split, \
         patch(f"{MOD}.rename_cluster"):
        f_apply_operations(ops, builder=builder)
    # All ops received the same builder; no turn-number argument exists anymore.
    assert mock_merge.call_args.kwargs["builder"] is builder
    assert mock_split.call_args.kwargs["builder"] is builder


# ── error propagation ──────────────────────────────────────────────────────

def test_value_error_from_merge_propagates():
    """A bad merge (e.g. already-dissolved cluster) must propagate so the router
    can surface it as HTTP 422."""
    op = {"type": "merge", "cluster_ids": ["c1", "c2"]}
    with patch(f"{MOD}.merge_clusters", side_effect=ValueError("already dissolved")):
        with pytest.raises(ValueError, match="already dissolved"):
            f_apply_operations([op], builder=_builder())


def test_missing_required_field_raises_key_error():
    op = {"type": "merge"}  # cluster_ids missing
    with pytest.raises(KeyError):
        f_apply_operations([op], builder=_builder())


# ── cluster_id transcription repair ────────────────────────────────────────

_ACTIVE = [
    "00322caf-296b-4b03-9080-99a5305bc071",
    "32e99fb7-fc54-4e13-ab72-89a911be06bb",
    "7333b026-a001-4532-86e9-fc77b2d0b108",
]


def test_resolve_exact_id_unchanged():
    assert _resolve_cluster_id(_ACTIVE[0], _ACTIVE, set(_ACTIVE)) == _ACTIVE[0]


def test_resolve_single_char_typo_corrected():
    typo = "00322caf-296b-4b03-9080-99a5303bc071"  # 5 -> 3
    assert _resolve_cluster_id(typo, _ACTIVE, set(_ACTIVE)) == _ACTIVE[0]


def test_resolve_unrelated_id_left_alone():
    bogus = "deadbeef-0000-0000-0000-000000000000"
    assert _resolve_cluster_id(bogus, _ACTIVE, set(_ACTIVE)) == bogus


def test_resolve_empty_and_non_string_pass_through():
    assert _resolve_cluster_id("", _ACTIVE, set(_ACTIVE)) == ""
    assert _resolve_cluster_id(None, _ACTIVE, set(_ACTIVE)) is None


def test_normalize_repairs_merge_ids_in_place():
    typo = "00322caf-296b-4b03-9080-99a5303bc071"  # 5 -> 3
    ops = [{"type": "merge", "cluster_ids": [typo, _ACTIVE[1]]}]
    builder = _builder()
    builder.active_clusters.return_value = [MagicMock(id=i) for i in _ACTIVE]
    _normalize_cluster_ids(ops, builder)
    assert ops[0]["cluster_ids"] == [_ACTIVE[0], _ACTIVE[1]]
