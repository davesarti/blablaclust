"""Oracle view formatting: deterministic, truncated, fallback-friendly."""

from __future__ import annotations

from src.eval.oracle_view import (
    build_oracle_view,
    filter_invented_ids,
    render_notes,
)


def _state(clusters):
    return {"clusters": clusters}


def test_view_renders_clusters_with_examples():
    state = _state([
        {"id": "c1", "name": "Positive", "description": "praise", "size": 10},
        {"id": "c2", "name": "Negative", "description": "complaints", "size": 6},
    ])
    points = {
        "c1": [
            {"id": "p1", "data": {"text": "Great product!"}},
            {"id": "p2", "data": {"text": "Five stars."}},
        ],
        "c2": [
            {"id": "p3", "data": {"text": "Broke after a week."}},
        ],
    }
    view = build_oracle_view(state, points, n_examples=3)
    rendered = view.render()
    assert "Cluster c1" in rendered
    assert "Cluster c2" in rendered
    assert "Great product!" in rendered
    assert "Broke after a week." in rendered
    assert view.cluster_ids == {"c1", "c2"}
    assert view.point_ids == {"p1", "p2", "p3"}


def test_examples_cap_respected():
    state = _state([{"id": "c1", "name": "X", "description": "y", "size": 4}])
    points = {"c1": [{"id": f"p{i}", "data": {"text": f"line {i}"}} for i in range(10)]}
    view = build_oracle_view(state, points, n_examples=3)
    rendered = view.render()
    assert rendered.count("line ") == 3


def test_long_example_text_is_truncated():
    long_text = "a" * 500
    state = _state([{"id": "c1", "name": "X", "description": "y", "size": 1}])
    points = {"c1": [{"id": "p1", "data": {"text": long_text}}]}
    view = build_oracle_view(state, points, n_examples=3)
    rendered = view.render()
    assert "…" in rendered
    assert len([ln for ln in rendered.splitlines() if "aaaaa" in ln]) == 1
    # Truncated line should be much shorter than the raw 500-char input.
    truncated = next(ln for ln in rendered.splitlines() if "aaaaa" in ln)
    assert len(truncated) < 250


def test_empty_state_render():
    view = build_oracle_view({"clusters": []}, {}, n_examples=3)
    assert view.render() == "(no active clusters)"
    assert view.cluster_ids == set()


def test_view_is_deterministic_by_cluster_id():
    # Insert in random order; output should be cluster-id sorted.
    state = _state([
        {"id": "c2", "name": "Z", "description": "z", "size": 1},
        {"id": "c1", "name": "A", "description": "a", "size": 1},
    ])
    view1 = build_oracle_view(state, {}, n_examples=3)
    view2 = build_oracle_view(state, {}, n_examples=3)
    assert view1.render() == view2.render()
    assert view1.render().index("c1") < view1.render().index("c2")


def test_fallback_text_when_no_text_or_title():
    state = _state([{"id": "c1", "name": "X", "description": "y", "size": 1}])
    points = {"c1": [{"id": "p1", "data": {"unrelated": "value"}}]}
    rendered = build_oracle_view(state, points, n_examples=3).render()
    assert "unrelated" in rendered  # fell back to JSON dump


def test_filter_invented_ids():
    view = build_oracle_view(
        _state([{"id": "c1", "name": "x", "description": "y", "size": 1}]),
        {"c1": [{"id": "p1", "data": {"text": "t"}}]},
    )
    out = filter_invented_ids(
        cluster_ids=["c1", "c_invented"],
        point_ids=["p1", "p_invented"],
        view=view,
    )
    assert out == {"target_cluster_ids": ["c1"], "target_point_ids": ["p1"]}


def test_render_notes_lists_keys():
    out = render_notes({"tone": "cool", "language": "English"})
    assert "- tone: cool" in out
    assert "- language: English" in out


def test_render_notes_empty_default():
    assert "no notes" in render_notes({})
    # None/"" values are filtered out — same as having no notes.
    assert "no notes" in render_notes({"tone": None, "language": ""})
