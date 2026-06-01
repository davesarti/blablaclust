"""Persona file loading: required fields, typos, permissive notes."""

from __future__ import annotations

import json

import pytest

from src.eval.persona import Persona, load_persona


def _write(tmp_path, payload):
    path = tmp_path / "p.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_persona_round_trip(tmp_path):
    path = _write(tmp_path, {
        "name": "demo",
        "dataset": "amazon_reviews",
        "k_initial": 4,
        "goal": "Try one split, then accept.",
        "notes": {"tone": "curious"},
    })
    p = load_persona(path)
    assert isinstance(p, Persona)
    assert p.name == "demo"
    assert p.k_initial == 4
    assert p.notes.tone == "curious"
    assert p.notes.should_contradict is False
    assert p.model is None


def test_extra_notes_keys_are_kept(tmp_path):
    path = _write(tmp_path, {
        "name": "demo",
        "dataset": "amazon_reviews",
        "k_initial": 3,
        "goal": "go",
        "notes": {"tone": "cool", "custom_axis": "battery life"},
    })
    p = load_persona(path)
    assert getattr(p.notes, "custom_axis") == "battery life"


def test_missing_goal_fails(tmp_path):
    path = _write(tmp_path, {
        "name": "demo",
        "dataset": "amazon_reviews",
        "k_initial": 3,
    })
    with pytest.raises(Exception):
        load_persona(path)


def test_unknown_top_level_key_fails(tmp_path):
    path = _write(tmp_path, {
        "name": "demo",
        "dataset": "amazon_reviews",
        "k_initial": 3,
        "goal": "go",
        "max_turns": 50,  # not a valid top-level key — typo guard
    })
    with pytest.raises(Exception):
        load_persona(path)


def test_k_initial_must_be_at_least_two(tmp_path):
    path = _write(tmp_path, {
        "name": "demo",
        "dataset": "amazon_reviews",
        "k_initial": 1,
        "goal": "go",
    })
    with pytest.raises(Exception):
        load_persona(path)


def test_shipped_personas_parse():
    import glob
    for path in sorted(glob.glob("personas/*.json")):
        p = load_persona(path)
        assert p.name
        assert p.goal
        assert p.k_initial >= 2
