"""LLMOracle: response parsing, invented-ID filtering, dialogue history."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from src.eval import llm_oracle as oracle_mod
from src.eval.llm_oracle import LLMOracle, OracleResponseError
from src.eval.oracle_view import build_oracle_view
from src.eval.persona import Persona, PersonaNotes


@dataclass
class _FakeResponse:
    text: str
    usage: dict
    model: str = "test-model"


@pytest.fixture
def persona():
    return Persona(
        name="t",
        dataset="amazon_reviews",
        k_initial=4,
        goal="Test the loop.",
        notes=PersonaNotes(tone="terse"),
    )


@pytest.fixture
def view():
    return build_oracle_view(
        {"clusters": [
            {"id": "c1", "name": "A", "description": "a", "size": 4},
            {"id": "c2", "name": "B", "description": "b", "size": 4},
        ]},
        {"c1": [{"id": "p1", "data": {"text": "hi"}}]},
    )


def _patch_call_llm(monkeypatch, response_text, usage=None):
    usage = usage or {"input_tokens": 10, "output_tokens": 5,
                      "cache_read_tokens": 0, "cache_creation_tokens": 0}
    def _fake(messages, system, max_tokens=2048):
        return _FakeResponse(text=response_text, usage=usage)
    monkeypatch.setattr(oracle_mod, "call_llm", _fake)


def test_happy_path_produces_valid_input_oracle(monkeypatch, persona, view):
    oracle_json = {
        "raw_text": "Split cluster A by sentiment.",
        "feedback_type": "cluster",
        "target_cluster_ids": ["c1"],
        "target_point_ids": [],
        "satisfied": False,
        "reasoning": "Largest cluster looks mixed.",
    }
    _patch_call_llm(monkeypatch, json.dumps(oracle_json))

    o = LLMOracle(persona=persona, session_id="s1", max_turns=10)
    o.observe_system("welcome")
    turn = o.next_turn(view, system_display="welcome", turn_number=1)

    assert turn.body["session_id"] == "s1"
    assert turn.body["raw_text"].startswith("Split cluster A")
    assert turn.body["feedback_type"] == "cluster"
    assert turn.body["target_cluster_ids"] == ["c1"]
    assert turn.body["target_point_ids"] == []
    assert turn.satisfied is False
    assert turn.body["metadata"]["llm"]["input_tokens"] == 10
    assert turn.body["metadata"]["persona_name"] == "t"


def test_invented_cluster_ids_are_filtered(monkeypatch, persona, view):
    _patch_call_llm(monkeypatch, json.dumps({
        "raw_text": "Touch c_invented please.",
        "feedback_type": "cluster",
        "target_cluster_ids": ["c1", "c_invented"],
        "target_point_ids": ["p1", "p_invented"],
        "satisfied": False,
        "reasoning": "test",
    }))
    o = LLMOracle(persona=persona, session_id="s1", max_turns=5)
    turn = o.next_turn(view, system_display="", turn_number=1)
    assert turn.body["target_cluster_ids"] == ["c1"]
    assert turn.body["target_point_ids"] == ["p1"]


def test_bad_feedback_type_defaults_to_global(monkeypatch, persona, view):
    _patch_call_llm(monkeypatch, json.dumps({
        "raw_text": "ok",
        "feedback_type": "bogus",
        "target_cluster_ids": [],
        "target_point_ids": [],
        "satisfied": False,
        "reasoning": "",
    }))
    o = LLMOracle(persona=persona, session_id="s1", max_turns=5)
    turn = o.next_turn(view, system_display="", turn_number=1)
    assert turn.body["feedback_type"] == "global"


def test_satisfied_flag_propagates(monkeypatch, persona, view):
    _patch_call_llm(monkeypatch, json.dumps({
        "raw_text": "All good, I'm done.",
        "feedback_type": "global",
        "target_cluster_ids": [],
        "target_point_ids": [],
        "satisfied": True,
        "reasoning": "Looks fine.",
    }))
    o = LLMOracle(persona=persona, session_id="s1", max_turns=5)
    turn = o.next_turn(view, system_display="", turn_number=1)
    assert turn.satisfied is True
    assert turn.body["metadata"]["satisfied"] is True


def test_malformed_json_raises(monkeypatch, persona, view):
    _patch_call_llm(monkeypatch, "not json at all")
    o = LLMOracle(persona=persona, session_id="s1", max_turns=5)
    with pytest.raises(OracleResponseError):
        o.next_turn(view, system_display="", turn_number=1)


def test_missing_raw_text_raises(monkeypatch, persona, view):
    _patch_call_llm(monkeypatch, json.dumps({
        "feedback_type": "global",
        "satisfied": False,
    }))
    o = LLMOracle(persona=persona, session_id="s1", max_turns=5)
    with pytest.raises(OracleResponseError):
        o.next_turn(view, system_display="", turn_number=1)


def test_running_totals_accumulate(monkeypatch, persona, view):
    _patch_call_llm(monkeypatch, json.dumps({
        "raw_text": "x",
        "feedback_type": "global",
        "target_cluster_ids": [],
        "target_point_ids": [],
        "satisfied": False,
        "reasoning": "",
    }), usage={"input_tokens": 7, "output_tokens": 3,
               "cache_read_tokens": 0, "cache_creation_tokens": 0})
    o = LLMOracle(persona=persona, session_id="s1", max_turns=5)
    o.next_turn(view, system_display="", turn_number=1)
    o.next_turn(view, system_display="", turn_number=2)
    usage, _ = o.totals
    assert usage["input_tokens"] == 14
    assert usage["output_tokens"] == 6


def test_dialogue_history_is_tracked(monkeypatch, persona, view):
    _patch_call_llm(monkeypatch, json.dumps({
        "raw_text": "first",
        "feedback_type": "global",
        "target_cluster_ids": [],
        "target_point_ids": [],
        "satisfied": False,
        "reasoning": "",
    }))
    o = LLMOracle(persona=persona, session_id="s1", max_turns=5)
    o.observe_system("welcome")
    o.next_turn(view, system_display="welcome", turn_number=1)
    # After one round-trip: system "welcome" + oracle "first" should be in history.
    msgs = o.context.build_messages()
    contents = [m["content"] for m in msgs]
    assert "welcome" in contents
    assert "first" in contents
