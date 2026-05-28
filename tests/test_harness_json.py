"""Tests for the tolerant LLM-JSON loader in src.harness.

loads_llm_json must parse clean JSON unchanged and recover the common LLM
mistake of leaving double quotes unescaped inside string values (inch marks,
quoted phrases), which otherwise raises "Expecting ',' delimiter".
"""

import json

import pytest

from src.harness import loads_llm_json


def test_clean_json_unchanged():
    assert loads_llm_json('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_strips_markdown_and_prose():
    assert loads_llm_json('Here you go:\n```json\n{"a": "b"}\n```') == {"a": "b"}


def test_recovers_inch_mark_in_value():
    raw = '{"name": "Reviews about 15.6" laptop screens"}'
    assert loads_llm_json(raw) == {"name": 'Reviews about 15.6" laptop screens'}


def test_recovers_quoted_phrase_in_value():
    raw = '{"name": "The "best" earbuds", "desc": "ok"}'
    assert loads_llm_json(raw) == {"name": 'The "best" earbuds', "desc": "ok"}


def test_recovers_multiple_inch_marks():
    raw = '{"desc": "Comparing 2.5" and 3.5" drives"}'
    assert loads_llm_json(raw) == {"desc": 'Comparing 2.5" and 3.5" drives'}


def test_preserves_already_escaped_quotes():
    raw = '{"desc": "She said \\"hi\\""}'
    assert loads_llm_json(raw) == {"desc": 'She said "hi"'}


def test_unrecoverable_input_raises():
    # Genuinely truncated JSON cannot be repaired by quote-escaping.
    with pytest.raises(json.JSONDecodeError):
        loads_llm_json('{"name": "unfinished')
