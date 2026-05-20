"""Tests for f_parse_clustering_intent — uses HARNESS_DRY_RUN=true."""

import json
import os
os.environ["HARNESS_DRY_RUN"] = "true"

from unittest.mock import patch

from src.engine.f_parse_clustering_intent import K_MAX, K_MIN, f_parse_clustering_intent


def _mock_llm(k, axis, reasoning="test"):
    """Return a mock call_llm that produces a valid JSON response."""
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.text = json.dumps({"k": k, "axis": axis, "reasoning": reasoning})
    return lambda *a, **kw: resp


def test_returns_dict_with_required_keys():
    result = f_parse_clustering_intent("group by sentiment")
    assert "k" in result
    assert "axis" in result
    assert "reasoning" in result


def test_k_is_integer():
    result = f_parse_clustering_intent("two groups")
    assert isinstance(result["k"], int)


def test_k_clamped_to_k_min():
    with patch("src.engine.f_parse_clustering_intent.call_llm", _mock_llm(0, "sentiment")):
        result = f_parse_clustering_intent("one cluster", k_min=2)
    assert result["k"] >= 2


def test_k_clamped_to_k_max():
    with patch("src.engine.f_parse_clustering_intent.call_llm", _mock_llm(100, "topic")):
        result = f_parse_clustering_intent("many clusters", k_max=10)
    assert result["k"] <= 10


def test_axis_is_string():
    result = f_parse_clustering_intent("positive vs negative")
    assert isinstance(result["axis"], str)


def test_fallback_on_bad_json():
    from unittest.mock import MagicMock
    bad_resp = MagicMock()
    bad_resp.text = "not json at all"
    with patch("src.engine.f_parse_clustering_intent.call_llm", lambda *a, **kw: bad_resp):
        result = f_parse_clustering_intent("something")
    assert result["k"] == 5
    assert result["axis"] == "semantic similarity"


def test_fallback_on_api_error():
    def raise_error(*a, **kw):
        raise ConnectionError("API down")
    with patch("src.engine.f_parse_clustering_intent.call_llm", raise_error):
        result = f_parse_clustering_intent("something")
    assert result["k"] == 5


def test_real_parse_sentiment():
    with patch("src.engine.f_parse_clustering_intent.call_llm",
               _mock_llm(2, "sentiment", "positive vs negative")):
        result = f_parse_clustering_intent("separate positive from negative reviews")
    assert result["k"] == 2
    assert result["axis"] == "sentiment"


def test_real_parse_topic():
    with patch("src.engine.f_parse_clustering_intent.call_llm",
               _mock_llm(4, "topic", "four topic areas")):
        result = f_parse_clustering_intent("group by topic into 4 clusters")
    assert result["k"] == 4
    assert result["axis"] == "topic"
