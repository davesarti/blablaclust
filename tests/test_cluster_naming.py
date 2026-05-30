"""Unit tests for cluster_naming.name_clusters.

All tests mock call_llm so no real API key is needed. The focus is on:
- a single LLM call is made regardless of the number of clusters
- names and descriptions are applied correctly from the response
- clusters with no assignments keep their placeholder names
- a total LLM failure leaves all placeholders intact
- a partial response (missing cluster IDs) leaves the missing ones intact
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from src.engine.cluster_naming import name_clusters
from src.harness import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cluster(id: str, name: str = "Cluster N", description: str = "") -> SimpleNamespace:
    return SimpleNamespace(id=id, name=name, description=description)


def _dp(id: str, title: str = "", text: str = "") -> SimpleNamespace:
    return SimpleNamespace(id=id, data={"title": title, "text": text})


def _assignment(dp_id: str, cluster_id: str, prob: float) -> SimpleNamespace:
    return SimpleNamespace(data_point_id=dp_id, cluster_id=cluster_id, probability=prob)


def _llm_response(payload: dict) -> LLMResponse:
    return LLMResponse(
        text=json.dumps(payload),
        usage={"input_tokens": 10, "output_tokens": 20,
               "cache_read_tokens": 0, "cache_creation_tokens": 0},
        model="test-model",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSingleLLMCall:
    """call_llm must be invoked exactly once, no matter how many clusters."""

    def test_two_clusters_one_call(self):
        clusters = [_cluster("c1"), _cluster("c2")]
        dps = [_dp("p1", text="great product"), _dp("p2", text="fast delivery")]
        assignments = [
            _assignment("p1", "c1", 0.9),
            _assignment("p2", "c2", 0.9),
        ]
        payload = {
            "c1": {"name": "Quality", "description": "High-quality items."},
            "c2": {"name": "Delivery", "description": "Fast shipping reviews."},
        }

        with patch("src.engine.cluster_naming.call_llm", return_value=_llm_response(payload)) as mock_llm:
            name_clusters(clusters, assignments, dps)

        mock_llm.assert_called_once()

    def test_five_clusters_still_one_call(self):
        n = 5
        clusters = [_cluster(f"c{i}") for i in range(n)]
        dps = [_dp(f"p{i}", text=f"review {i}") for i in range(n)]
        assignments = [_assignment(f"p{i}", f"c{i}", 0.9) for i in range(n)]
        payload = {
            f"c{i}": {"name": f"Name {i}", "description": f"Desc {i}."} for i in range(n)
        }

        with patch("src.engine.cluster_naming.call_llm", return_value=_llm_response(payload)) as mock_llm:
            name_clusters(clusters, assignments, dps)

        mock_llm.assert_called_once()


class TestNamesApplied:
    """Names and descriptions from the LLM response are written to the cluster objects."""

    def test_names_and_descriptions_set(self):
        clusters = [_cluster("c1"), _cluster("c2")]
        dps = [_dp("p1", text="good"), _dp("p2", text="bad")]
        assignments = [_assignment("p1", "c1", 0.9), _assignment("p2", "c2", 0.8)]
        payload = {
            "c1": {"name": "Positive", "description": "Happy customers."},
            "c2": {"name": "Negative", "description": "Unhappy customers."},
        }

        with patch("src.engine.cluster_naming.call_llm", return_value=_llm_response(payload)):
            result = name_clusters(clusters, assignments, dps)

        assert result[0].name == "Positive"
        assert result[0].description == "Happy customers."
        assert result[1].name == "Negative"
        assert result[1].description == "Unhappy customers."

    def test_name_truncated_at_255(self):
        clusters = [_cluster("c1")]
        dps = [_dp("p1", text="something")]
        assignments = [_assignment("p1", "c1", 0.9)]
        long_name = "A" * 300
        payload = {"c1": {"name": long_name, "description": "desc."}}

        with patch("src.engine.cluster_naming.call_llm", return_value=_llm_response(payload)):
            name_clusters(clusters, assignments, dps)

        assert len(clusters[0].name) == 255

    def test_returns_same_list(self):
        clusters = [_cluster("c1")]
        dps = [_dp("p1", text="something")]
        assignments = [_assignment("p1", "c1", 0.9)]
        payload = {"c1": {"name": "X", "description": "Y."}}

        with patch("src.engine.cluster_naming.call_llm", return_value=_llm_response(payload)):
            result = name_clusters(clusters, assignments, dps)

        assert result is clusters


class TestEmptyCluster:
    """Clusters with no assignments (or no usable text) keep their placeholder names."""

    def test_empty_cluster_keeps_placeholder(self):
        clusters = [_cluster("c1"), _cluster("c2", name="Placeholder")]
        dps = [_dp("p1", text="great")]
        # p1 only assigned to c1; c2 has no assignments
        assignments = [_assignment("p1", "c1", 0.9)]
        payload = {"c1": {"name": "Good Stuff", "description": "Positive reviews."}}

        with patch("src.engine.cluster_naming.call_llm", return_value=_llm_response(payload)):
            name_clusters(clusters, assignments, dps)

        assert clusters[0].name == "Good Stuff"
        assert clusters[1].name == "Placeholder"  # unchanged

    def test_all_empty_no_llm_call(self):
        clusters = [_cluster("c1")]
        assignments = []  # no assignments at all
        dps = []

        with patch("src.engine.cluster_naming.call_llm") as mock_llm:
            name_clusters(clusters, assignments, dps)

        mock_llm.assert_not_called()
        assert clusters[0].name == "Cluster N"  # original placeholder


class TestLLMFailure:
    """A failing LLM call must leave all clusters with their original names."""

    def test_exception_keeps_all_placeholders(self):
        clusters = [_cluster("c1"), _cluster("c2")]
        dps = [_dp("p1", text="a"), _dp("p2", text="b")]
        assignments = [_assignment("p1", "c1", 0.9), _assignment("p2", "c2", 0.8)]

        with patch("src.engine.cluster_naming.call_llm", side_effect=RuntimeError("API down")):
            name_clusters(clusters, assignments, dps)

        assert clusters[0].name == "Cluster N"
        assert clusters[1].name == "Cluster N"

    def test_invalid_json_keeps_all_placeholders(self):
        clusters = [_cluster("c1")]
        dps = [_dp("p1", text="something")]
        assignments = [_assignment("p1", "c1", 0.9)]
        bad_response = LLMResponse(
            text="this is not json at all",
            usage={"input_tokens": 0, "output_tokens": 0,
                   "cache_read_tokens": 0, "cache_creation_tokens": 0},
            model="test",
        )

        with patch("src.engine.cluster_naming.call_llm", return_value=bad_response):
            name_clusters(clusters, assignments, dps)

        assert clusters[0].name == "Cluster N"


class TestUnescapedQuotes:
    """LLM responses with raw double quotes in values (inch marks, quoted phrases)
    are invalid JSON but must still be recovered — this is the dominant real-world
    failure for electronics-review naming (15.6", 3.5", etc.)."""

    def _raw(self, text: str) -> LLMResponse:
        return LLMResponse(
            text=text,
            usage={"input_tokens": 0, "output_tokens": 0,
                   "cache_read_tokens": 0, "cache_creation_tokens": 0},
            model="test",
        )

    def test_inch_mark_in_description_is_recovered(self):
        clusters = [_cluster("c1"), _cluster("c2")]
        dps = [_dp("p1", text="laptop"), _dp("p2", text="cable")]
        assignments = [_assignment("p1", "c1", 0.9), _assignment("p2", "c2", 0.9)]
        # Note the raw, unescaped " after 15.6 — exactly what breaks json.loads.
        raw = (
            '{\n'
            '  "c1": {\n'
            '    "name": "Laptop screens",\n'
            '    "description": "Reviews about 15.6" laptop screens with dead pixels"\n'
            '  },\n'
            '  "c2": {\n'
            '    "name": "Cables",\n'
            '    "description": "USB cables that stopped working"\n'
            '  }\n'
            '}'
        )

        with patch("src.engine.cluster_naming.call_llm", return_value=self._raw(raw)):
            name_clusters(clusters, assignments, dps)

        assert clusters[0].name == "Laptop screens"
        assert clusters[0].description == 'Reviews about 15.6" laptop screens with dead pixels'
        assert clusters[1].name == "Cables"

    def test_quoted_phrase_in_value_is_recovered(self):
        clusters = [_cluster("c1")]
        dps = [_dp("p1", text="x")]
        assignments = [_assignment("p1", "c1", 0.9)]
        raw = '{"c1": {"name": "The "best" earbuds", "description": "ok."}}'

        with patch("src.engine.cluster_naming.call_llm", return_value=self._raw(raw)):
            name_clusters(clusters, assignments, dps)

        assert clusters[0].name == 'The "best" earbuds'


class TestPartialResponse:
    """If the LLM omits some cluster IDs, only the present ones get named."""

    def test_missing_cluster_keeps_placeholder(self):
        clusters = [_cluster("c1"), _cluster("c2")]
        dps = [_dp("p1", text="good"), _dp("p2", text="bad")]
        assignments = [_assignment("p1", "c1", 0.9), _assignment("p2", "c2", 0.8)]
        # LLM only returns c1
        payload = {"c1": {"name": "Electronics", "description": "Electronic items."}}

        with patch("src.engine.cluster_naming.call_llm", return_value=_llm_response(payload)):
            name_clusters(clusters, assignments, dps)

        assert clusters[0].name == "Electronics"
        assert clusters[1].name == "Cluster N"  # omitted → placeholder kept

    def test_malformed_entry_keeps_placeholder(self):
        clusters = [_cluster("c1"), _cluster("c2")]
        dps = [_dp("p1", text="good"), _dp("p2", text="bad")]
        assignments = [_assignment("p1", "c1", 0.9), _assignment("p2", "c2", 0.8)]
        # c2 entry is a string, not a dict
        payload = {
            "c1": {"name": "Good", "description": "Positive."},
            "c2": "oops not a dict",
        }

        with patch("src.engine.cluster_naming.call_llm", return_value=_llm_response(payload)):
            name_clusters(clusters, assignments, dps)

        assert clusters[0].name == "Good"
        assert clusters[1].name == "Cluster N"


class TestPromptContent:
    """The prompt sent to the LLM must include each cluster's id and its sample texts."""

    def test_prompt_contains_all_cluster_ids(self):
        clusters = [_cluster("alpha-1"), _cluster("beta-2")]
        dps = [_dp("p1", text="review one"), _dp("p2", text="review two")]
        assignments = [_assignment("p1", "alpha-1", 0.9), _assignment("p2", "beta-2", 0.9)]
        payload = {
            "alpha-1": {"name": "A", "description": "a."},
            "beta-2": {"name": "B", "description": "b."},
        }

        captured_system: list[str] = []

        def capture_call(messages, system, **kwargs):
            captured_system.append(system)
            return _llm_response(payload)

        with patch("src.engine.cluster_naming.call_llm", side_effect=capture_call):
            name_clusters(clusters, assignments, dps)

        assert len(captured_system) == 1
        system = captured_system[0]
        assert "alpha-1" in system
        assert "beta-2" in system
        assert "review one" in system
        assert "review two" in system


class TestUnescapedQuotes:
    """LLM responses with raw double quotes in values (inch marks, quoted phrases)
    are invalid JSON but must still be recovered — this is the dominant real-world
    failure for electronics-review naming (15.6", 3.5", etc.)."""

    def _raw(self, text: str) -> LLMResponse:
        return LLMResponse(
            text=text,
            usage={"input_tokens": 0, "output_tokens": 0,
                   "cache_read_tokens": 0, "cache_creation_tokens": 0},
            model="test",
        )

    def test_inch_mark_in_description_is_recovered(self):
        clusters = [_cluster("c1"), _cluster("c2")]
        dps = [_dp("p1", text="laptop"), _dp("p2", text="cable")]
        assignments = [_assignment("p1", "c1", 0.9), _assignment("p2", "c2", 0.9)]
        raw = (
            '{\n'
            '  "c1": {\n'
            '    "name": "Laptop screens",\n'
            '    "description": "Reviews about 15.6" laptop screens with dead pixels"\n'
            '  },\n'
            '  "c2": {\n'
            '    "name": "Cables",\n'
            '    "description": "USB cables that stopped working"\n'
            '  }\n'
            '}'
        )

        with patch("src.engine.cluster_naming.call_llm", return_value=self._raw(raw)):
            name_clusters(clusters, assignments, dps)

        assert clusters[0].name == "Laptop screens"
        assert clusters[0].description == 'Reviews about 15.6" laptop screens with dead pixels'
        assert clusters[1].name == "Cables"

    def test_quoted_phrase_in_value_is_recovered(self):
        clusters = [_cluster("c1")]
        dps = [_dp("p1", text="x")]
        assignments = [_assignment("p1", "c1", 0.9)]
        raw = '{"c1": {"name": "The "best" earbuds", "description": "ok."}}'

        with patch("src.engine.cluster_naming.call_llm", return_value=self._raw(raw)):
            name_clusters(clusters, assignments, dps)

        assert clusters[0].name == 'The "best" earbuds'
