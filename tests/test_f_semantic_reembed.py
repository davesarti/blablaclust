"""Tests for f_semantic_reembed.

SentenceTransformer and LLM calls are mocked throughout so no models are
downloaded and no API credits are consumed.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models import DataPoint

MOD = "src.engine.f_semantic_reembed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_points(n: int, dim: int = 4) -> list[DataPoint]:
    """n DataPoints with deterministic L2-normalisable embeddings."""
    rng = np.random.default_rng(0)
    points = []
    for i in range(n):
        dp = DataPoint()
        dp.id = f"p{i}"
        dp.embedding = rng.standard_normal(dim).astype(np.float32).tolist()
        dp.data = {"text": f"text {i}"}
        dp.dataset_name = "ds"
        points.append(dp)
    return points


# ---------------------------------------------------------------------------
# _cosine_axis_scores
# ---------------------------------------------------------------------------


class TestCosineAxisScores:
    def test_returns_correct_shape(self):
        points = _make_points(6)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones(4, dtype=np.float64)

        with patch(f"{MOD}.SentenceTransformer", return_value=mock_model):
            from src.engine.f_semantic_reembed import _cosine_axis_scores
            scores = _cosine_axis_scores(points, "very angry text", "calm satisfied text")

        assert scores.shape == (6,)
        assert scores.dtype == np.float64

    def test_encodes_the_provided_pole_texts(self):
        """The function encodes the pole texts it receives, not derived phrases."""
        points = _make_points(3)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(4, dtype=np.float64)

        with patch(f"{MOD}.SentenceTransformer", return_value=mock_model):
            from src.engine.f_semantic_reembed import _cosine_axis_scores
            _cosine_axis_scores(points, "high pole text", "low pole text")

        encode_calls = [str(c.args[0]) for c in mock_model.encode.call_args_list]
        assert any("high pole text" in c for c in encode_calls)
        assert any("low pole text" in c for c in encode_calls)

    def test_score_direction(self):
        """A point aligned with pole_pos scores positive; one aligned with pole_neg
        scores negative."""
        pos_emb = [1.0, 0.0, 0.0, 0.0]
        neg_emb = [0.0, 1.0, 0.0, 0.0]

        dp_pos = DataPoint()
        dp_pos.id = "pp"
        dp_pos.embedding = pos_emb
        dp_pos.data = {}
        dp_pos.dataset_name = "ds"

        dp_neg = DataPoint()
        dp_neg.id = "pn"
        dp_neg.embedding = neg_emb
        dp_neg.data = {}
        dp_neg.dataset_name = "ds"

        pole_pos = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        pole_neg = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

        mock_model = MagicMock()
        mock_model.encode.side_effect = [pole_pos, pole_neg]

        with patch(f"{MOD}.SentenceTransformer", return_value=mock_model):
            from src.engine.f_semantic_reembed import _cosine_axis_scores
            scores = _cosine_axis_scores([dp_pos, dp_neg], "high text", "low text")

        # dp_pos aligns with pole_pos → score = 1 - 0 = 1
        # dp_neg aligns with pole_neg → score = 0 - 1 = -1
        assert scores[0] > scores[1]
        assert pytest.approx(scores[0], abs=1e-5) == 1.0
        assert pytest.approx(scores[1], abs=1e-5) == -1.0


# ---------------------------------------------------------------------------
# _llm_axis_scores
# ---------------------------------------------------------------------------


class TestLlmAxisScores:
    def test_returns_correct_shape_and_values(self):
        points = _make_points(5)
        mock_response = MagicMock()

        with patch(f"{MOD}.call_llm", return_value=mock_response), \
             patch(f"{MOD}.render_prompt", return_value="prompt"), \
             patch(f"{MOD}.loads_llm_json", return_value=[3, 7, 2, 9, 5]):
            from src.engine.f_semantic_reembed import _llm_axis_scores
            scores = _llm_axis_scores(points, "angry")

        assert scores.shape == (5,)
        np.testing.assert_array_almost_equal(scores, [3.0, 7.0, 2.0, 9.0, 5.0])

    def test_batches_across_points(self):
        """With batch_size=2 and 5 points there should be 3 LLM calls."""
        points = _make_points(5)
        mock_response = MagicMock()

        with patch(f"{MOD}.call_llm", return_value=mock_response) as mock_call, \
             patch(f"{MOD}.render_prompt", return_value="prompt"), \
             patch(f"{MOD}.loads_llm_json", return_value=[5, 5]):
            from src.engine.f_semantic_reembed import _llm_axis_scores
            _llm_axis_scores(points, "angry", batch_size=2)

        # batches: [0:2], [2:4], [4:5] → 3 calls
        assert mock_call.call_count == 3

    def test_large_dataset_uses_sampling(self):
        """With N > LLM_SAMPLE_SIZE, LLM calls are capped at ceil(200/batch)."""
        from src.engine.f_semantic_reembed import LLM_SAMPLE_SIZE
        n_points = LLM_SAMPLE_SIZE + 50
        points = _make_points(n_points)
        mock_response = MagicMock()

        with patch(f"{MOD}.call_llm", return_value=mock_response) as mock_call, \
             patch(f"{MOD}.render_prompt", return_value="prompt"), \
             patch(f"{MOD}.loads_llm_json", return_value=[5.0] * 25), \
             patch(f"{MOD}.deviation"):
            from src.engine.f_semantic_reembed import _llm_axis_scores
            scores = _llm_axis_scores(points, "angry", batch_size=25)

        # Should call LLM ceil(200/25)=8 times, not ceil(250/25)=10 times
        assert mock_call.call_count == (LLM_SAMPLE_SIZE + 24) // 25
        assert len(scores) == n_points

    def test_malformed_response_fills_neutral(self):
        """A JSON parse error must not abort — fill batch with 5.0."""
        points = _make_points(3)

        with patch(f"{MOD}.call_llm", side_effect=ValueError("timeout")), \
             patch(f"{MOD}.render_prompt", return_value="prompt"), \
             patch(f"{MOD}.deviation"):
            from src.engine.f_semantic_reembed import _llm_axis_scores
            scores = _llm_axis_scores(points, "angry")

        assert all(s == 5.0 for s in scores)

    def test_short_llm_list_fills_neutral(self):
        """LLM returning fewer scores than batch size → neutral fill for whole batch."""
        points = _make_points(4)
        mock_response = MagicMock()

        with patch(f"{MOD}.call_llm", return_value=mock_response), \
             patch(f"{MOD}.render_prompt", return_value="prompt"), \
             patch(f"{MOD}.loads_llm_json", return_value=[8]), \
             patch(f"{MOD}.deviation"):
            from src.engine.f_semantic_reembed import _llm_axis_scores
            scores = _llm_axis_scores(points, "angry")

        assert all(s == 5.0 for s in scores)

    def test_render_prompt_receives_axis_and_n(self):
        points = _make_points(3)
        mock_response = MagicMock()

        with patch(f"{MOD}.call_llm", return_value=mock_response), \
             patch(f"{MOD}.render_prompt", return_value="prompt") as mock_render, \
             patch(f"{MOD}.loads_llm_json", return_value=[1, 2, 3]):
            from src.engine.f_semantic_reembed import _llm_axis_scores
            _llm_axis_scores(points, "battery life")

        call_kwargs = mock_render.call_args.kwargs
        assert call_kwargs["axis"] == "battery life"
        assert call_kwargs["n"] == 3


# ---------------------------------------------------------------------------
# reembed_for_axis
# ---------------------------------------------------------------------------


_FAKE_POLES = ("high pole text", "low pole text")


class TestReembedForAxis:
    def test_output_shape_is_n_by_d_plus_1(self):
        """Output shape must be (N, D+1) where D is the original embedding dim."""
        points = _make_points(6, dim=4)  # D=4
        cosine_scores = np.linspace(0.0, 5.0, 6, dtype=np.float64)  # high variance

        with patch(f"{MOD}._generate_axis_poles", return_value=_FAKE_POLES), \
             patch(f"{MOD}._cosine_axis_scores", return_value=cosine_scores):
            from src.engine.f_semantic_reembed import reembed_for_axis
            result = reembed_for_axis(points, "battery")

        assert result.shape == (6, 5)  # D+1 = 4+1
        assert result.dtype == np.float32

    def test_high_variance_uses_cosine_only(self):
        """When cosine variance > threshold, _llm_axis_scores must NOT be called."""
        points = _make_points(6)
        cosine_scores = np.linspace(0.0, 5.0, 6, dtype=np.float64)

        with patch(f"{MOD}._generate_axis_poles", return_value=_FAKE_POLES), \
             patch(f"{MOD}._cosine_axis_scores", return_value=cosine_scores), \
             patch(f"{MOD}._llm_axis_scores") as mock_llm:
            from src.engine.f_semantic_reembed import reembed_for_axis
            reembed_for_axis(points, "angry")

        mock_llm.assert_not_called()

    def test_generate_axis_poles_called_with_axis_label(self):
        """reembed_for_axis must call _generate_axis_poles with the axis label."""
        points = _make_points(6)
        cosine_scores = np.linspace(0.0, 5.0, 6, dtype=np.float64)

        with patch(f"{MOD}._generate_axis_poles", return_value=_FAKE_POLES) as mock_gen, \
             patch(f"{MOD}._cosine_axis_scores", return_value=cosine_scores):
            from src.engine.f_semantic_reembed import reembed_for_axis
            reembed_for_axis(points, "quality")

        mock_gen.assert_called_once_with("quality")

    def test_low_variance_falls_back_to_llm(self):
        """When cosine variance <= threshold, _llm_axis_scores must be called."""
        points = _make_points(6)
        cosine_scores = np.zeros(6, dtype=np.float64)  # variance = 0
        llm_scores = np.linspace(1.0, 6.0, 6, dtype=np.float64)

        with patch(f"{MOD}._generate_axis_poles", return_value=_FAKE_POLES), \
             patch(f"{MOD}._cosine_axis_scores", return_value=cosine_scores), \
             patch(f"{MOD}._llm_axis_scores", return_value=llm_scores) as mock_llm:
            from src.engine.f_semantic_reembed import reembed_for_axis
            reembed_for_axis(points, "angry")

        mock_llm.assert_called_once()

    def test_axis_weight_column_scaling(self):
        """Original cols are scaled by sqrt(1-w); axis col by sqrt(w)."""
        dp0 = DataPoint()
        dp0.id = "p0"
        dp0.embedding = [1.0, 0.0]
        dp0.data = {}
        dp0.dataset_name = "ds"

        dp1 = DataPoint()
        dp1.id = "p1"
        dp1.embedding = [0.0, 1.0]
        dp1.data = {}
        dp1.dataset_name = "ds"

        cosine_scores = np.array([0.0, 5.0], dtype=np.float64)

        with patch(f"{MOD}._generate_axis_poles", return_value=_FAKE_POLES), \
             patch(f"{MOD}._cosine_axis_scores", return_value=cosine_scores):
            from src.engine.f_semantic_reembed import reembed_for_axis
            result = reembed_for_axis([dp0, dp1], "test", axis_weight=0.75)

        assert result.shape == (2, 3)
        assert pytest.approx(float(result[0, 0]), abs=0.01) == 0.5
        assert pytest.approx(float(result[0, 1]), abs=0.01) == 0.0

    def test_raises_on_missing_embedding(self):
        dp = DataPoint()
        dp.id = "bad"
        dp.embedding = None
        dp.data = {}
        dp.dataset_name = "ds"

        with pytest.raises(ValueError, match="missing embeddings"):
            from src.engine.f_semantic_reembed import reembed_for_axis
            reembed_for_axis([dp], "angry")

    def test_output_is_float32(self):
        points = _make_points(4, dim=8)
        cosine_scores = np.linspace(0.0, 3.0, 4, dtype=np.float64)

        with patch(f"{MOD}._generate_axis_poles", return_value=_FAKE_POLES), \
             patch(f"{MOD}._cosine_axis_scores", return_value=cosine_scores):
            from src.engine.f_semantic_reembed import reembed_for_axis
            result = reembed_for_axis(points, "quality")

        assert result.dtype == np.float32

    def test_raises_axis_not_discriminative_when_llm_scores_are_uniform(self):
        """When cosine variance is too low AND LLM scores are nearly uniform,
        reembed_for_axis must raise AxisNotDiscriminativeError so the oracle is
        asked to provide a different axis instead of silently falling back."""
        from src.engine.f_semantic_reembed import (
            AxisNotDiscriminativeError,
            reembed_for_axis,
        )
        points = _make_points(6, dim=8)
        flat_cosine = np.full(6, 0.0, dtype=np.float64)
        flat_llm = np.full(6, 5.0, dtype=np.float64)

        with patch(f"{MOD}._generate_axis_poles", return_value=_FAKE_POLES), \
             patch(f"{MOD}._cosine_axis_scores", return_value=flat_cosine), \
             patch(f"{MOD}._llm_axis_scores", return_value=flat_llm):
            with pytest.raises(AxisNotDiscriminativeError, match="battery life"):
                reembed_for_axis(points, "battery life")


# ---------------------------------------------------------------------------
# _generate_axis_poles
# ---------------------------------------------------------------------------


class TestGenerateAxisPoles:
    """_generate_axis_poles must call the LLM with the axis label and return
    (high_text, low_text). On any failure it must fall back to abstract phrases
    rather than crashing."""

    def _llm_response(self, high: str, low: str):
        import json
        from src.harness import LLMResponse
        return LLMResponse(
            text=json.dumps({"high": high, "low": low}),
            usage={"input_tokens": 10, "output_tokens": 50,
                   "cache_read_tokens": 0, "cache_creation_tokens": 0},
            model="test",
        )

    def test_returns_high_and_low_texts(self):
        from src.engine.f_semantic_reembed import _generate_axis_poles
        resp = self._llm_response(
            high="Absolutely furious, the worst experience ever.",
            low="Perfectly satisfied, works exactly as described.",
        )
        with patch(f"{MOD}.call_llm", return_value=resp):
            high, low = _generate_axis_poles("angry tone")
        assert "furious" in high
        assert "satisfied" in low

    def test_prompt_contains_axis_label(self):
        from src.engine.f_semantic_reembed import _generate_axis_poles
        resp = self._llm_response(high="very angry text", low="very calm text")
        captured = []

        def capture(messages, system, **kwargs):
            captured.append(system)
            return resp

        with patch(f"{MOD}.call_llm", side_effect=capture):
            _generate_axis_poles("battery life")

        assert captured and "battery life" in captured[0]

    def test_falls_back_on_llm_error(self):
        """Any LLM failure must return abstract phrases, not raise."""
        from src.engine.f_semantic_reembed import _generate_axis_poles
        with patch(f"{MOD}.call_llm", side_effect=RuntimeError("API down")):
            high, low = _generate_axis_poles("quality")
        assert high == "very quality"
        assert low == "not quality at all"

    def test_falls_back_on_empty_response(self):
        """An empty high/low in the JSON also triggers fallback."""
        from src.engine.f_semantic_reembed import _generate_axis_poles
        from src.harness import LLMResponse
        resp = LLMResponse(
            text='{"high": "", "low": "something"}',
            usage={"input_tokens": 0, "output_tokens": 0,
                   "cache_read_tokens": 0, "cache_creation_tokens": 0},
            model="test",
        )
        with patch(f"{MOD}.call_llm", return_value=resp):
            high, low = _generate_axis_poles("speed")
        assert high == "very speed"
        assert low == "not speed at all"
