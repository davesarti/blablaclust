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
            scores = _cosine_axis_scores(points, "angry")

        assert scores.shape == (6,)
        assert scores.dtype == np.float64

    def test_encodes_positive_and_negative_poles(self):
        points = _make_points(3)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(4, dtype=np.float64)

        with patch(f"{MOD}.SentenceTransformer", return_value=mock_model):
            from src.engine.f_semantic_reembed import _cosine_axis_scores
            _cosine_axis_scores(points, "sentiment")

        encode_calls = [str(c.args[0]) for c in mock_model.encode.call_args_list]
        assert any("very sentiment" in c for c in encode_calls)
        assert any("not sentiment at all" in c for c in encode_calls)

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
            scores = _cosine_axis_scores([dp_pos, dp_neg], "test")

        # dp_pos: dot([1,0,0,0],[1,0,0,0]) - dot([1,0,0,0],[0,1,0,0]) = 1 - 0 = 1
        # dp_neg: dot([0,1,0,0],[1,0,0,0]) - dot([0,1,0,0],[0,1,0,0]) = 0 - 1 = -1
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

        with (
            patch(f"{MOD}.call_llm", return_value=mock_response),
            patch(f"{MOD}.render_prompt", return_value="prompt"),
            patch(f"{MOD}.extract_json_text", return_value="[3, 7, 2, 9, 5]"),
        ):
            from src.engine.f_semantic_reembed import _llm_axis_scores
            scores = _llm_axis_scores(points, "angry")

        assert scores.shape == (5,)
        np.testing.assert_array_almost_equal(scores, [3.0, 7.0, 2.0, 9.0, 5.0])

    def test_batches_across_points(self):
        """With batch_size=2 and 5 points there should be 3 LLM calls."""
        points = _make_points(5)
        mock_response = MagicMock()

        with (
            patch(f"{MOD}.call_llm", return_value=mock_response) as mock_call,
            patch(f"{MOD}.render_prompt", return_value="prompt"),
            patch(f"{MOD}.extract_json_text", return_value="[5, 5]"),
        ):
            from src.engine.f_semantic_reembed import _llm_axis_scores
            _llm_axis_scores(points, "angry", batch_size=2)

        # batches: [0:2], [2:4], [4:5] → 3 calls
        assert mock_call.call_count == 3

    def test_malformed_response_fills_neutral(self):
        """A JSON parse error must not abort — fill batch with 5.0."""
        points = _make_points(3)

        with (
            patch(f"{MOD}.call_llm", side_effect=ValueError("timeout")),
            patch(f"{MOD}.render_prompt", return_value="prompt"),
            patch(f"{MOD}.deviation"),
        ):
            from src.engine.f_semantic_reembed import _llm_axis_scores
            scores = _llm_axis_scores(points, "angry")

        assert all(s == 5.0 for s in scores)

    def test_short_llm_list_fills_neutral(self):
        """LLM returning fewer scores than batch size → neutral fill for whole batch."""
        points = _make_points(4)
        mock_response = MagicMock()

        with (
            patch(f"{MOD}.call_llm", return_value=mock_response),
            patch(f"{MOD}.render_prompt", return_value="prompt"),
            patch(f"{MOD}.extract_json_text", return_value="[8]"),  # only 1, need 4
            patch(f"{MOD}.deviation"),
        ):
            from src.engine.f_semantic_reembed import _llm_axis_scores
            scores = _llm_axis_scores(points, "angry")

        assert all(s == 5.0 for s in scores)

    def test_render_prompt_receives_axis_and_n(self):
        points = _make_points(3)
        mock_response = MagicMock()

        with (
            patch(f"{MOD}.call_llm", return_value=mock_response),
            patch(f"{MOD}.render_prompt", return_value="prompt") as mock_render,
            patch(f"{MOD}.extract_json_text", return_value="[1, 2, 3]"),
        ):
            from src.engine.f_semantic_reembed import _llm_axis_scores
            _llm_axis_scores(points, "battery life")

        call_kwargs = mock_render.call_args.kwargs
        assert call_kwargs["axis"] == "battery life"
        assert call_kwargs["n"] == 3


# ---------------------------------------------------------------------------
# reembed_for_axis
# ---------------------------------------------------------------------------


class TestReembedForAxis:
    def test_output_shape_is_n_by_d_plus_1(self):
        """Output shape must be (N, D+1) where D is the original embedding dim."""
        points = _make_points(6, dim=4)  # D=4
        cosine_scores = np.linspace(0.0, 5.0, 6, dtype=np.float64)  # high variance

        with patch(f"{MOD}._cosine_axis_scores", return_value=cosine_scores):
            from src.engine.f_semantic_reembed import reembed_for_axis
            result = reembed_for_axis(points, "battery")

        assert result.shape == (6, 5)  # D+1 = 4+1
        assert result.dtype == np.float32

    def test_high_variance_uses_cosine_only(self):
        """When cosine variance > threshold, _llm_axis_scores must NOT be called."""
        points = _make_points(6)
        cosine_scores = np.linspace(0.0, 5.0, 6, dtype=np.float64)

        with (
            patch(f"{MOD}._cosine_axis_scores", return_value=cosine_scores),
            patch(f"{MOD}._llm_axis_scores") as mock_llm,
        ):
            from src.engine.f_semantic_reembed import reembed_for_axis
            reembed_for_axis(points, "angry")

        mock_llm.assert_not_called()

    def test_low_variance_falls_back_to_llm(self):
        """When cosine variance <= threshold, _llm_axis_scores must be called."""
        points = _make_points(6)
        cosine_scores = np.zeros(6, dtype=np.float64)  # variance = 0
        llm_scores = np.linspace(1.0, 6.0, 6, dtype=np.float64)

        with (
            patch(f"{MOD}._cosine_axis_scores", return_value=cosine_scores),
            patch(f"{MOD}._llm_axis_scores", return_value=llm_scores) as mock_llm,
        ):
            from src.engine.f_semantic_reembed import reembed_for_axis
            reembed_for_axis(points, "angry")

        mock_llm.assert_called_once()

    def test_alpha_beta_column_weighting(self):
        """First D cols are alpha-scaled; last col is beta-scaled axis."""
        # Two points so variance is non-zero → cosine path taken
        dp0 = DataPoint()
        dp0.id = "p0"
        dp0.embedding = [1.0, 0.0]  # unit vector → row-norm = 1 → stays [1, 0]
        dp0.data = {}
        dp0.dataset_name = "ds"

        dp1 = DataPoint()
        dp1.id = "p1"
        dp1.embedding = [0.0, 1.0]
        dp1.data = {}
        dp1.dataset_name = "ds"

        # High-variance cosine scores: 0.0 vs 5.0 → variance = 6.25 > 0.01
        cosine_scores = np.array([0.0, 5.0], dtype=np.float64)

        with patch(f"{MOD}._cosine_axis_scores", return_value=cosine_scores):
            from src.engine.f_semantic_reembed import reembed_for_axis
            result = reembed_for_axis([dp0, dp1], "test", alpha=0.6, beta=0.4)

        assert result.shape == (2, 3)
        # dp0 embedding [1, 0] → row-normed [1, 0] → alpha-scaled [0.6, 0]
        assert pytest.approx(float(result[0, 0]), abs=0.01) == 0.6
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

        with patch(f"{MOD}._cosine_axis_scores", return_value=cosine_scores):
            from src.engine.f_semantic_reembed import reembed_for_axis
            result = reembed_for_axis(points, "quality")

        assert result.dtype == np.float32
