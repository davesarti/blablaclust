"""Semantic re-embedding: project data points along a user-specified semantic axis.

Two strategies and a hybrid selector:
- Cosine anchor poles: free, uses the existing MiniLM embeddings and two
  sentinel phrases to define the positive/negative poles of the axis.
- LLM batch scoring: ~ceil(N/batch_size) LLM calls; works for any axis,
  even ones the embedding model does not capture geometrically.
The hybrid tries cosine first and falls back to LLM only when the cosine
signal has insufficient variance across the dataset.

The result of reembed_for_axis is a hybrid matrix of shape (N, D+1).
The axis_weight parameter controls what fraction of the k-means geometry is
driven by the semantic axis (the rest comes from the original embeddings).
With row-normalised original embeddings and a standardised axis score, both
parts have the same expected squared L2 distance between random pairs (~2),
so sqrt-scaling gives exact geometric fractions:

    X = [ orig_norm × sqrt(1 - axis_weight),  axis_score × sqrt(axis_weight) ]

axis_weight=0.7 means 70% of clustering signal comes from the axis.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.harness import call_llm, loads_llm_json, render_prompt, hash_prompt, estimate_cost_usd
from src.logger import deviation, log_llm_call
from src.models import DataPoint

# If the cosine scores have variance below this threshold the embedding model
# does not discriminate the axis well — fall back to LLM scoring instead.
COSINE_VARIANCE_THRESHOLD = 0.01

# Module-level singleton — loading 103 weight files takes ~0.5s even from disk
# cache. Re-embed can be triggered multiple times per session, so we load once.
_ST_MODEL: SentenceTransformer | None = None


def _get_st_model() -> SentenceTransformer:
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL

# Minimum std of LLM scores (0-10 scale) to consider the axis discriminative.
# Below this threshold the axis doesn't separate the data and the oracle is
# asked to pick a different one rather than silently falling back to topic clustering.
LLM_STD_THRESHOLD = 1.0


class AxisNotDiscriminativeError(ValueError):
    """The requested axis doesn't meaningfully vary across the dataset."""


def _generate_axis_poles(axis_label: str) -> tuple[str, str]:
    """Ask the LLM to generate two example texts as concrete axis poles.

    Returns (high_pole_text, low_pole_text). Domain-appropriate examples embed
    better than abstract phrases like "very {axis}" because they live in the
    same distribution as the data points.

    Falls back to abstract phrases ("very {axis}" / "not {axis} at all") on any
    error so the pipeline degrades gracefully without crashing.
    """
    try:
        prompt = render_prompt("semantic_axis_poles", axis_label=axis_label)
        response = call_llm(
            [{"role": "user", "content": f"Generate poles for axis: {axis_label}"}],
            system=prompt,
            max_tokens=512,
        )
        log_llm_call(
            session_id="-",
            prompt_name="semantic_axis_poles",
            prompt_hash=hash_prompt("semantic_axis_poles"),
            usage=response.usage,
            cost_usd=estimate_cost_usd(response.usage, response.model),
            model=response.model,
        )
        parsed = loads_llm_json(response.text)
        high = str(parsed.get("high", "")).strip()
        low = str(parsed.get("low", "")).strip()
        if not high or not low:
            raise ValueError(f"empty pole text: {parsed!r}")
        return high, low
    except Exception as exc:
        deviation(
            "semantic_axis_poles: LLM pole generation failed — using abstract phrases",
            axis_label=axis_label,
            error=str(exc),
        )
        return f"very {axis_label}", f"not {axis_label} at all"


def _cosine_axis_scores(
    points: list[DataPoint],
    pole_pos_text: str,
    pole_neg_text: str,
) -> np.ndarray:
    """Score each point along a semantic axis via cosine similarity with pole texts.

    Score for point p = cos_sim(emb_p, pole_pos) - cos_sim(emb_p, pole_neg)

    Returns an (N,) float64 array of signed scores.
    """
    model = _get_st_model()
    pole_pos = model.encode(pole_pos_text, convert_to_numpy=True).astype(np.float64)
    pole_neg = model.encode(pole_neg_text, convert_to_numpy=True).astype(np.float64)
    # Normalize defensively — SentenceTransformer usually returns unit vectors,
    # but explicit normalization ensures correct cosine similarity.
    pole_pos /= np.linalg.norm(pole_pos) + 1e-8
    pole_neg /= np.linalg.norm(pole_neg) + 1e-8

    scores = np.empty(len(points), dtype=np.float64)
    for i, p in enumerate(points):
        emb = np.array(p.embedding, dtype=np.float64)
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        scores[i] = float(np.dot(emb_norm, pole_pos) - np.dot(emb_norm, pole_neg))
    return scores


# For large datasets, only this many points are scored via LLM; the rest
# inherit the score of their nearest neighbour in the original embedding space.
LLM_SAMPLE_SIZE = 200


def _llm_score_sample(
    points: list[DataPoint],
    axis_label: str,
    batch_size: int,
) -> np.ndarray:
    """Score exactly `points` via LLM batching. Returns float64 (N,)."""
    scores: list[float] = []
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        texts = "\n".join(
            f"{j}. {(p.text or '')[:200]}"
            for j, p in enumerate(batch)
        )
        prompt = render_prompt(
            "semantic_reembed",
            axis=axis_label,
            texts=texts,
            n=len(batch),
        )
        messages = [{"role": "user", "content": texts}]
        try:
            response = call_llm(messages, system=prompt)
            log_llm_call(
                session_id="-",
                prompt_name="semantic_reembed",
                prompt_hash=hash_prompt("semantic_reembed"),
                usage=response.usage,
                cost_usd=estimate_cost_usd(response.usage, response.model),
                model=response.model,
            )
            raw = loads_llm_json(response.text)
            if isinstance(raw, list) and len(raw) >= len(batch):
                batch_scores = [float(raw[j]) for j in range(len(batch))]
            else:
                raise ValueError(
                    f"expected list of {len(batch)} scores, got {raw!r}"
                )
        except Exception as exc:
            deviation(
                "semantic_reembed LLM batch scoring failed — using neutral 5.0",
                batch_start=i,
                axis_label=axis_label,
                error=str(exc),
            )
            batch_scores = [5.0] * len(batch)
        scores.extend(batch_scores)
    return np.array(scores, dtype=np.float64)


def _llm_axis_scores(
    points: list[DataPoint],
    axis_label: str,
    batch_size: int = 25,
) -> np.ndarray:
    """Score each point along the axis via LLM batch scoring.

    For datasets larger than LLM_SAMPLE_SIZE, scores a random sample and
    propagates each score to the nearest-neighbour in the original embedding
    space, keeping LLM calls to ceil(LLM_SAMPLE_SIZE / batch_size) regardless
    of dataset size.

    On a malformed LLM response the affected batch is filled with 5.0 (neutral).

    Returns an (N,) float64 array with values approximately in [0, 10].
    """
    n = len(points)
    if n <= LLM_SAMPLE_SIZE:
        n_calls = (n + batch_size - 1) // batch_size
        print(
            f"[semantic-reembed] LLM scoring {n} points  calls={n_calls}",
            flush=True,
        )
        return _llm_score_sample(points, axis_label, batch_size)

    # Sample LLM_SAMPLE_SIZE points, score them, propagate via NN.
    rng = np.random.default_rng(42)
    sample_idx = sorted(
        rng.choice(n, LLM_SAMPLE_SIZE, replace=False).tolist()
    )
    sampled = [points[i] for i in sample_idx]
    n_calls = (LLM_SAMPLE_SIZE + batch_size - 1) // batch_size
    print(
        f"[semantic-reembed] LLM scoring {LLM_SAMPLE_SIZE}/{n} points (sample)  "
        f"calls={n_calls}  (was {(n + batch_size - 1) // batch_size} without sampling)",
        flush=True,
    )
    sample_scores = _llm_score_sample(sampled, axis_label, batch_size)

    sample_embs = np.array([p.embedding for p in sampled], dtype=np.float64)
    all_embs    = np.array([p.embedding for p in points],  dtype=np.float64)

    sample_norms = np.linalg.norm(sample_embs, axis=1, keepdims=True) + 1e-8
    all_norms    = np.linalg.norm(all_embs,    axis=1, keepdims=True) + 1e-8
    sims    = (all_embs / all_norms) @ (sample_embs / sample_norms).T
    nearest = sims.argmax(axis=1)
    scores  = sample_scores[nearest].copy()
    for local_i, global_i in enumerate(sample_idx):
        scores[global_i] = sample_scores[local_i]

    print(
        f"[semantic-reembed] NN propagation  "
        f"std={scores.std():.3f}  "
        f"min={scores.min():.2f}  max={scores.max():.2f}",
        flush=True,
    )
    return scores


def reembed_for_axis(
    points: list[DataPoint],
    axis_label: str,
    axis_weight: float = 0.7,
) -> np.ndarray:
    """Compute a hybrid embedding where axis_weight controls geometric influence.

    axis_weight is the exact fraction of k-means distance driven by the
    semantic axis. axis_weight=0.7 means 70% axis, 30% original embeddings.

    Strategy selection:
    - Try cosine anchor poles (free, no LLM call).
    - If cosine variance <= COSINE_VARIANCE_THRESHOLD the model does not
      discriminate the axis; fall back to LLM batch scoring.

    Args:
        points: DataPoint rows, all must have non-None embeddings.
        axis_label: Semantic axis (e.g. "angry", "battery life").
        axis_weight: Fraction [0, 1] of k-means signal from the axis.

    Returns:
        Float32 array of shape (N, D+1) where D is the original embedding dim.

    Raises:
        ValueError: if any point has a None embedding.
    """
    missing = [p.id for p in points if p.embedding is None]
    if missing:
        raise ValueError(
            f"points missing embeddings (first 5): {missing[:5]}"
        )

    pole_pos_text, pole_neg_text = _generate_axis_poles(axis_label)
    print(
        f"[semantic-reembed] poles  "
        f"pos={pole_pos_text[:70]!r}  neg={pole_neg_text[:70]!r}",
        flush=True,
    )
    cosine_scores = _cosine_axis_scores(points, pole_pos_text, pole_neg_text)
    cosine_var = float(np.var(cosine_scores))
    if cosine_var > COSINE_VARIANCE_THRESHOLD:
        print(
            f"[semantic-reembed] axis='{axis_label}'  strategy=cosine  "
            f"variance={cosine_var:.4f}  "
            f"scores min={cosine_scores.min():.3f} max={cosine_scores.max():.3f} "
            f"mean={cosine_scores.mean():.3f} std={cosine_scores.std():.3f}",
            flush=True,
        )
        axis_scores = cosine_scores
    else:
        print(
            f"[semantic-reembed] axis='{axis_label}'  strategy=LLM-fallback  "
            f"cosine_variance={cosine_var:.4f} <= threshold={COSINE_VARIANCE_THRESHOLD}",
            flush=True,
        )
        axis_scores = _llm_axis_scores(points, axis_label)
        llm_std = float(axis_scores.std())
        print(
            f"[semantic-reembed] LLM scores  "
            f"min={axis_scores.min():.1f} max={axis_scores.max():.1f} "
            f"mean={axis_scores.mean():.2f} std={llm_std:.2f}",
            flush=True,
        )
        if llm_std < LLM_STD_THRESHOLD:
            raise AxisNotDiscriminativeError(
                f"The axis '{axis_label}' does not vary significantly across the "
                f"dataset (LLM score std={llm_std:.2f} < threshold={LLM_STD_THRESHOLD}). "
                f"Try an axis that is clearly present and varies in the data."
            )

    # Standardise axis scores (zero-mean, unit-variance). Combined with
    # row-normalised embeddings (unit norm), both components have expected
    # squared pairwise distance ~2, so sqrt-scaling gives exact fractions.
    axis_norm = (axis_scores - axis_scores.mean()) / (axis_scores.std() + 1e-8)

    # Row-normalise the original embeddings.
    original = np.array([p.embedding for p in points], dtype=np.float32)
    row_norms = np.linalg.norm(original, axis=1, keepdims=True)
    orig_norm = original / (row_norms + 1e-8)

    orig_scale = float(np.sqrt(1.0 - axis_weight))
    ax_scale   = float(np.sqrt(axis_weight))
    print(
        f"[semantic-reembed] axis_weight={axis_weight:.2f}  "
        f"orig_scale={orig_scale:.3f}  axis_scale={ax_scale:.3f}",
        flush=True,
    )
    return np.hstack(
        [orig_norm * orig_scale, axis_norm.reshape(-1, 1) * ax_scale]
    ).astype(np.float32)
