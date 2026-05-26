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

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from src.harness import call_llm, extract_json_text, render_prompt
from src.logger import deviation
from src.models import DataPoint

# If the cosine scores have variance below this threshold the embedding model
# does not discriminate the axis well — fall back to LLM scoring instead.
COSINE_VARIANCE_THRESHOLD = 0.01


def _cosine_axis_scores(
    points: list[DataPoint],
    axis_label: str,
) -> np.ndarray:
    """Score each point along a semantic axis via cosine similarity with anchor poles.

    The axis is defined by two sentinel phrases encoded with the same model:
      positive pole: "very {axis_label}"
      negative pole: "not {axis_label} at all"

    Score for point p = dot(emb_p, pole_pos) - dot(emb_p, pole_neg)

    Returns an (N,) float64 array of signed scores.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pole_pos = model.encode(f"very {axis_label}", convert_to_numpy=True).astype(np.float64)
    pole_neg = model.encode(
        f"not {axis_label} at all", convert_to_numpy=True
    ).astype(np.float64)

    scores = np.empty(len(points), dtype=np.float64)
    for i, p in enumerate(points):
        emb = np.array(p.embedding, dtype=np.float64)
        scores[i] = float(np.dot(emb, pole_pos) - np.dot(emb, pole_neg))
    return scores


def _llm_axis_scores(
    points: list[DataPoint],
    axis_label: str,
    batch_size: int = 25,
) -> np.ndarray:
    """Score each point along the axis via LLM batch scoring.

    Costs ceil(N / batch_size) LLM calls. On a malformed LLM response for a
    batch the batch is filled with the neutral value 5.0 and a deviation is
    logged (no exception raised — we never abort clustering because of a
    scoring glitch).

    Returns an (N,) float64 array with values approximately in [0, 10].
    """
    scores: list[float] = []
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        texts = "\n".join(
            f"{j}. {(p.data or {}).get('text', '')[:200]}"
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
            raw = json.loads(extract_json_text(response.text))
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

    cosine_scores = _cosine_axis_scores(points, axis_label)
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
        print(
            f"[semantic-reembed] LLM scores  "
            f"min={axis_scores.min():.1f} max={axis_scores.max():.1f} "
            f"mean={axis_scores.mean():.2f} std={axis_scores.std():.2f}",
            flush=True,
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
