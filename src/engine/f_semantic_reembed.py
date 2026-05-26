"""Semantic re-embedding: project data points along a user-specified semantic axis.

Two strategies and a hybrid selector:
- Cosine anchor poles: free, uses the existing MiniLM embeddings and two
  sentinel phrases to define the positive/negative poles of the axis.
- LLM batch scoring: ~ceil(N/batch_size) LLM calls; works for any axis,
  even ones the embedding model does not capture geometrically.
The hybrid tries cosine first and falls back to LLM only when the cosine
signal has insufficient variance across the dataset.

The result of reembed_for_axis is a hybrid matrix of shape (N, D+1) that
combines the original normalised embeddings (weight alpha) with the
normalised axis score (weight beta). This matrix is fed to k-means in
semantic_clustering.py to produce a clustering whose geometry already
reflects the oracle's semantic intent.
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
    alpha: float = 0.7,
    beta: float = 0.3,
) -> np.ndarray:
    """Compute a hybrid embedding: row-normalised original × alpha || axis × beta.

    Strategy selection:
    - Try cosine anchor poles (free, no LLM call).
    - If cosine variance <= COSINE_VARIANCE_THRESHOLD the model does not
      discriminate the axis; fall back to LLM batch scoring.

    Args:
        points: DataPoint rows, all must have non-None embeddings.
        axis_label: Semantic axis (e.g. "angry", "battery life").
        alpha: Weight for the normalised original embedding component.
        beta: Weight for the normalised axis score component.

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
    if float(np.var(cosine_scores)) > COSINE_VARIANCE_THRESHOLD:
        axis_scores = cosine_scores
    else:
        axis_scores = _llm_axis_scores(points, axis_label)

    # Normalise axis scores to zero-mean / unit-variance so they sit on the
    # same scale as the L2-normalised original embeddings.
    axis_norm = (axis_scores - axis_scores.mean()) / (axis_scores.std() + 1e-8)

    # Row-normalise the original embeddings.
    original = np.array([p.embedding for p in points], dtype=np.float32)
    row_norms = np.linalg.norm(original, axis=1, keepdims=True)
    orig_norm = original / (row_norms + 1e-8)

    # Concatenate: alpha-scaled original | beta-scaled axis column.
    return np.hstack(
        [orig_norm * alpha, axis_norm.reshape(-1, 1) * beta]
    ).astype(np.float32)
