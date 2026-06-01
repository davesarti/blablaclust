"""Generalization on Amazon along the SENTIMENT axis (semantic re-embed).

Companion to ``run_generalization_eval.py``. The baseline there clusters Amazon
with plain k-means on MiniLM embeddings and generalizes to held-out *sentiment*
at ~50.7% — the base rate — because the embeddings cluster by **topic**, not
sentiment (both k=2 clusters end up mapped to the same label).

This eval asks the brief's question for the *steered* case: if the oracle
re-orients the embedding space along a **sentiment axis before clustering**
(the system's semantic-re-embed capability), does the codified clustering then
generalize to held-out sentiment labels?

Re-embedding is the same hybrid construction as ``f_semantic_reembed`` but kept
**deterministic** (cosine anchor poles, no LLM) so the number is reproducible:

    sentiment_score(p) = cos(emb_p, positive_pole) - cos(emb_p, negative_pole)
    X = [ row_norm(emb) * sqrt(1 - w) ,  standardize(score) * sqrt(w) ]   # w=0.7

Held-out items are projected into the SAME hybrid space: same pole texts, and
the score is standardized with the TRAIN mean/std (no held-out peeking). Then
the exact same nearest-centroid generalization + Wilson/bootstrap CI as the
baseline is applied.

Usage:
    PYTHONPATH=. python scripts/run_generalization_amazon_axis.py
    PYTHONPATH=. python scripts/run_generalization_amazon_axis.py \
        --train data/train.csv --frozen data/frozen_eval.csv --k 2 --axis-weight 0.7
"""

from __future__ import annotations

import argparse
import collections

import numpy as np

from src.engine.generalization import assign_nearest, centroids_from_snapshot
from src.engine.initial_clustering import KMEANS_RANDOM_STATE, initial_clustering
from src.models import DataPoint
from scripts.run_generalization_eval import (
    EMBEDDING_MODEL,
    _embed,
    _read_split,
    bootstrap_interval,
    BOOTSTRAP_ITERS,
    wilson_interval,
)

# Concrete, in-distribution review-style poles — these embed closer to the data
# than abstract phrases like "positive", so the cosine axis carries more signal.
POSITIVE_POLE = (
    "I love this product. It works perfectly, great quality, exceeded my "
    "expectations. Highly recommend — five stars."
)
NEGATIVE_POLE = (
    "I hate this product. It broke immediately, poor quality, a complete waste "
    "of money. Very disappointed — one star, do not buy."
)


def _cosine_axis(embeddings: np.ndarray, model, pos_text: str, neg_text: str) -> np.ndarray:
    """Signed sentiment score per row: cos(emb, pos) - cos(emb, neg)."""
    pos = model.encode(pos_text, convert_to_numpy=True).astype(np.float64)
    neg = model.encode(neg_text, convert_to_numpy=True).astype(np.float64)
    pos /= np.linalg.norm(pos) + 1e-8
    neg /= np.linalg.norm(neg) + 1e-8
    e = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return e @ pos - e @ neg


def _hybrid(embeddings: np.ndarray, scores: np.ndarray, mu: float, sd: float, w: float) -> np.ndarray:
    """[row_norm(emb) * sqrt(1-w),  standardize(score) * sqrt(w)] — the re-embed space."""
    on = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    ax = ((scores - mu) / sd).reshape(-1, 1)
    return np.hstack([on * np.sqrt(1.0 - w), ax * np.sqrt(w)]).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--frozen", default="data/frozen_eval.csv")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--axis-weight", type=float, default=0.7)
    args = parser.parse_args()

    # Redirect the engine's clustering log to a throwaway file (not a real session).
    import tempfile
    from pathlib import Path
    import src.logger as logger

    logger._clustering_log_path = Path(tempfile.gettempdir()) / "gen_axis_clustering.jsonl"

    print("Loading + cleaning splits…")
    train_texts, train_labels = _read_split(args.train)
    frozen_texts, frozen_labels = _read_split(args.frozen)
    print(f"  train:  {len(train_texts)} rows")
    print(f"  frozen: {len(frozen_texts)} rows")

    print(f"Loading {EMBEDDING_MODEL} and embedding…")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    train_emb = _embed(train_texts, model)
    frozen_emb = _embed(frozen_texts, model)

    # ── Sentiment axis (cosine) + hybrid re-embedding ─────────────────────────
    train_scores = _cosine_axis(train_emb, model, POSITIVE_POLE, NEGATIVE_POLE)
    frozen_scores = _cosine_axis(frozen_emb, model, POSITIVE_POLE, NEGATIVE_POLE)
    mu, sd = float(train_scores.mean()), float(train_scores.std()) + 1e-8
    w = args.axis_weight
    print(
        f"Sentiment axis (cosine): train score mean={mu:.4f} std={sd:.4f} "
        f"min={train_scores.min():.3f} max={train_scores.max():.3f}  axis_weight={w}"
    )
    train_h = _hybrid(train_emb, train_scores, mu, sd, w)
    frozen_h = _hybrid(frozen_emb, frozen_scores, mu, sd, w)

    print(f"Clustering train in the re-embedded space (k={args.k}, seed={KMEANS_RANDOM_STATE})…")
    train_points = []
    for i, hv in enumerate(train_h):
        dp = DataPoint()
        dp.id = f"train-{i}"
        dp.embedding = hv.tolist()
        dp.data = {"title": "", "text": train_texts[i]}
        train_points.append(dp)

    _clusters, db_assignments, _ = initial_clustering(
        train_points, k=args.k, session_id="gen-axis-eval", turn_number=0
    )

    snapshot: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for a in db_assignments:
        snapshot[a.data_point_id][a.cluster_id] = a.probability
    point_embeddings = {f"train-{i}": hv for i, hv in enumerate(train_h)}
    centroid_ids, centroids = centroids_from_snapshot(point_embeddings, snapshot)

    train_hard = {pid: max(dist, key=dist.get) for pid, dist in snapshot.items()}
    members: dict[str, list[int]] = collections.defaultdict(list)
    for i, true_lbl in enumerate(train_labels):
        members[train_hard[f"train-{i}"]].append(true_lbl)
    cluster_to_label = {
        cid: collections.Counter(lbls).most_common(1)[0][0]
        for cid, lbls in members.items()
    }

    # Project held-out into the SAME hybrid space and assign by nearest centroid.
    predicted_clusters = assign_nearest(frozen_h, centroid_ids, centroids)
    predicted_labels = [cluster_to_label[c] for c in predicted_clusters]

    correct = np.array(
        [int(p == t) for p, t in zip(predicted_labels, frozen_labels)], dtype=float
    )
    n = len(correct)
    successes = int(correct.sum())
    accuracy = successes / n
    wilson_lo, wilson_hi = wilson_interval(successes, n)
    boot_lo, boot_hi = bootstrap_interval(correct, BOOTSTRAP_ITERS)

    print("\n" + "=" * 64)
    print("GENERALIZATION (Amazon, SENTIMENT axis) — held-out accuracy")
    print("=" * 64)
    print(f"Held-out items:        {n}")
    print(f"Correct:               {successes}")
    print(f"Accuracy:              {accuracy:.4f}  ({accuracy * 100:.1f}%)")
    print(f"95% CI (Wilson):       [{wilson_lo:.4f}, {wilson_hi:.4f}]")
    print(f"95% CI (bootstrap):    [{boot_lo:.4f}, {boot_hi:.4f}]")
    print(f"Baseline (topic k-means): 0.5067  (50.7%)  — base rate")
    print("-" * 64)
    n_labels = len(set(train_labels))
    mapped = set(cluster_to_label.values())
    print(f"Clusters: {len(centroid_ids)}   distinct true labels in train: {n_labels}")
    print(f"Cluster → majority-label map: {dict(sorted(cluster_to_label.items()))}")
    if len(mapped) < n_labels:
        missing = set(train_labels) - mapped
        print(f"  ⚠ {len(missing)} label(s) own NO cluster: {sorted(missing)}")
    print("-" * 64)
    print("Per-true-label held-out accuracy:")
    by_total = collections.Counter(frozen_labels)
    by_correct: dict[int, int] = collections.defaultdict(int)
    for t, c in zip(frozen_labels, correct):
        by_correct[t] += int(c)
    for lbl in sorted(by_total):
        print(f"  label {lbl}: {by_correct[lbl]}/{by_total[lbl]}  ({by_correct[lbl] / by_total[lbl] * 100:.1f}%)")
    print("=" * 64)


if __name__ == "__main__":
    main()
