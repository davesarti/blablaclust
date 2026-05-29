"""Generalization evaluation — does a codified clustering label NEW items well?

Protocol (honest, no held-out peeking):
  1. Embed the TRAIN split with all-MiniLM-L6-v2.
  2. Cluster it with the real engine (`initial_clustering`, k-means).
  3. Codify the result into a nearest-centroid mapping function
     (`src.engine.generalization`).
  4. Map each cluster to its MAJORITY TRUE label, using TRAIN labels only.
  5. Embed the FROZEN held-out split, assign each item to the nearest centroid,
     read off the predicted label via the train-derived map.
  6. Report held-out accuracy vs. ground truth + 95% confidence intervals
     (Wilson, closed-form for a proportion; and bootstrap as a cross-check).

This is the project brief's "generalization" question made concrete: once a
clustering exists, can a reusable function assign new items consistently?
20 Newsgroups has real category labels, so this is a genuine accuracy number.

Usage:
    PYTHONPATH=. python scripts/run_generalization_eval.py            # 20NG defaults
    PYTHONPATH=. python scripts/run_generalization_eval.py \
        --train data/20newsgroups_train.csv \
        --frozen data/20newsgroups_frozen.csv --k 6
"""

from __future__ import annotations

import argparse
import collections
import csv
import math

import numpy as np

from src.dataset_processing.text_cleaning import clean_text
from src.engine.generalization import assign_nearest, centroids_from_snapshot
from src.engine.initial_clustering import KMEANS_RANDOM_STATE, initial_clustering
from src.models import DataPoint

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
Z_95 = 1.959963984540054        # standard normal 97.5th percentile
BOOTSTRAP_ITERS = 10000


# ---------------------------------------------------------------------------
# Data loading + embedding
# ---------------------------------------------------------------------------
def _read_split(path: str) -> tuple[list[str], list[int]]:
    """Return (cleaned_texts, true_labels) from a label,title,text CSV."""
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = clean_text(row.get("title") or "", row.get("text") or "")
            if not text:
                continue
            try:
                label = int(row["label"])
            except (KeyError, ValueError):
                continue
            texts.append(text)
            labels.append(label)
    return texts, labels


def _embed(texts: list[str], model) -> np.ndarray:
    return model.encode(
        texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# Confidence intervals for a proportion
# ---------------------------------------------------------------------------
def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score 95% interval — the standard CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_interval(correct: np.ndarray, iters: int, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI over the held-out correct/incorrect flags."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    means = rng.choice(correct, size=(iters, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/20newsgroups_train.csv")
    parser.add_argument("--frozen", default="data/20newsgroups_frozen.csv")
    parser.add_argument("--k", type=int, default=6, help="number of clusters")
    args = parser.parse_args()

    # The clustering run inside this eval is not a real session — redirect its
    # structured log to a throwaway temp file so we don't append to the
    # tracked logs/clustering_runs.jsonl. (logger opens it via Path.open, so
    # this must be a Path.)
    import tempfile
    from pathlib import Path
    import src.logger as logger

    logger._clustering_log_path = Path(tempfile.gettempdir()) / "gen_eval_clustering.jsonl"

    print(f"Loading + cleaning splits…")
    train_texts, train_labels = _read_split(args.train)
    frozen_texts, frozen_labels = _read_split(args.frozen)
    print(f"  train:  {len(train_texts)} rows")
    print(f"  frozen: {len(frozen_texts)} rows")

    print(f"Loading {EMBEDDING_MODEL} and embedding (one-time, ~60-90s on CPU)…")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    train_emb = _embed(train_texts, model)
    frozen_emb = _embed(frozen_texts, model)

    print(f"Clustering train split with k={args.k} (k-means, seed={KMEANS_RANDOM_STATE})…")
    # Run the REAL engine path: in-memory DataPoints → initial_clustering →
    # soft-assignment snapshot. This is exactly what the app persists for a
    # session, so the centroids we codify are the ones the system would use.
    train_points = []
    for i, emb in enumerate(train_emb):
        dp = DataPoint()
        dp.id = f"train-{i}"
        dp.embedding = emb.tolist()
        dp.data = {"title": "", "text": train_texts[i]}
        train_points.append(dp)

    db_clusters, db_assignments, _ = initial_clustering(
        train_points, k=args.k, session_id="generalization-eval", turn_number=0
    )

    # Build the snapshot the engine would persist, then codify it into a
    # nearest-centroid mapping function (exercises the public API).
    snapshot: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for a in db_assignments:
        snapshot[a.data_point_id][a.cluster_id] = a.probability
    point_embeddings = {f"train-{i}": emb for i, emb in enumerate(train_emb)}
    centroid_ids, centroids = centroids_from_snapshot(point_embeddings, snapshot)

    # Hard cluster per training point (argmax of its distribution).
    train_hard = {pid: max(dist, key=dist.get) for pid, dist in snapshot.items()}

    # Map each cluster → its majority TRUE label, using TRAIN labels only.
    members: dict[str, list[int]] = collections.defaultdict(list)
    for i, true_lbl in enumerate(train_labels):
        members[train_hard[f"train-{i}"]].append(true_lbl)
    cluster_to_label = {
        cid: collections.Counter(lbls).most_common(1)[0][0]
        for cid, lbls in members.items()
    }

    # Apply the codified function to the held-out split.
    predicted_clusters = assign_nearest(frozen_emb, centroid_ids, centroids)
    predicted_labels = [cluster_to_label[c] for c in predicted_clusters]

    correct = np.array(
        [int(p == t) for p, t in zip(predicted_labels, frozen_labels)], dtype=float
    )
    n = len(correct)
    successes = int(correct.sum())
    accuracy = successes / n

    wilson_lo, wilson_hi = wilson_interval(successes, n)
    boot_lo, boot_hi = bootstrap_interval(correct, BOOTSTRAP_ITERS)

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 64)
    print("GENERALIZATION EVAL — held-out accuracy via codified clustering")
    print("=" * 64)
    print(f"Held-out items:        {n}")
    print(f"Correct:               {successes}")
    print(f"Accuracy:              {accuracy:.4f}  ({accuracy * 100:.1f}%)")
    print(f"95% CI (Wilson):       [{wilson_lo:.4f}, {wilson_hi:.4f}]")
    print(f"95% CI (bootstrap):    [{boot_lo:.4f}, {boot_hi:.4f}]")
    print("-" * 64)

    n_labels = len(set(train_labels))
    mapped_labels = set(cluster_to_label.values())
    print(f"Clusters: {len(centroid_ids)}   distinct true labels in train: {n_labels}")
    print(f"Cluster → majority-label map: {dict(sorted(cluster_to_label.items()))}")
    if len(mapped_labels) < n_labels:
        missing = set(train_labels) - mapped_labels
        print(f"  ⚠ {len(missing)} true label(s) own NO cluster: {sorted(missing)} "
              f"— their held-out items can never be predicted correctly.")

    # Per-label held-out accuracy (where the system is strong / weak).
    print("-" * 64)
    print("Per-true-label held-out accuracy:")
    by_label_total: dict[int, int] = collections.Counter(frozen_labels)
    by_label_correct: dict[int, int] = collections.defaultdict(int)
    for t, c in zip(frozen_labels, correct):
        by_label_correct[t] += int(c)
    for lbl in sorted(by_label_total):
        tot = by_label_total[lbl]
        cor = by_label_correct[lbl]
        print(f"  label {lbl}: {cor}/{tot}  ({cor / tot * 100:.1f}%)")
    print("=" * 64)


if __name__ == "__main__":
    main()
