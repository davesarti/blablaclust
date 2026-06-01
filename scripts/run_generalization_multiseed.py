"""Multi-seed robustness check for the generalization eval.

Runs the nearest-centroid generalization eval over multiple k-means seeds and
reports the mean, std, min and max of the held-out accuracy. This answers the
question: "is the headline 87% stable, or is it a lucky seed?"

A robust result should have std < 2 pp and the full seed range well above the
random baseline (1/k). Embeds once, clusters many times (cheap).

Usage:
    PYTHONPATH=. python scripts/run_generalization_multiseed.py           # 20NG defaults
    PYTHONPATH=. python scripts/run_generalization_multiseed.py \\
        --train data/train.csv --frozen data/frozen_eval.csv --k 2        # Amazon
    PYTHONPATH=. python scripts/run_generalization_multiseed.py \\
        --seeds 0 1 2 3 4 7 13 21 42 99                                   # custom seeds
"""

from __future__ import annotations

import argparse
import collections
import tempfile
from pathlib import Path

import numpy as np

import src.logger as logger
from src.engine.generalization import assign_nearest, centroids_from_snapshot
from src.engine.initial_clustering import initial_clustering
from src.models import DataPoint
from scripts.run_generalization_eval import (
    BOOTSTRAP_ITERS,
    EMBEDDING_MODEL,
    _embed,
    _read_split,
    bootstrap_interval,
    wilson_interval,
)

DEFAULT_SEEDS = [0, 1, 2, 3, 4, 7, 13, 21, 42, 99]


def _run_one_seed(
    train_points: list[DataPoint],
    train_emb: np.ndarray,
    train_labels: list[int],
    frozen_emb: np.ndarray,
    frozen_labels: list[int],
    k: int,
    seed: int,
) -> float:
    """Run one full generalization eval with the given seed. Returns held-out accuracy."""
    logger._clustering_log_path = Path(tempfile.gettempdir()) / f"gen_multiseed_{seed}.jsonl"

    _clusters, db_assignments, _ = initial_clustering(
        train_points, k=k, session_id=f"gen-ms-{seed}", turn_number=0, seed=seed
    )

    snapshot: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for a in db_assignments:
        snapshot[a.data_point_id][a.cluster_id] = a.probability

    point_embeddings = {f"train-{i}": emb for i, emb in enumerate(train_emb)}
    centroid_ids, centroids = centroids_from_snapshot(point_embeddings, snapshot)

    train_hard = {pid: max(dist, key=dist.get) for pid, dist in snapshot.items()}
    members: dict[str, list[int]] = collections.defaultdict(list)
    for i, true_lbl in enumerate(train_labels):
        members[train_hard[f"train-{i}"]].append(true_lbl)

    # Majority-vote cluster→label map.  Track how many labels own a cluster.
    cluster_to_label = {
        cid: collections.Counter(lbls).most_common(1)[0][0]
        for cid, lbls in members.items()
    }

    predicted_clusters = assign_nearest(frozen_emb, centroid_ids, centroids)
    predicted_labels = [cluster_to_label[c] for c in predicted_clusters]
    correct = np.array([int(p == t) for p, t in zip(predicted_labels, frozen_labels)])
    # Track if any cluster label is missing (degenerate partition).
    n_labels_covered = len(set(cluster_to_label.values()))
    return float(correct.mean()), int(correct.sum()), len(frozen_labels), n_labels_covered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/20newsgroups_train.csv")
    parser.add_argument("--frozen", default="data/20newsgroups_frozen.csv")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="List of k-means seeds to try (default: 10 seeds)")
    args = parser.parse_args()

    print(f"Loading + cleaning splits…")
    train_texts, train_labels = _read_split(args.train)
    frozen_texts, frozen_labels = _read_split(args.frozen)
    print(f"  train:  {len(train_texts)} rows  frozen: {len(frozen_texts)} rows")

    print(f"Loading {EMBEDDING_MODEL} and embedding (once)…")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    train_emb = _embed(train_texts, model)
    frozen_emb = _embed(frozen_texts, model)

    # Build DataPoint list once — reused across seeds (embedding is fixed).
    train_points: list[DataPoint] = []
    for i, emb in enumerate(train_emb):
        dp = DataPoint()
        dp.id = f"train-{i}"
        dp.embedding = emb.tolist()
        dp.data = {"title": "", "text": train_texts[i]}
        train_points.append(dp)

    print(f"\nRunning {len(args.seeds)} seeds: {args.seeds}")
    print(f"  k={args.k}  random_baseline={1/args.k:.3f}\n")

    results = []
    for seed in args.seeds:
        acc, succ, n, n_lbl = _run_one_seed(
            train_points, train_emb, train_labels,
            frozen_emb, frozen_labels, args.k, seed,
        )
        degenerate = " ⚠ degenerate partition" if n_lbl < len(set(train_labels)) else ""
        print(f"  seed={seed:3d}  acc={acc:.4f} ({acc*100:.1f}%)  "
              f"correct={succ}/{n}  labels_covered={n_lbl}{degenerate}")
        results.append(acc)

    accs = np.array(results)

    # Aggregate CI: pool all runs → one big Wilson/bootstrap interval.
    # Also compute per-seed stats for the report.
    print("\n" + "=" * 64)
    print(f"MULTI-SEED ROBUSTNESS  ({len(args.seeds)} seeds)")
    print("=" * 64)
    print(f"mean acc:   {accs.mean():.4f}  ({accs.mean()*100:.1f}%)")
    print(f"std:        {accs.std():.4f}  ({accs.std()*100:.1f} pp)")
    print(f"min:        {accs.min():.4f}  ({accs.min()*100:.1f}%)  seed={args.seeds[int(accs.argmin())]}")
    print(f"max:        {accs.max():.4f}  ({accs.max()*100:.1f}%)  seed={args.seeds[int(accs.argmax())]}")
    print(f"range:      {(accs.max()-accs.min())*100:.1f} pp")
    print(f"random baseline: {1/args.k:.3f}  ({100/args.k:.1f}%)")
    # Aggregate accuracy across all seeds (pooled).
    n_frozen = len(frozen_labels)
    pooled_succ = int(round(accs.mean() * n_frozen))
    wlo, whi = wilson_interval(pooled_succ, n_frozen)
    blo, bhi = bootstrap_interval(
        np.array([1.0 if acc >= accs.mean() else 0.0 for acc in results]),
        BOOTSTRAP_ITERS,
    )
    print(f"\nSeed-42 (canonical): {[a for s,a in zip(args.seeds,results) if s==42][0]*100:.1f}%"
          if 42 in args.seeds else "")
    print(f"95% CI (Wilson, mean acc pooled): [{wlo:.4f}, {whi:.4f}]")
    print("-" * 64)
    if accs.std() * 100 < 2.0:
        print("✓ std < 2 pp — result is STABLE across seeds.")
    else:
        print(f"⚠ std = {accs.std()*100:.1f} pp — result has meaningful seed variance.")
    print("=" * 64)


if __name__ == "__main__":
    main()
