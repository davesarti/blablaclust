"""Compare MiniLM vs BAAI/bge-base-en-v1.5 on 20newsgroups_frozen.csv.

Metrics:
  - Silhouette score (geometric separation, no ground truth needed)
  - NMI and ARI against true labels (ground truth available here)
  - Inertia (within-cluster sum of squares)

Run from the repo root:
    python scripts/compare_embeddings.py
"""

import csv
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

DATA_PATH = Path("data/20newsgroups_frozen.csv")
MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
]
K = 6  # number of unique labels in the dataset
KMEANS_SEED = 42


def load_data(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            text = ((row.get("title") or "") + " " + (row.get("text") or "")).strip()
            texts.append(text[:512])
            labels.append(int(row["label"]))
    return texts, labels


def embed(model_name: str, texts: list[str]) -> np.ndarray:
    print(f"  loading {model_name}...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s  shape={vecs.shape}", flush=True)
    return normalize(vecs)  # L2-normalize (standard for cosine-space clustering)


def cluster_and_score(
    vecs: np.ndarray, true_labels: list[int], k: int
) -> dict:
    km = KMeans(n_clusters=k, random_state=KMEANS_SEED, n_init=10)
    pred = km.fit_predict(vecs)
    return {
        "silhouette": silhouette_score(vecs, pred),
        "nmi":        normalized_mutual_info_score(true_labels, pred),
        "ari":        adjusted_rand_score(true_labels, pred),
        "inertia":    km.inertia_,
        "dims":       vecs.shape[1],
    }


def main():
    print(f"Loading data from {DATA_PATH}...")
    texts, labels = load_data(DATA_PATH)
    print(f"  {len(texts)} documents, {K} classes\n")

    results = {}
    for model_name in MODELS:
        print(f"── {model_name}")
        vecs = embed(model_name, texts)
        scores = cluster_and_score(vecs, labels, K)
        results[model_name] = scores
        print(
            f"  silhouette={scores['silhouette']:.4f}  "
            f"NMI={scores['nmi']:.4f}  "
            f"ARI={scores['ari']:.4f}  "
            f"inertia={scores['inertia']:.1f}  "
            f"dims={scores['dims']}\n"
        )

    print("── Summary ──────────────────────────────────────────")
    base = results[MODELS[0]]
    new  = results[MODELS[1]]
    for metric in ("silhouette", "nmi", "ari"):
        delta = new[metric] - base[metric]
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:12s}  {MODELS[0]}: {base[metric]:.4f}  {MODELS[1]}: {new[metric]:.4f}  Δ={sign}{delta:.4f}")


if __name__ == "__main__":
    main()
