# Embedding Model Comparison — MiniLM vs BGE-base

**BlaBlaClust** — DOLSAS 2025-26, University of Trento
Experiment date: 2026-06-04

---

## Motivation

`all-MiniLM-L6-v2` (22M parameters, 384 dimensions) is the current embedding
model for initial dataset embeddings (`generate_embeddings.py`,
`dataset_load_utils.py`). As discussed in [related-work.md](related-work.md),
the related systems use substantially larger models:

| System | Model | Params | Dims |
|---|---|---|---|
| ClusterLLM | Instructor / E5-large | ~330M | 768 |
| Dial-In LLM | BGE-large-zh-v1.5 | 326M | 1024 |
| Perspectives | multilingual-e5-large-instruct | 560M | 1024 |
| **BlaBlaClust (current)** | all-MiniLM-L6-v2 | 22M | 384 |

The architectural gap is real: MiniLM cannot reliably discriminate subtle
semantic axes (sentiment, formality, tone) — this is the root cause of the
frequent LLM fallback in `f_semantic_reembed.py`. The question is whether
upgrading the base embedding model improves clustering quality enough to
justify the cost.

---

## Experiment

### Setup

- **Dataset**: `data/20newsgroups_frozen.csv` — 300 documents, 6 ground-truth
  topic categories (labels 0–5).
- **Comparison**: `all-MiniLM-L6-v2` vs `BAAI/bge-base-en-v1.5` (110M params,
  768 dims). BGE-base is a modern general-purpose encoder with strong MTEB
  benchmark performance; it is in the same size class as the models used in the
  related work.
- **Clustering**: KMeans, k=6 (matching ground truth), 10 random initialisations,
  seed 42. Embeddings L2-normalised before clustering (standard for
  cosine-space models).
- **Hardware**: CPU only (no GPU). Model weights loaded from local HuggingFace
  cache after first download.
- **Script**: `scripts/compare_embeddings.py`

### Metrics

| Metric | What it measures |
|---|---|
| **Silhouette** | Geometric cluster separation (no ground truth needed; directly comparable to production metric A1) |
| **NMI** | Normalised Mutual Information against ground truth labels (0=random, 1=perfect) |
| **ARI** | Adjusted Rand Index against ground truth labels (0=random, 1=perfect) |

---

## Results

| Metric | MiniLM (384d) | BGE-base (768d) | Δ | Relative |
|---|---|---|---|---|
| **Silhouette** | 0.0463 | 0.0702 | +0.0239 | +52% |
| **NMI** | 0.6439 | 0.7707 | +0.1268 | +20% |
| **ARI** | 0.6074 | 0.7518 | +0.1444 | +24% |
| Embedding time (300 docs, CPU) | 28.6 s | 242.2 s | +213.6 s | ×8.5 slower |

The quality improvement is **consistent and significant** across all three
metrics. NMI +0.13 on a 0–1 scale represents a meaningful gain in how well the
initial clustering aligns with the underlying topic structure. ARI +0.14 is
similarly substantial.

The inertia also drops from 251.5 to 118.6 — reflecting that BGE-base produces
more compact, better-separated clusters in its higher-dimensional space.

---

## Decision: keep MiniLM for the current delivery

Despite the quality improvement, we chose **not to merge the BGE upgrade**
(`feature/upgrade-embedding-bge` branch) for the following reasons:

1. **8× slower embedding at dataset load.** For 300 documents on CPU,
   embedding takes 242s with BGE vs 28s with MiniLM. Dataset loading is a
   blocking operation visible to the user in the UI — an 8× slowdown degrades
   the demo experience significantly, and the effect grows linearly with dataset
   size.

2. **One-time cost but high for interactive use.** The embedding runs once at
   dataset load (not per session or per turn), but in an interactive evaluation
   context where datasets are reloaded frequently, the latency accumulates.

3. **No GPU available in the current deployment environment.** BGE-base on GPU
   would reduce the embedding time to approximately 3–5s for 300 documents,
   which would make the tradeoff clearly worthwhile. Without GPU, the raw
   CPU inference cost is too high.

4. **MiniLM is sufficient for topic-level clustering.** The related work confirms
   that MiniLM successfully separates broad topic categories (NMI=0.64 is not
   poor). The quality ceiling only becomes the primary constraint for
   fine-grained axes (sentiment, formality) — which are handled by the
   semantic reembedding pipeline, not the initial embeddings.

The quality gain is real and the upgrade is the correct long-term direction.
The decision to defer is purely pragmatic and time-constrained.

---

## Future directions

**GPU inference.** Running `BAAI/bge-base-en-v1.5` (or larger) on GPU reduces
embedding time by approximately 20–50× compared to CPU, eliminating the
latency tradeoff entirely. `SentenceTransformer.encode()` uses the GPU
automatically when PyTorch detects CUDA — no code change is required beyond
hardware availability.

**Faster intermediate model.** `BAAI/bge-small-en-v1.5` (33M params, 384
dims) and `intfloat/e5-small-v2` (33M params, 384 dims) offer substantial
quality improvements over MiniLM with approximately 2× rather than 8× latency
increase on CPU — a better tradeoff for CPU-only deployments.

**Asynchronous embedding at dataset load.** Making dataset embedding
non-blocking (via RQ background workers, already present in the stack) would
allow users to begin configuring a session while embeddings are computed,
hiding the latency cost entirely at the cost of deferred clustering
availability.

---

## References

- BAAI/bge-base-en-v1.5: [HuggingFace](https://huggingface.co/BAAI/bge-base-en-v1.5)
- MTEB Leaderboard: Massive Text Embedding Benchmark — used as quality proxy
- See [related-work.md](related-work.md) for full discussion of related systems
  and their embedding strategies.
