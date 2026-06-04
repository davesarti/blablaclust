# Instruct-Tuned Reembedding — Test Plan and Status

**BlaBlaClust** — DOLSAS 2025-26, University of Trento
Branch: `feature/test-instruct-reembed`

---

## Motivation

The current `f_semantic_reembed.py` builds a hybrid `(N, D+1)` matrix: 384
dimensions from MiniLM plus one scalar axis score (cosine or LLM fallback).
As discussed in [semantic-reembed-report.md](semantic-reembed-report.md), this
creates a structural asymmetry — the axis enters as a single column while the
remaining 384 dimensions still encode the original topic geometry.

The proposed alternative is to use an instruction-tuned embedding model
(`intfloat/multilingual-e5-large-instruct`, 560M parameters) with the axis as
an instruction prefix. The model produces a full `(N, 1024)` space where the
entire geometry is already oriented toward the requested axis — no scalar
appended, no hybrid weighting needed.

---

## Test Design

**Dataset**: `data/imdb_frozen.csv` — 300 documents, binary sentiment labels
(1=negative, 2=positive). Chosen because sentiment is a known-difficult axis
for MiniLM (frequently triggers the LLM fallback at `cosine_variance < 0.01`).

**Three strategies** compared on k-means k=2, measured against ground truth
sentiment labels:

| Strategy | Model | Space shape | What it tests |
|---|---|---|---|
| **baseline** | MiniLM, no axis | (N, 384) | Starting point with no reembedding |
| **current** | MiniLM cosine hybrid | (N, 385) | Production path, cosine-only (no LLM fallback) |
| **instruct** | `multilingual-e5-large-instruct` + instruction | (N, 1024) | Proposed upgrade |

**Metrics**: Silhouette, NMI, ARI against sentiment labels.

**Script**: `scripts/compare_reembed_strategies.py` — standalone, no DB or LLM
calls required.

---

## Status: not run — requires GPU

`intfloat/multilingual-e5-large-instruct` (560M parameters) on CPU exceeds the
available execution time for 300 documents. Encoding 300 texts with this model
takes approximately 15–25 minutes on CPU; the test was not completed in the
current environment.

The script is correct and ready to run. On a GPU (e.g., a single T4 or
equivalent), the same encoding would take approximately 10–30 seconds, making
the comparison fully practical.

**This test has not been run and no results are available.** The script is
committed on `feature/test-instruct-reembed` for future execution in a GPU
environment.

---

## Expected outcome and literature support

Based on the related work, the instruct-tuned approach should outperform the
cosine hybrid on sentiment specifically:

- Fischer & Biemann (2026) — *Perspectives* — use `multilingual-e5-large-instruct`
  and report improvements on instruction-steered axes in 8/9 datasets.
- Su et al. (2023) — *One Embedder, Any Task* — demonstrate that instruction
  fine-tuning substantially improves axis-specific clustering over fixed
  general-purpose embeddings.

The cosine hybrid is expected to show low cosine variance on sentiment
(`< 0.01`), triggering the LLM fallback in production — confirming that MiniLM
does not geometrically discriminate this axis.

---

## Future directions

Running this test on GPU is a concrete next step before deciding whether to
implement the instruct-tuned reembedding path in production. The implementation
complexity is higher than a model swap (the `(N, D+1)` hybrid pipeline and
`_build_hybrid_space` in `umap_projection.py` would need reworking), so
empirical validation of the quality gain first is the right approach.
