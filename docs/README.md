# Documentation Index

This folder contains technical reports and research documentation for
**BlaBlaClust** (DOLSAS 2025-26, University of Trento).

---

## How to read this folder

Documents are grouped below by type. Each entry shows the author, status,
and a one-line summary. **Start with the design documents** to understand the
system, then read the implementation reports for the features you are evaluating.

---

## System design

| Document | Author | Description |
|---|---|---|
| [data-model.md](data-model.md) | P1 | Database schema, ORM models, entity relationships |
| [quality_specs.md](quality_specs.md) | P1/P3 | Evaluation metric definitions (A1–A3, B1–B4) and acceptance thresholds |
| [api-interface.md](api-interface.md) | P1 | Full HTTP/JSON API reference: all endpoints, request/response schemas, error codes |

---

## Implementation reports

Technical documentation of features developed during the project.

| Document | Author | Description |
|---|---|---|
| [semantic-reembed-report.md](semantic-reembed-report.md) | P5 | Full implementation of the semantic reembedding pipeline (`f_semantic_reembed.py`): architecture, cosine strategy, LLM fallback, hybrid matrix, test results, future directions |
| [semantic-reembed-diagrams.ipynb](semantic-reembed-diagrams.ipynb) | P5 | Slide-ready matplotlib diagrams: pipeline flowchart, MiniLM embedding space, variance check, hybrid matrix construction |
| [gmm-vs-kmeans-report.md](gmm-vs-kmeans-report.md) | P2 | Comparison of GMM and k-means clustering algorithms; rationale for the final choice |
| [generalization-stability-report.md](generalization-stability-report.md) | P4 | Online generalization (robustness under in-distribution growth) across the 6 user sessions: A1 / A4 / B2 paired Δ with bootstrap + Wilson CIs, forest plots, per-session and cohort-level analysis, failure cases |
| [final_experimental_report.md](final_experimental_report.md) | P4 | End-to-end experimental results: B2 vs no-dialogue baseline, persona quality (B1–B4), real-user session overlay, silhouette evolution — all with CIs and forest plots |

---

## Research and state of the art

Literature review and positioning of BlaBlaClust relative to related work.

| Document | Author | Description |
|---|---|---|
| [related-work.md](related-work.md) | P5 | Academic comparison with 5 related systems (Schild 2021, ClusterLLM, Dial-In LLM, Perspectives); interaction paradigm, embedding strategy, evaluation tables; original contributions |
| [related-work-radar.ipynb](related-work-radar.ipynb) | P5 | Radar chart comparing all 5 systems across 6 design dimensions with scoring rationale |
| [evaluation_literature_report.md](evaluation_literature_report.md) | P4 | Literature grounding for every evaluation metric (A1–A3, B1–B4): why ARI/NMI are excluded, academic basis for each metric (Rousseeuw 1987, PARADISE, NASA-TLX, G-Eval, Malaviya 2025), originality table, and recommended framing for the final presentation |

---

## Experiments

Quantitative tests on design alternatives that informed implementation decisions.

| Document | Author | Status | Description |
|---|---|---|---|
| [embedding-model-comparison.md](embedding-model-comparison.md) | P5 | ✅ Run | MiniLM vs `BAAI/bge-base-en-v1.5` on 20newsgroups: BGE achieves NMI +20%, ARI +24% at 8× CPU cost. Upgrade deferred — see branch `feature/upgrade-embedding-bge` |
| [instruct-reembed-comparison.md](instruct-reembed-comparison.md) | P5 | ⏳ GPU required | Test plan for MiniLM cosine hybrid vs `multilingual-e5-large-instruct` on IMDB sentiment axis. Not run — model requires GPU (>15 min on CPU for 300 docs) |

---

## Archived / superseded

> These documents are preserved as historical record. Do not use them as
> reference for the current implementation.

| Document | Superseded by |
|---|---|
| `semantic-reembed-proposal.md` *(deleted)* | [semantic-reembed-report.md](semantic-reembed-report.md) |