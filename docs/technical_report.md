# BlaBlaClust: A Conversational Clustering System for Iterative Text Dataset Organisation

**DLSAS 2025–26 · University of Trento · Team: vibe-coders**

*P1 — Data ingestion · P2 — Cluster operations · P3 — Core engine · P4 — Evaluation · P5 — Frontend*

---

## Abstract

We present BlaBlaClust, a conversational clustering system that lets a human expert (the *oracle*) iteratively define how a text dataset should be grouped through free-form natural language dialogue. Unlike prior work that requires binary pairwise annotation, structured question answering, or direct UI manipulation, BlaBlaClust interprets unrestricted oracle utterances, translates them into structured clustering operations, and converges on a clustering that reflects the oracle's subjective intent — without any pre-defined ground truth. The system uses a Planner/Executor agentic architecture with GMM-based soft assignments, LLM-driven boundary repair, and an on-demand semantic re-embedding pipeline that reorients the clustering geometry along oracle-specified axes. We evaluate across three public text datasets (Amazon Reviews, IMDB, 20 Newsgroups) using a pre-committed eight-metric quality specification covering geometric quality (A1–A4) and LLM-judge dimensions (B1–B4), with 63 LLM-persona sessions and 6 real-user sessions. The system achieves 84% oracle-satisfied termination, pooled compliance B3 = 0.885 [0.828, 0.935], and a silhouette improvement from 0.031 to 0.140 across the dialogue turns. A separate generalization evaluation shows that converged clusterings absorb 300 new in-domain points with cohort-mean OOD rate 5.5% [3.7%, 7.3%], consistent with the 5% distributional null.

---

## 1. Introduction

Text clustering is a fundamental data exploration task: given a corpus of documents, partition them into meaningful groups. Classical clustering algorithms — k-means, GMM, HDBSCAN — operate on a fixed embedding and produce a single partition without any mechanism for the analyst to express preferences or domain knowledge. The result depends entirely on the distance geometry induced by the embedding model, which rarely aligns with the analyst's intent.

The standard remedy — providing a labelled reference partition and iterating against ARI or NMI — is only applicable when a "correct" grouping exists and is known in advance. In many real-world settings this assumption fails: an analyst exploring a new corpus does not know the correct taxonomy until they have seen the data, and the taxonomy they are building is inherently subjective. What makes a set of customer complaints "the same cluster" depends on the analyst's task, not on an objective ground truth.

BlaBlaClust addresses this by redefining the objective: **the oracle's judgment is the ground truth, and the oracle defines it through dialogue**. Rather than annotating binary pairwise constraints (Schild et al., 2021, 2024), answering structured triplet questions (ClusterLLM, Zhang et al. 2023), or manipulating a visual interface (Perspectives, Fischer & Biemann 2026), the oracle simply describes what is wrong — "this group is too broad", "split by sentiment", "merge these two" — and the system interprets the utterance and acts.

This conversational paradigm creates three non-trivial engineering problems. First, the system must reliably parse free-form natural language into precise structured operations (merge, split, move, rename, re-embed). Second, after each operation it must repair geometric boundary mistakes introduced by distance-only clustering. Third, it must know when the oracle is satisfied and stop proactively rather than awaiting an explicit end signal.

Our contributions are:

1. **A complete conversational clustering system** implementing the Planner/Executor agentic pattern with atomic snapshot semantics, GMM soft assignments, LLM boundary repair, and cognitive load monitoring.
2. **On-demand semantic axis steering** via a hybrid embedding (`D+1` dimensions: original MiniLM + oracle-specified axis score) with auto-selected axis weight and scope control (global vs single-cluster).
3. **A pre-committed eight-metric evaluation framework** covering geometry, dialogue efficiency, system cognitive load, generalization stability, and four LLM-judge dimensions — three of which (A3 cognitive load, B3 compliance, B4 contradiction) have no equivalent in prior interactive clustering literature.
4. **Empirical evaluation** across three public datasets with 63 LLM-persona sessions, 6 real-user sessions, and a generalization test under distribution shift.

---

## 2. Related Work

### 2.1 Schild et al. (2021, 2024) — Interactive Clustering for Chatbot Design

Schild et al. address the same core problem as BlaBlaClust — building an intent taxonomy from scratch without a pre-defined label set — through iterative constraint-based clustering. A domain expert annotates sampled document pairs as MUST-LINK or CANNOT-LINK; the system re-clusters respecting accumulated constraints via COP-KMeans. The 2024 PhD thesis extends this to a full industrial system with parameter optimisation and multi-annotator conflict resolution, validated at scale in a French bank.

This work is the **direct pre-LLM precursor** to BlaBlaClust's paradigm: the oracle discovers a taxonomy iteratively, without a pre-specified label set, by expressing local similarity judgments. BlaBlaClust inherits the core insight but replaces binary annotation with free-form natural language enabled by LLMs, and replaces classical TF-IDF representations with transformer-based embeddings.

### 2.2 ClusterLLM (Zhang et al., 2023)

ClusterLLM uses an LLM as a relational oracle: it poses ~1024 triplet questions (*"which of B or C is semantically closer to A?"*) and fine-tunes a small embedder on the answers, targeting the clustering perspective the user intends (topic, intent, emotion). A second stage uses binary pairwise questions to auto-select the cluster count. Total cost per dataset is approximately $0.60 using GPT-3.5.

BlaBlaClust differs fundamentally in interaction mode: the oracle does not answer structured questions but drives the session conversationally. BlaBlaClust's semantic re-embedding (`f_semantic_reembed`) achieves perspective steering without fine-tuning, instead constructing a temporary hybrid space on request.

### 2.3 Dial-In LLM (Hong et al., 2025)

Dial-In LLM targets large-scale Chinese customer service intent clustering using two fine-tuned LLM judges: a coherence evaluator that labels clusters as good/bad and an intent labeller in Action-Objective format. The iterative algorithm freezes good clusters and re-clusters bad-cluster sentences until a small residual remains, followed by a label-based post-merge step on the unit hypersphere.

This system is **fully automated** — the oracle is a fine-tuned judge, not a human. It achieves strong results (NMI >0.8 on a 55,085-sentence proprietary benchmark) but requires a closed domain with a knowable intent set. BlaBlaClust explicitly relaxes this assumption: there is no correct taxonomy prior to the oracle's session.

### 2.4 Perspectives (Fischer & Biemann, 2026)

Perspectives is a Digital Humanities interactive clustering tool providing direct manipulation on an interactive 2D document map: the user changes cluster assignments, adds or removes clusters, and triggers few-shot contrastive fine-tuning (SetFit + LoRA, 2–16 examples) after accumulating enough validated examples. The system uses `multilingual-e5-large-instruct` embeddings with LLM rewriting for aspect steering.

BlaBlaClust's architecture (React + FastAPI) is directly comparable to Perspectives'. The key distinction is interaction mode: Perspectives requires spatial drag-and-drop on a visual map, while BlaBlaClust requires only text input. BlaBlaClust's semantic re-embedding is also on-demand and session-scoped, while Perspectives' fine-tuning modifies the embedding model persistently.

### 2.5 Positioning

| System | Interaction mode | Ground truth required | Session scope |
|---|---|---|---|
| Schild et al. (2021) | Binary MUST/CANNOT-LINK | No | Per-corpus |
| ClusterLLM | Structured triplet/binary Q&A | No | Per-corpus (offline) |
| Dial-In LLM | Fully automated (no human) | Domain-closed | Per-corpus |
| Perspectives | UI drag-and-drop | No (post-hoc labels for eval) | Per-analyst |
| **BlaBlaClust** | **Free-form conversation** | **No** | **Per-session** |

BlaBlaClust occupies a unique position: the oracle interacts through unrestricted natural language, with no pre-defined taxonomy, and convergence is session-specific — the same corpus might be clustered differently by two oracles with different tasks.

---

## 3. System Design

### 3.1 Agentic Pattern: Planner / Executor

BlaBlaClust implements a Planner/Executor architecture. The system does not respond to oracle feedback in a single monolithic step; instead it separates *what to do next* from *doing it*. The two roles are always distinct, which makes the decision logic independently testable and the execution logic independently replaceable.

The nine functional roles are:

| Role | Function | What it does |
|---|---|---|
| Executor | `f_output` + `f_apply_operations` | LLM turns oracle text into structured operations; dispatched to the clustering engine via TurnBuilder |
| Repair | `f_boundary_repair` | Post-op LLM pass correcting geometrically misplaced boundary points |
| Staging | `TurnBuilder` | In-memory staging layer; single `commit()` writes the turn atomically |
| Sensor | `f_cluster_uncertainty` | Reads GMM soft-assignment posteriors; detects cluster-pair overlap and low cohesion |
| Planner | `f_next_best_step` | Decides: show result, ask a question, or stop |
| Preference tracker | `f_update_preferences` | Distils oracle feedback history into a rolling 3–5 bullet summary injected into subsequent prompts |
| Cognitive load | `f_cognitive_load` | Estimates session complexity from turn count, prompt tokens, and cluster count; caps in `cognitive_load_caps.py` |
| Judge | `f_eval` | End-of-session quality assessment producing A1–A3, B1–B4 |
| Reembedder | `f_semantic_reembed` | Hybrid axis-weighted embedding pipeline for semantic perspective steering |

### 3.2 Turn Loop

Each oracle turn follows a fixed sequence:

```
Oracle natural language input
           │
           ▼
    f_output ──▶ LLM  →  structured operations
           │              (merge / split / rename / move /
           │               semantic_reembed / cluster_reembed)
           │
           ├── semantic_reembed ──▶ semantic_clustering (whole dataset)
           │
           └── structural ops ──▶ f_apply_operations ──▶ TurnBuilder
                                          │
                                          ├──▶ cluster_reembed
                                          │    (single-cluster semantic re-embed)
                                          │
                                          ├──▶ f_boundary_repair
                                          │
                                          ▼
                                  TurnBuilder.commit()
                                  (renormalized soft assignments)
           │
           ▼
    f_cluster_uncertainty ──reads──▶ SoftAssignment table
           │
           ▼
    f_next_best_step ──▶ show / ask / stop
           │
           ├──▶ f_update_preferences
           ▼
       f_eval (at session end)
```

### 3.3 Snapshot Model and TurnBuilder

A critical design decision is the **TurnBuilder staging pattern**. All operations within a single oracle turn read and write an in-memory Python dict (`point_id → {cluster_id: probability}`). New clusters and dissolutions are staged in memory. A single `commit()` at the turn's end writes the entire snapshot atomically, ensuring that turn number, cluster lifecycle columns (`created_at_turn`, `dissolved_at_turn`), and soft-assignment `turn_number` are always in lockstep.

Before TurnBuilder, operations wrote snapshots to the DB immediately for the next operation to read back. This caused snapshot-axis drift and inconsistent state when two operations occurred in the same turn. TurnBuilder eliminates both problems. No engine function calls `db.commit()` directly; only the router owns the transaction.

### 3.4 API and Data Model

The system exposes a REST API over four routers: `/datasets` (upload/list), `/sessions` (create, initial clustering), `/clusters` (read clusters and representative points), and `/turns` (the main oracle-interaction endpoint). A fifth endpoint `/umap` provides 2D UMAP projections for the frontend analytics panel.

The core data model has seven SQLite tables: `datasets`, `data_points`, `sessions`, `clusters`, `soft_assignments`, `turns`, and `eval_cache`. The `soft_assignments` table is the central inference surface: every turn writes a full snapshot of `(data_point_id, cluster_id, turn_number, probability)` rows, enabling the sensor and generalization evaluator to read soft-assignment posteriors at any historical turn.

### 3.5 Frontend

The primary interface is a React + TypeScript + Vite application running at `localhost:5173`. It proxies API calls to the FastAPI backend at `localhost:8000`. The interface provides a cluster panel (names, sizes, descriptions, representative examples), a conversation panel, a UMAP scatter plot for spatial exploration, and an evaluation modal showing per-session A1–B4 scores. A legacy HTML prototype from Sprint 1–2 is preserved at `/ui`.

---

## 4. Implementation

### 4.1 Embeddings

Text embeddings use `all-MiniLM-L6-v2` (sentence-transformers), producing 384-dimensional vectors. The model is loaded as a module-level singleton to avoid 0.5-second reload overhead per re-embed call. Embeddings are stored as JSON arrays in the `data_points` table and generated once at dataset upload time.

An embedding model comparison experiment (detailed in `docs/embedding-model-comparison.md`) showed that BGE-base-768 achieves NMI +20% and ARI +24% over MiniLM on 20 Newsgroups. The upgrade was deferred due to an 8× CPU embedding cost; all experiments reported here use MiniLM.

### 4.2 Initial Clustering: GMM with Soft Assignments

Initial clustering uses a diagonal-covariance Gaussian Mixture Model (`sklearn.mixture.GaussianMixture`, `covariance_type='diag'`). GMM produces native posterior probabilities from the EM algorithm — every data point receives a genuine probability distribution over clusters, as opposed to the softmax-of-distances approximation required by k-means. This matters for downstream components:

- `f_cluster_uncertainty` detects overlap (two clusters share significant posterior mass) and low cohesion (mean winner-probability below threshold) from these posteriors.
- `f_boundary_repair` ranks boundary candidates by the margin between their top-2 posterior probabilities.
- The generalization evaluator uses cluster centroids and per-dimension variances extracted from the converged GMM state.

A comparison experiment (Sprint 4, `docs/gmm-vs-kmeans-report.md`) showed GMM achieves sharper soft assignments (mean max-probability 0.999 vs 0.762 at k=5 for k-means) with no convergence failures, at 1.65× latency overhead and identical end-to-end B1/A1 quality across four evaluation scenarios.

K-means is used as a fallback when GMM fails to converge, and for the `sweep_k` silhouette diagnostic (which predates the GMM integration).

A critical correctness fix (issue #68): after any merge or split operation, the dissolved cluster's probability mass was dropped from non-affected points but never renormalized. A point originally at `{A:0.5, B:0.3, C:0.2}` after dissolving A and B would become `{C:0.2}`, summing to 0.2 rather than 1.0. This silently broke `f_cluster_uncertainty` from turn 2 onward because overlap thresholds (0.30) would never trigger on sub-1.0 distributions. The fix adds a renormalization pass at the end of each merge and split operation.

### 4.3 Operation Dispatch

`f_output` calls the LLM (via `src/harness/`, OpenRouter provider, `google/gemini-3.1-flash-lite`) with the current session state and oracle input, and receives a JSON object with an action type and an operations list. The prompt (`prompts/f_output.txt`) defines the grammar of valid operations and their constraints: `semantic_reembed` is exclusive (cannot be combined with other ops), cluster IDs must be UUIDs from the current cluster list, and cluster count arithmetic (split-only for increases, merge-only for decreases) is explicitly enforced.

`f_apply_operations` routes each operation type to the appropriate function. Unknown operation types are silently skipped for forward-compatibility. A UUID fuzzy-repair step (`_normalize_cluster_ids`) corrects single-character transcription errors in LLM-emitted cluster IDs using difflib sequence matching.

Supported operations: `merge`, `split`, `rename`, `move`, `cluster_reembed`. The `semantic_reembed` operation is intercepted at the router level and routed to `semantic_clustering` rather than through `f_apply_operations`.

### 4.4 Boundary Repair

After any structural operation (merge, split, or `cluster_reembed`), `f_boundary_repair` samples the most uncertain boundary points from each affected cluster — 10% of cluster size, capped at 30 — ranked by the margin between their top-2 soft-assignment probabilities. A single LLM call receives these points and their cluster context and produces accept/reject judgments. Misplaced points are moved via `batch_move_points` before `TurnBuilder.commit()`. This corrects geometric boundary errors where the distance-only GMM placed a point in the wrong cluster relative to the oracle's semantic intent.

### 4.5 Semantic Re-embedding

When the oracle requests a semantic axis ("cluster by sentiment", "split the electronics cluster by battery performance"), the system constructs a hybrid embedding space:

```
X = [ orig_norm × √(1 - w),   axis_score × √w ]    shape: (N, D+1)
```

where `orig_norm` is the L2-normalised MiniLM embedding and `axis_score` is a scalar capturing each point's position on the requested axis. Two scoring strategies are used:

1. **Cosine strategy** (preferred): the axis is defined by two LLM-generated pole texts ("very positive review" / "very negative review"). Each point's axis score is `cosine(emb, pole_pos) - cosine(emb, pole_neg)`. This is free and fast.
2. **LLM fallback**: when cosine variance across the dataset is below threshold (the axis is not captured by the original embedding geometry), the LLM batch-scores up to 200 sampled points on a 0–10 scale anchored to the same poles. The remaining points inherit scores from their nearest neighbor in the original embedding space.

The axis weight `w` is **auto-selected** based on the strategy: cosine strategy → `w = 0.5` (axis already partially in embedding space); LLM strategy → `w = 0.9` (axis orthogonal to embeddings, e.g. sentiment on Amazon Reviews). This auto-selection was critical for correct sentiment splits: at `w = 0.7` the original 30% topic signal dominated, producing extreme-vs-moderate clusters rather than positive-vs-negative.

**Single-cluster re-embedding** (`cluster_reembed`, issue #70): the oracle can zoom into one cluster and apply an axis without affecting the rest of the dataset. `semantic_reembed_cluster` extracts the hard-assigned subset, runs `reembed_for_axis` on those points only, fits GMM on the hybrid subset, dissolves only the parent cluster, and renormalizes non-affected points. Oracle-supplied sub-cluster names are matched to children by sentence-transformer centroid cosine similarity rather than positional assignment, avoiding name swaps.

### 4.6 Cognitive Load Monitoring

`f_cognitive_load` produces a deterministic score in [1, 5] from three signals: turn count, pre-trim prompt token size, and active cluster count, each scored against caps in `cognitive_load_caps.py` (20 turns, 16,000 tokens, 25 clusters). The composite is the maximum of the three sub-scores. The Planner stops the session at score 5. The caps are engineering estimates, not empirically validated thresholds — a cap-sweep experiment is noted as a future improvement.

---

## 5. Evaluation Methodology

### 5.1 Why Not ARI and NMI

ARI and NMI measure how closely a produced clustering matches a reference partition. BlaBlaClust has no reference partition: the oracle defines correctness during the session. On Amazon Reviews, k-means with k=2 produces a topic-split (electronics vs. clothing); if the oracle wants a sentiment split (positive vs. negative), these metrics would return near-zero scores for a correct outcome. As stated in Bontempelli et al. (2020): *"the quality of an interactive clustering solution must be assessed in terms of its alignment with the expert's intent, not in terms of its distance from a predefined ground truth partition."*

### 5.2 Quality Specification (pre-committed)

The eight metrics were defined and frozen before data collection. No post-hoc metric selection or collapse to a single scalar was performed.

**Family A — Mathematical (deterministic)**

| Metric | Operationalisation |
|---|---|
| **A1 Silhouette** | Mean sklearn silhouette over the final clustering (secondary diagnostic — never optimised against, as the oracle may intentionally want a low-silhouette axis split). |
| **A2 Turns to convergence** | Raw and feedback-type-weighted turn count until the Planner returns `stop`. Weighted by: global ×2, cluster ×1, point ×0.5, instructional ×0. Only `converged` sessions enter the distribution; `cognitive_overload` sessions are reported as a failure rate. |
| **A3 Cognitive load** | Deterministic 1–5 score from turn count, prompt tokens, cluster count. Session halts at 5. |
| **A4 OOD rate** | Generalization eval only. Per-cluster Mahalanobis distance reference from base points; new arrivals with d² > base d²₉₅ are flagged as out-of-distribution. Pooled OOD rate; baseline ~5% under the null by construction. |

**Family B — LLM-as-judge**

| Metric | Operationalisation |
|---|---|
| **B1 Overall** | Synthesis verdict by LLM judge over B2/B3/B4. B4 (contradiction) is forgiveness context: a high-B4 (oracle was contradictory) partially excuses low B2/B3 — the system is penalised less when the oracle made it difficult. |
| **B2 Coherence** | Per-cluster judge over top-3 + bottom-2 members (bottom-2 stress-test the edges). Aggregated as mean and min. |
| **B3 Compliance** | Per-turn judge over (oracle request → executed operations) pairs. The cleanest "did the system do what was asked" metric. |
| **B4 Contradiction** | Judge over the oracle's turn history for self-contradictions, criteria drift, and vagueness. Higher = harder oracle. Feeds B1 as forgiveness. |

B1, B3, and B4 have no equivalent in the related work. B2 is conceptually identical to Dial-In LLM's Goodness score but differs in implementation: general-purpose LLM judge (not domain-specific classifier) producing a continuous 0–1 score with textual reasoning.

**Important caveat.** The LLM judges (B1–B4) use the same model family as the system. Inter-rater agreement against a human-labelled subset was not measured; B-scores are relative within-system comparators, not absolute quality scores.

### 5.3 Datasets

Three publicly available text corpora were used:

| Dataset | Domain | Initial silhouette (mean, 95% CI) |
|---|---|---|
| Amazon Reviews | Heterogeneous product reviews (music, books, film, electronics, household goods) | 0.040 [0.039, 0.041] |
| IMDB | Film and TV criticism (homogeneous) | 0.006 [0.005, 0.007] |
| 20 Newsgroups | Usenet posts (5 newsgroups: baseball, computer graphics, gun control, medicine, space) | 0.047 [0.045, 0.049] |

Each dataset uses 1,200 data points. The low initial silhouette reflects the 384-dimensional MiniLM embedding space; the curse of dimensionality reduces Euclidean discriminability before any oracle guidance.

### 5.4 LLM Oracle Simulation

21 LLM-oracle personas were designed to cover the space of realistic oracle interaction modes: conversational (bilingual, vague, contradictory, brisk), structural (count-flipper, delete-requester, multi-merge), and adversarial (prompt-injection red team). Each persona defines a goal, tone notes, target dataset, initial k, and an optional model override. The runner creates a session per persona, drives the turn loop via the live API, and evaluates at session end. Sessions run for up to 15 turns.

### 5.5 Generalization Procedure

To evaluate whether converged clusterings remain coherent under new arrivals, each of the 6 real-user sessions is cloned in-memory. The converged snapshot is frozen. A per-cluster A4 calibration is built from base in-cluster Mahalanobis distances. Then 300 held-out points from the same dataset are embedded and assigned via `assign_gmm_posterior` to the frozen centroids (without recomputing centroids). A1 (silhouette), A4 (OOD rate), and B2 (coherence) are measured before and after ingestion; paired Δ + bootstrap 95% CIs are the inferential signal.

---

## 6. Experimental Results

### 6.1 Persona Arm (LLM-as-oracle, n = 63)

Pooled results across 63 sessions (21 personae × 3 datasets):

| Metric | Pooled (n=63) |
|---|---|
| A1 silhouette (turn-0) | 0.031 [0.026, 0.035] |
| A1 silhouette (last logged) | 0.140 [0.090, 0.194] |
| A2 turns to convergence | 4.08 |
| A2 weighted turns | 5.63 |
| A3 mean cognitive load (1–5) | 1.277 [1.176, 1.387] |
| B1 overall quality | 0.658 [0.621, 0.695] |
| B2 coherence mean | 0.625 [0.583, 0.665] |
| B2 coherence min | 0.485 [0.442, 0.530] |
| B3 compliance | 0.885 [0.828, 0.935] |
| B4 contradiction | 0.266 |

Termination breakdown:

| Dataset | oracle_satisfied | system_stop | max_turns | oracle_parse_error |
|---|---|---|---|---|
| Amazon Reviews (n=21) | 19 (90%) | 1 | 1 | 0 |
| IMDB (n=21) | 18 (86%) | 1 | 1 | 1 |
| 20 Newsgroups (n=21) | 16 (76%) | 4 | 1 | 0 |
| **Pooled (n=63)** | **53 (84%)** | **6** | **3** | **1** |

The system reaches oracle-satisfied termination in 84% of sessions. The 20 Newsgroups dataset shows the highest system-stop rate (4/21 = 19%), driven by persona-dataset mismatch: the 21 personae were designed for product reviews, and Usenet off-topic thread drift caused some personas to produce repetitive requests that escalated cognitive load. A3 stays comfortably below the cap (pooled mean 1.28 / 5), consistent with the low system-stop rate on Amazon and IMDB.

Per-dataset quality:

| Metric | Amazon | IMDB | 20 Newsgroups |
|---|---|---|---|
| B1 overall | 0.654 [0.591, 0.711] | 0.702 [0.636, 0.767] | 0.619 [0.552, 0.682] |
| B2 coherence mean | 0.638 [0.586, 0.687] | 0.698 [0.642, 0.756] | 0.539 [0.454, 0.617] |
| B3 compliance | 0.900 [0.814, 0.967] | 0.885 [0.781, 0.970] | 0.869 [0.757, 0.962] |
| B4 contradiction | 0.274 | 0.240 | 0.283 |

IMDB leads on quality (B1, B2) because the homogeneous film-domain data makes clusters semantically tight. Amazon leads on compliance (B3 = 0.90) and oracle satisfaction rate (90%), as the dataset's natural topic diversity makes structural operations reliable. 20 Newsgroups scores lowest across B1 and B2 due to Usenet off-topic bleed (posts in a baseball newsgroup frequently contain tangential personal anecdotes) reducing LLM-judged coherence.

### 6.2 Dialogue vs No-dialogue Baseline (B2 only)

A no-dialogue baseline was run by an oracle that explicitly approved the initial clustering without changes (n=1 per dataset; a CI over sessions is not computable per dataset).

| Dataset | Baseline B2 (n=1) | Persona dialogue B2 | Real users B2 |
|---|---|---|---|
| Amazon | 0.850 | 0.638 [0.586, 0.687] | 0.670 [0.590, 0.750] |
| IMDB | 0.784 | 0.698 [0.642, 0.756] | 0.633 [0.600, 0.667] |
| 20 Newsgroups | 0.806 | 0.539 [0.454, 0.617] | 0.745 [0.660, 0.830] |
| **Pooled** | **0.813** [0.784, 0.850] | **0.625** [0.583, 0.665] | **0.683** [0.619, 0.751] |

Dialogue consistently reduces judge-rated B2 coherence relative to the no-dialogue baseline on all three datasets (persona CIs exclude the baseline point on all three). This is the most important qualitative finding: **the dialogue loop improves A1 silhouette geometry (0.031 → 0.140) but reduces LLM-judged cluster coherence**. The interpretation is that oracle-guided splits along non-topic axes (sentiment, writing style) produce geometrically meaningful but topically mixed clusters. The trade is not necessarily net-negative — it reflects the oracle intentionally overriding topic structure — but it is a real measurement that must be reported honestly.

### 6.3 Real Users vs Persona Simulation (n=6 vs n=63)

Six real-user sessions were collected (two per dataset) to assess whether the LLM-persona simulation is a valid proxy for live use.

| Metric | Real users (n=6) | Persona pooled (n=63) |
|---|---|---|
| A2 weighted turns | 7.75 | 5.63 |
| A3 mean cognitive load | 1.505 [1.210, 1.805] | 1.277 [1.176, 1.387] |
| B1 overall quality | 0.633 [0.533, 0.745] | 0.658 [0.621, 0.695] |
| B2 coherence mean | 0.683 [0.619, 0.751] | 0.625 [0.583, 0.665] |
| B3 compliance | 0.753 [0.610, 0.900] | 0.885 [0.828, 0.935] |
| B4 contradiction | 0.553 | 0.266 |

Key observations: B1 and B2 CIs overlap substantially — real users produce quality consistent with the persona simulation. However, real users are approximately **2× harder** than the simulation (B4 contradiction 0.553 vs 0.266), use more weighted turns (7.75 vs 5.63), and receive lower compliance scores (B3 0.753 vs 0.885). The compliance gap is sharpest on Amazon (real 0.580 vs persona 0.900), suggesting that vague, position-based ("that last cluster"), and language-switching requests degrade compliance in ways the persona oracle never surfaced.

The persona simulation is a defensible proxy for B1 and B2, but **overstates compliance (B3) and understates oracle difficulty (B4)** relative to real use.

### 6.4 Generalization (n=6 real-user sessions)

After converging 6 real-user sessions, 300 held-out in-domain points were ingested into each frozen clustering.

**A1 (geometry):** Paired Δ is directionally detectable in 2 of 6 sessions but magnitudes are small. 4 of 6 sessions hold A1 (CI spans 0 or positive); 2 degrade (user2/amazon, user6/imdb). Absolute silhouette levels are near zero on IMDB (reflecting heavy text overlap in that embedding space), so "degraded" means a confirmed direction, not structural collapse.

**A4 (calibration):** Cohort-mean OOD rate is 5.5% [3.7%, 7.3%], consistent with the 5% null. Three sessions are unremarkable; user1 and user3 show elevated OOD rates (8.7% and 8.0%) despite stable A1 — a case where A1 alone would miss what A4 catches (new arrivals stretch the cluster envelope without moving existing points). user5/amazon is anomalously tight (1.7% OOD), consistent with a highly concentrated converged geometry.

**B2 (coherence):** One clear failure: user2/amazon shows B2 paired Δ = −0.175 [−0.258, −0.083] — the only session with simultaneous confirmed degradation on both A1 and B2. The remaining 5 sessions hold (CIs span 0 or positive). Cohort-mean B2 drops from 0.675 → 0.640, but the two CIs overlap heavily; the user2 outlier drives the cohort mean.

**Headline:** On average across 6 sessions, ingesting 300 in-domain points does not break converged clusterings. One clean failure (user2/amazon) is surfaced by per-session forest plots but masked by cohort means — a concrete argument for reporting per-session rather than pooled-only.

---

## 7. Discussion

### 7.1 Strengths

**Compliance is the system's most consistent strength.** Pooled B3 = 0.885 [0.828, 0.935] across 63 sessions, reaching 0.980 [0.960, 1.000] for real users on 20 Newsgroups. The dialogue loop executes oracle requests faithfully across very different interaction styles (brisk one-turn, methodical multi-turn, contradictory, multilingual, typo-noisy, adversarial).

**The dialogue loop measurably improves geometric structure.** Pooled A1 silhouette rises from 0.031 at turn-0 to 0.140 at last-logged — a 4.5× improvement. On IMDB and 20 Newsgroups, where semantic re-embedding is most effective (cosine variance is near-zero without oracle guidance), individual sessions show silhouette gains of +0.50 to +0.70.

**Cognitive load is well-calibrated.** Pooled A3 = 1.28/5 on the persona arm and 1.51/5 on real users; `max_turns` and `system_stop` terminations account for 16% of sessions combined, and the `never_satisfied` stress-test persona (designed to never halt) escalates correctly to cognitive load 4 over 14 turns before the cap fires.

**The system is robust to adversarial input.** The `prompt_injector_redteam` persona — which attempted injection payloads including "IGNORE PREVIOUS INSTRUCTIONS" as a cluster name and a request to wipe all clusters — completed without the system leaking architecture or executing injected commands.

### 7.2 Weaknesses

**Dialogue reduces coherence.** The baseline B2 mean (0.813) consistently exceeds the persona dialogue mean (0.625) across all three datasets with non-overlapping CIs. The tradeoff is inherent to the conversational paradigm: an oracle who re-clusters by sentiment intentionally breaks the topical coherence that the initial GMM produced. Whether this tradeoff is net-positive is left to the oracle, but it must be disclosed rather than explained away.

**Real users are harder than the simulation.** B4 contradiction nearly doubles from persona (0.266) to real users (0.553), and B3 compliance drops correspondingly (0.885 → 0.753). The compliance gap on Amazon (0.580 real vs 0.900 persona) suggests that real users more frequently issue vague, position-based, or implicit requests that the current prompt engineering does not handle as reliably.

**20 Newsgroups stresses the system.** The highest system-stop rate (4/21, 19%), lowest B1 (0.619), and lowest B2 mean and min across the persona arm. Contributing factors: Usenet off-topic bleed makes clusters harder to judge as coherent, and the personae (designed for product reviews) produce more confused/repetitive oracle behavior that accelerates cognitive load escalation.

**Single-cluster re-embedding has NN-propagation noise.** When the LLM scores only 200 sampled points and propagates to the rest via nearest-neighbor in the original embedding space, boundary points can inherit incorrect scores (nearest neighbor by topic ≠ nearest neighbor by sentiment). The result is occasional misclassifications in the re-embedded sub-clusters that require oracle correction.

---

## 8. Limitations

**Judge bias is unmeasured.** B1–B4 use the same model family (Gemini Flash Lite) that drives the oracle simulation. No human-labeled validation subset was scored against the judges; inter-rater agreement is not reported. All B-scores should be interpreted as within-system relative measures, not absolute quality judgments.

**Small n on the real-user arm.** Two real users per dataset, six pooled. Per-dataset CIs are wide; dataset-level user-vs-persona claims are suggestive, not confirmatory.

**A1 silhouette is not a clean effect-of-dialogue measure.** Split operations log the subset silhouette; merge and move operations do not log at all. Initial and final silhouette values use different populations, so the observed gain (0.031 → 0.140) is an upper bound on the dialogue effect, not a controlled measurement.

**Cognitive load caps are engineering estimates.** The thresholds in `cognitive_load_caps.py` (20 turns, 16,000 tokens, 25 clusters) were set by engineering judgment, not empirically validated. A cap-sweep experiment was identified as a priority improvement but was not run before submission.

**Embedding model is sub-optimal.** `all-MiniLM-L6-v2` (384-dim) was retained for speed. An embedding model comparison showed BGE-base-768 achieves NMI +20%, ARI +24% on 20 Newsgroups. Upgrading would require an 8× longer embedding pass at dataset upload time.

**Single embedding model for generalization.** The generalization procedure evaluates robustness with the same encoder used during session convergence. Robustness under encoder shift — ingesting points embedded with a different model — is not addressed.

**In-distribution arrivals only.** The frozen evaluation splits are held-out rows of the same source datasets. The A4 OOD analysis measures "does the converged structure absorb more of the same?" — not "does it survive a genuine distribution shift." A cross-domain generalization experiment (e.g., 20 Newsgroups arrivals into an Amazon session) is a natural follow-up.

**B2 sampling in generalization differs from the live endpoint.** The generalization script uses 6 top + 4 bottom cluster members for the B2 judge (raised for stability), while the live `/eval` endpoint uses 3 + 2. B2 scores between these two contexts are not directly comparable.

---

## 9. Ethical Considerations

**Data.** All three datasets (Amazon Reviews, IMDB, 20 Newsgroups) are publicly available research corpora that do not contain personally identifying information. No data was collected from users beyond the 6 in-the-loop sessions conducted by team members. No proprietary, sensitive, or personal data was used at any stage.

**Oracle role.** BlaBlaClust is a tool that *assists* human clustering decisions; it does not automate them. The oracle retains full control: every structural change requires an explicit oracle utterance, the system proposes (show/ask) rather than acting silently, and the oracle can always override the system's suggestions. The system is explicitly designed for tasks where no objective ground truth exists — it empowers the oracle to define correctness, rather than imposing one.

**LLM as judge.** B1–B4 scores are produced by an LLM judge with known biases. We disclose that these judges were not validated against human raters and that all B-metric claims are relative within-system measures. We do not make claims about absolute quality that would require human-validated ground truth.

**Simulation gap.** The LLM-persona simulation overstates compliance (B3) and understates oracle difficulty (B4) relative to real use. We disclose this gap explicitly and recommend against using the persona arm alone to make deployment claims.

**Environmental cost.** LLM API calls for the full evaluation suite (63 persona sessions + baseline + generalization) incur approximately $0 in direct API cost (using `google/gemini-3.1-flash-lite` via OpenRouter, a free-tier model), but represent non-trivial compute and energy use at the model provider's infrastructure. No quantification of carbon footprint was performed.

---

## 10. Conclusion

BlaBlaClust demonstrates that free-form conversational dialogue is a viable interaction paradigm for iterative text clustering without pre-defined ground truth. The system achieves 84% oracle-satisfied termination, strong compliance (pooled B3 = 0.885), and meaningful geometric improvement over initial clustering (A1: 0.031 → 0.140). Converged clusterings absorb in-domain new arrivals without structural collapse (cohort OOD rate 5.5%, consistent with the 5% null). The core tradeoff — dialogue improves geometry but reduces LLM-judged coherence relative to the no-dialogue baseline — is an inherent property of oracle-guided axis reorientation and is reported honestly rather than optimised away.

The main gaps for future work are: (1) replacing `all-MiniLM-L6-v2` with an instruction-tuned embedding model (e.g., `multilingual-e5-large-instruct`) to eliminate the LLM fallback in semantic re-embedding entirely; (2) mining the session log for contrastive fine-tuning signal (splits as negative pairs, merges as positive pairs, following Perspectives); (3) a genuine out-of-distribution generalization test; and (4) a human inter-rater study to validate the B1–B4 LLM judges.

---

## References

- Bontempelli, A. et al. (2020). *Concept-level AI-assisted data annotation: A human-in-the-loop approach for labeling unstructured text data.* (cited in evaluation methodology).
- Fischer, T. & Biemann, C. (2026). *Perspectives – Interactive Document Clustering in the Discourse Analysis Tool Suite.* arXiv:2602.15540.
- Hong, M., Ng, W., Zhang, C.J., Song, Y., & Jiang, D. (2025). *Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues.* EMNLP 2025. arXiv:2412.09049.
- Rousseeuw, P.J. (1987). *Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis.* Journal of Computational and Applied Mathematics, 20, 53–65.
- Schild, E., Lossio-Ventura, M., Lamirel, J.-C., & Roche, M. (2021). *Concevoir un assistant conversationnel de manière itérative et semi-supervisée avec le clustering interactif.* EGC 2021.
- Schild, E. (2024). *Annotation collaborative par clustering interactif pour la conception d'assistants conversationnels.* PhD thesis, Université de Lorraine.
- Walker, M.A., Litman, D.J., Kamm, C.A., & Abella, A. (1997). *PARADISE: A Framework for Evaluating Spoken Dialogue Agents.* ACL 1997.
- Zhang, Y., Wang, Z., & Shang, J. (2023). *ClusterLLM: Large Language Models as a Guide for Text Clustering.* EMNLP 2023. arXiv:2305.14871.
