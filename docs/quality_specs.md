# Quality Specification — v1

*Conversational Clustering (BlaBlaClust) · committed before the main evaluation experiment runs.*

Fixes **which metrics we use and why**, before any numbers are collected. v1
freezes the metric definitions and the LLM-as-oracle script so runs stay
comparable.

## Why this is hard

The **oracle's judgment is the objective — there is no ground truth**
([`README.md`](../README.md), [`AGENTS.md`](../AGENTS.md)). That rules out
supervised metrics (ARI/NMI). We combine two families and treat **no single metric
as the objective**:

- **A. Mathematical** — deterministic, from the DB and logs.
- **B. LLM-as-oracle + LLM-as-judge** — a scripted (v1, then LLM) oracle drives the conversation;
  a separate judge ([`f_eval`](../src/engine/f_eval.py)) scores it.

Headline question: *Does conversational refinement converge toward oracle-accepted
clusterings, and at what cognitive-load cost?*

## Family A — Mathematical

**A1. Cluster goodness** — mean **silhouette** over the embeddings, already logged
per run by [`initial_clustering`](../src/engine/initial_clustering.py) to
[`logs/clustering_runs.jsonl`](../logs/clustering_runs.jsonl), tracked across turns.
**Secondary diagnostic, never optimized against**: the oracle may legitimately want
a low-silhouette grouping (e.g. "enthusiastic" vs "disappointed" reviews — same
topic, opposite axis). Companion: **soft-assignment calibration** —
[`f_uncertainty`](../src/engine/f_uncertainty.py) defines
`uncertainty = 1 − max(prob)`; validated against B4.

**A2. Turns to convergence (primary process metric)** — turns until the Planner
returns `stop` ([`f_next_best_step`](../src/engine/f_next_best_step.py)) or `status`
= `converged`. **Weighted by feedback type** (`FeedbackEntry.type`: a `global`
reframe ≫ a `point` nudge; weights frozen in the harness). **Genuine convergence is
distinguished from a forced stop** (`cognitive_load >= 4` or `MAX_TURNS = 20`) via
`state_snapshot.reason`.

## Family B — LLM-as-oracle + LLM-as-judge

The scripted/LLM **oracle** produces
turns shaped like `InputOracle`; a distinct **judge** scores the result, so the
agent never grades itself.

**B1. Oracle satisfaction (primary outcome)** — the judge
([`prompts/f_eval.txt`](../prompts/f_eval.txt)) reads final clusters + feedback
history and returns a `coherence_score`. Measured relative to what the oracle asked for.

**B2. Cognitive load vs. oracle input** — `cognitive_load_score ∈ [1,5]` from the
Executor/Planner, plus objective surface (item count + text length in
`Display`). Reported **conditioned on `FeedbackEntry.type`**: a light `point` nudge
should not produce a heavy turn.

**B3. Contradiction tracking** — measure detection
rate, severity, and resolution. Backed by
[`detect_contradiction`](../src/harness.py) (keyword split↔merge on a shared
cluster) and the Executor's `contradiction_detected`. Latest intent must win *and*
the drift must be surfaced.

**B4. One-point validation** — for sampled points, ask the judge whether point *n*
belongs in cluster *c*; report endorsement rate + mean confidence.

## Reproducibility & caveats

- k-means seed fixed (`KMEANS_RANDOM_STATE = 42`); LLM calls logged with **prompt
  hash** ([`logs/llm_calls.jsonl`](../logs/llm_calls.jsonl)); oracle script + type
  weights frozen for v1.
- **Judge model-dependence** (B1/B4): mitigated by a distinct judge, relative
  scoring, and a small human study (N ≈ 5–10).
- **Contradiction detector is keyword-based** (B3): misses paraphrased/semantic
  contradictions — a known v1 limit.

## Implementation status

[`f_eval`](../src/engine/f_eval.py) is **implemented and exported but not yet wired
into the live turn path** ([`backend/routers/turns.py`](../backend/routers/turns.py)
runs only `f_output → f_apply_operations → f_uncertainty → f_next_best_step`). For
v1 the eval harness **calls the judge (B1) out-of-band at end-of-session**; B4's
per-point probe is a planned addition, not yet a function.

## Summary

| ID | Metric | Family | Role |
|---|---|---|---|
| A1 | Silhouette + soft-assignment calibration | Math | Secondary diagnostic |
| A2 | Turns to convergence (type-weighted) | Math | **Primary process** |
| B1 | Oracle satisfaction (coherence/coverage) | LLM-judge | **Primary outcome** |
| B2 | Cognitive load vs. oracle input | LLM + objective | Interaction cost |
| B3 | Contradiction tracking | LLM-oracle | Robustness |
| B4 | One-point validation | LLM-judge | Per-point sanity |
