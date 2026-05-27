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
`uncertainty = 1 − max(prob)`. Validation depends on B4 (per-point endorsement
rate); since B4 is not yet implemented, A1's calibration is **unvalidated in v1**
and reported as a raw diagnostic only.

**A2. Turns to convergence (primary process metric)** — turns until the Planner
returns `stop` ([`f_next_best_step`](../src/engine/f_next_best_step.py)) or `status`
= `converged`. **Weighted by feedback type** (`FeedbackEntry.type`: a `global`
reframe ≫ a `point` nudge; weights frozen in the harness). Termination is recorded
via `state_snapshot.reason` with **three distinct codes**, not a binary
genuine/forced split:

- `converged` — Planner returned `stop`; healthy outcome.
- `cognitive_overload` — `cognitive_load >= 4` triggered halt; **system failure
  signal**, reported separately.
- `max_turns_reached` — hit `MAX_TURNS = 20`; ambiguous (long-but-healthy session
  vs. non-convergence), reported separately and inspected case by case.

Aggregate "turns to convergence" statistics are computed over `converged` runs only;
the other two codes are reported as failure rates.

## Family B — LLM-as-oracle + LLM-as-judge

The scripted/LLM **oracle** produces
turns shaped like `InputOracle`; a distinct **judge** scores the result, so the
agent never grades itself.

**B1. Oracle satisfaction (primary outcome)** — the judge
([`prompts/f_eval.txt`](../prompts/f_eval.txt)) reads final clusters + feedback
history and returns a `coherence_score`. Measured relative to what the oracle asked for.

**B2. Cognitive load vs. oracle input** — `cognitive_load_score ∈ [1,5]` produced
**by the Executor** (authoritative source; the Planner consumes but does not
re-score it), plus an objective surface signal (item count + text length in
`Display`). Reported **conditioned on `FeedbackEntry.type`**: a light `point` nudge
should not produce a heavy turn.

**B3. Contradiction tracking** — measure detection
rate, severity, and resolution. Backed by
[`detect_contradiction`](../src/harness.py) (keyword split↔merge on a shared
cluster) and the Executor's `contradiction_detected`. Latest intent must win *and*
the drift must be surfaced. **v1 limit**: detector is keyword-based and misses
paraphrased/semantic contradictions. **v2 candidate**: embed contradiction pairs
and threshold cosine similarity between successive feedback entries on the same
cluster.

**B4. One-point validation** — for sampled points, ask the judge whether point *n*
belongs in cluster *c*; report endorsement rate + mean confidence. Doubles as the
validator for A1's soft-assignment calibration (endorsement should correlate
negatively with `uncertainty`).

## Reproducibility & caveats

- k-means seed fixed (`KMEANS_RANDOM_STATE = 42`); LLM calls logged with **prompt
  hash** ([`logs/llm_calls.jsonl`](../logs/llm_calls.jsonl)); oracle script + type
  weights frozen for v1.
- **Judge model-dependence** (B1/B4): mitigated by a distinct judge, relative
  scoring, and a small human study (N ≈ 5–10). **Protocol**: human raters receive
  the same final-clusters + feedback-history payload the judge sees, score
  `coherence` on the same 1–5 rubric, and (for B4) endorse/reject sampled
  point-cluster assignments. We report Spearman correlation between human and
  judge scores; correlation < 0.6 invalidates the judge for that metric.
- **Contradiction detector is keyword-based** (B3): see B3 entry above for the v2
  plan.

## Implementation status

[`f_eval`](../src/engine/f_eval.py) is **implemented and exported but not yet wired
into the live turn path** ([`backend/routers/turns.py`](../backend/routers/turns.py)
runs only `f_output → f_apply_operations → f_uncertainty → f_next_best_step`). For
v1 the eval harness **calls the judge (B1) out-of-band at end-of-session**; B4's
per-point probe is a planned addition, not yet a function.

## Summary

| ID | Metric | Family | Role | Status |
|---|---|---|---|---|
| A1 | Silhouette + soft-assignment calibration | Math | Secondary diagnostic | Silhouette implemented; calibration unvalidated (blocks on B4) |
| A2 | Turns to convergence (type-weighted) | Math | **Primary process** | Implemented; three-way termination coding to add |
| B1 | Oracle satisfaction (coherence/coverage) | LLM-judge | **Primary outcome** | Judge implemented, out-of-band only |
| B2 | Cognitive load vs. oracle input | LLM + objective | Interaction cost | Implemented (Executor-authored) |
| B3 | Contradiction tracking | LLM-oracle | Robustness | Implemented (keyword v1); semantic v2 planned |
| B4 | One-point validation | LLM-judge | Per-point sanity | Planned |