# Quality Specification

Fixes **which metrics we use and why**, before any numbers are collected. v1
freezes the metric definitions and the LLM-as-oracle script so runs stay
comparable.

## Why this is hard

The **oracle's judgment is the objective — there is no ground truth**
([`README.md`](../README.md), [`AGENTS.md`](../AGENTS.md)). We combine two families and treat **no single metric
as the objective**:

- **A. Mathematical** — deterministic, from the DB and logs.
- **B. LLM-as-oracle + LLM-as-judge** — a scripted (v1, then LLM) oracle drives the conversation;
  a separate judge ([`f_eval`](../src/engine/f_eval.py)) scores it.



## Family A — Mathematical

**A1. Cluster goodness** — mean **silhouette** over the embeddings, already logged
per run by [`initial_clustering`](../src/engine/initial_clustering.py) to
[`logs/clustering_runs.jsonl`](../logs/clustering_runs.jsonl), tracked across turns.
**Secondary diagnostic, never optimized against**: the oracle may legitimately want
a low-silhouette grouping (e.g. "enthusiastic" vs "disappointed" reviews — same
topic, opposite axis). 

**A2. Turns to convergence (primary process metric)** — turns until the Planner
returns `stop` ([`f_next_best_step`](../src/engine/f_next_best_step.py)) or `status`
= `converged`. **Weighted by feedback type** (`FeedbackEntry.type`: a `global`
reframe ≫ a `point` nudge; weights frozen in
[`backend/routers/sessions.py`](../backend/routers/sessions.py):
`global` 2.0, `cluster` 1.0, `point` 0.5, `instructional` 0.0). Termination is recorded
via `state_snapshot.reason` with **two codes**:

- `converged` — Planner did not trigger any stop; healthy outcome.
- `cognitive_overload` — A3 saturated at 5; **system failure signal**, reported
  separately. The `state_snapshot.cognitive_load_driver` field distinguishes
  which signal saturated (`turns` / `tokens` / `clusters`), so "ran out of
  turns" and "prompt got too heavy" are still separable in the report.

Aggregate "turns to convergence" statistics are computed over `converged` runs only;
the `cognitive_overload` code is reported as a failure rate, broken down by driver.

**A3. Cognitive load (LLM-side)** — `cognitive_load_score ∈ [1,5]` computed
**deterministically by `f_cognitive_load`** from three signals: turn count,
pre-trim prompt token size, and active cluster count. The composite is the
max of three per-signal sub-scores against fixed caps
(`src/engine/cognitive_load_caps.py`). The Planner reads `.score` and stops
the session at 5; the eval report reports the `driver` (which signal saturated)
to distinguish stop causes. This metric estimates **agent-side** load — when
the LLM's prompt is bloated enough to start degrading.
The caps (20 turns, 16 000 tokens, 25 clusters) are **arbitrary engineering
estimates**, not empirically validated thresholds. A worthwhile experiment would be to sweep the caps in `src/engine/cognitive_load_caps.py`, re-run the eval suite,
and measure the effect on the metrics.

**A4. Generalization OOD rate (assignment-function fitness)** — for the
generalization procedure only. After a converged session freezes its
centroids, we calibrate a per-cluster **Mahalanobis-distance** reference from
the base in-cluster points. Each cluster ``c`` contributes a per-dim sample
variance vector ``σ²_c ∈ ℝ^d``; new arrivals are scored as:
``d²_M(x, c) = Σ_j (x_j − μ_{c,j})² / (σ²_{c,j} + ε)``
where ``ε = 1e-4`` matches the GMM `reg_covar` regularisation. The OOD
threshold is the 95th percentile of the base ``d²_M`` distribution for that
cluster. Aggregates: **pooled OOD rate** (baseline ≈ 5% under the null — the
complement of the 95th-percentile threshold) and **pooled mean z** with
bootstrap 95% CI, plus a per-cluster breakdown (``n``, ``ood_rate``,
``mean_z``). The covariance is recovered from
base hard-label assignments (argmax of snapshot), so it works whether the
engine ran GMM or k-means fallback without coupling to the fitted model.

## Family B — LLM-as-oracle + LLM-as-judge

The scripted/LLM **oracle** produces
turns shaped like `InputOracle`; distinct **judges** score the result, so the
agent never grades itself. Family B is four independent metrics; **B1 is the
reasoning-based synthesis of the other three**.

**B1. Overall verdict (primary outcome)**
([`prompts/f_eval_overall.txt`](../prompts/f_eval_overall.txt)) — combines B2,
B3, and B4 by reasoning, not arithmetic — a formula cannot distinguish "bad
system" from "bad oracle." When coherence (B2) or compliance (B3) is low but
contradiction (B4) is high, the system is forgiven; it executed a messy oracle
faithfully. When compliance is low against low contradiction, the system is
penalised.

**B2. Cluster coherence**
([`prompts/f_eval_coherence.txt`](../prompts/f_eval_coherence.txt)) — scores each
cluster 0–1 on internal thematic coherence, seeing top-3 + bottom-2 sampled
members. The bottom-2 stress-test the cluster's edges. Aggregates: mean and
**min** (a single bad cluster pulls the run down).

**B3. Oracle compliance**
([`prompts/f_eval_compliance.txt`](../prompts/f_eval_compliance.txt)) — pairs
each oracle turn's request with the operations the system performed and scores
fidelity of request → operation translation. Judges only what the system did,
not whether the oracle was clear (B4 handles that).

**B4. Oracle contradiction**
([`prompts/f_eval_contradiction.txt`](../prompts/f_eval_contradiction.txt)) —
how hard would it have been for the system to understand the oracle's intent?
Looks at the feedback history for self-contradictions, drift in judging
criteria, vague targets, and ambiguity. Higher = harder. Feeds B1 as forgiveness
context: a high B4 partially excuses low B2 / B3.

## Generalization evaluation procedure

Generalization is a procedure that evaluates three signals at two snapshots
around an ingestion event — **A1** (silhouette), **B2** (cluster coherence),
and **A4** (distance-based OOD rate on new arrivals, the only one specific to
this procedure). It reframes the brief's "generalization" question
operationally: *once the oracle is happy, do new points entering the running
system keep the structure intact?* 

- **t0** — eval A1 + B2 on the converged state; build the **A4 calibration**
  once (`calibrate_distance_reference` — per-cluster `(d²_95, μ, σ)` over
  base in-cluster squared distances). The calibration is frozen for the whole
  run, mirroring the centroid freeze.
- **Ingest** a batch of new points: embed → nearest-centroid `assign_nearest`
  against the **frozen** convergence centroids → write a fresh full snapshot at
  `turn + 1` (`ingest_points` in
  [`src/engine/generalization.py`](../src/engine/generalization.py)). Centroids
  are not recomputed; pre-existing points are carried forward verbatim.
- **t1** — eval A1 + B2 again, plus:
    - an A1 sub-aggregate over just the new batch ("do the new points sit
      cleanly relative to the centroids?")
    - **A4 on the new batch** (`assignment_ood`): Mahalanobis d² per new point
      scored against the frozen cluster calibration; pooled OOD rate, pooled
      mean z + bootstrap CI, per-cluster breakdown
    - B2's bottom-2 stress sample naturally surfaces bad new members.
- Report **paired Δ + bootstrap 95% CI** on A1 (over the common, pre-existing
  points) and B2 (over the per-cluster scores), and **pooled OOD rate + mean z
  + CI** on A4. Generalization *holds* when A1 does not drop (paired Δ CI spans
  0 or is positive), **A4's OOD rate stays near its 5% baseline**, and B2 does
  not decline; repeat the ingest/eval loop over multiple batches for a drift
  curve on A1 and A4 (B2 is per-snapshot only by default; pass
  `--b2-every-batch` to score B2 mid-run).

Runner: [`scripts/run_generalization_stability_eval.py`](../scripts/run_generalization_stability_eval.py).