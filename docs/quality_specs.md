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
`uncertainty = 1 − max(prob)`. Calibration is **unvalidated** and reported as a raw
diagnostic only; consumers (UI, eval) should not rely on it as a quality signal.

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
the LLM's prompt is bloated enough to start degrading. Oracle-side cognitive
load (what the human has to process) is a separate concern not measured here.
Deterministic and read straight from the per-turn `state_snapshot`.

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
not whether the oracle was clear (B4 handles that). **Caveat**: the compliance
judge receives whatever `target_cluster_ids` the oracle provides; sessions
where target IDs are omitted (e.g. `contradictory_oracle.json`) score 0 on the
operation-target matching signal even when textual intent is clear. Use B3 scores
from such scenarios to test agent robustness, not compliance fidelity.

**B4. Oracle contradiction**
([`prompts/f_eval_contradiction.txt`](../prompts/f_eval_contradiction.txt)) —
how hard would it have been for the system to understand the oracle's intent?
Looks at the feedback history for self-contradictions, drift in judging
criteria, vague targets, and ambiguity. Higher = harder. Feeds B1 as forgiveness
context: a high B4 partially excuses low B2 / B3.

## Reproducibility & caveats

- k-means seed fixed (`KMEANS_RANDOM_STATE = 42`); LLM calls logged with **prompt
  hash** ([`logs/llm_calls.jsonl`](../logs/llm_calls.jsonl)); oracle script + type
  weights frozen for v1.
- **Judge model-dependence** (B1): mitigated by a distinct judge, relative
  scoring, and a small human study (N ≈ 5–10). **Protocol**: human raters receive
  the same final-clusters + feedback-history payload the judge sees and score
  `coherence` on the same 1–5 rubric. We report Spearman correlation between human
  and judge scores; correlation < 0.6 invalidates the judge for that metric.

## Implementation status

The Family-B judges are **out-of-band**: the live turn path
([`backend/routers/turns.py`](../backend/routers/turns.py)) runs only
`f_output → f_apply_operations → f_uncertainty → f_next_best_step`; the eval harness
calls the four judges (coherence, compliance, contradiction, overall) at
end-of-session.

## Summary

| ID | Metric | Family | Role | Status |
|---|---|---|---|---|
| A1 | Silhouette + soft-assignment calibration | Math | Secondary diagnostic | Silhouette implemented; calibration unvalidated |
| A2 | Turns to convergence (type-weighted) | Math | **Primary process** | Implemented; two termination codes (`converged`, `cognitive_overload`) |
| A3 | Cognitive load vs. oracle input | Math (engine-authored) | Interaction cost | Implemented (Executor-authored) |
| B1 | Overall verdict (synthesis of B2 + B3 + B4) | LLM-judge | **Primary outcome** | Implemented |
| B2 | Cluster coherence (per-cluster) | LLM-judge | Output quality | Implemented |
| B3 | Oracle compliance (request → operation fidelity) | LLM-judge | System behaviour | Implemented |
| B4 | Oracle contradiction (how clear was the oracle) | LLM-judge | Forgiveness context for B1 | Implemented |

## Evaluation runner

```bash
# start the API server first
PYTHONPATH=. python scripts/serve_ui.py

# run all scenarios
PYTHONPATH=. python scripts/run_eval.py --scenarios scenarios/*.json --out reports/<timestamp>/
```

Output in `reports/<timestamp>/`:
- `results.jsonl` — one record per scenario, machine-readable
- `summary.md` — human-readable per-scenario breakdown with aggregate means/medians

**`results.jsonl` record schema:**
```jsonc
{
  "scenario": "sentiment_split",
  "session_id": "uuid...",
  "k_initial": 5, "k_final": 6,
  "A1": {"silhouette_initial": 0.31, "silhouette_final": 0.34, "trend": [...]},
  "A2": {"turns": 7, "weighted_turns": 5.4, "termination": "converged",
         "last_action": "show", "ops_per_turn": [["split"], ["rename"], []]},
  "A3": {"cognitive_load_by_turn": [1,2,1,3,1,2,1], "mean_cognitive_load": 1.6,
         "cognitive_load_driver_by_turn": [...], "final_cognitive_load_breakdown": {...}},
  "B1": {"overall_score": 0.78, "notes": "..."},
  "B2": {"coherence_mean": 0.72, "coherence_min": 0.55, "per_cluster": [...]},
  "B3": {"compliance_score": 0.81, "notes": "..."},
  "B4": {"contradiction_score": 0.15, "notes": "...", "examples": [...]},
  "wall_time_s": 48.2
}
```

## Scenarios (v1 corpus)

Five scenarios in `scenarios/`, each a JSON file with `{name, dataset, k_initial, oracle_turns}`:

| File | Purpose | Expected outcome |
|---|---|---|
| `stable_oracle.json` | Oracle approves initial clustering with minimal changes | Fast `converged`, high B1/B2 |
| `sentiment_split.json` | Oracle splits a mixed cluster by sentiment, renames, accepts | Split + rename ops fire; `converged` |
| `topic_merge.json` | Oracle merges over-fragmented topic clusters | Merge ops fire; `converged` |
| `high_load_oracle.json` | Oracle floods with many conflicting tweaks | Termination = `cognitive_overload` |
| `contradictory_oracle.json` | Oracle splits by sentiment then reverses twice before accepting | Agent robustness (no crash, valid state); B3 unreliable — see B3 caveat above |

All five exercise every termination code. Estimated cost: ~$0.50–$1.00 per full run on a Flash-class model.