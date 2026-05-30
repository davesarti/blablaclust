# Evaluation Implementation Plan — v1

*Companion to [`quality_specs.md`](quality_specs.md). The spec freezes **what** we
measure and why; this plan freezes **how** we build the measurement, in what order,
and what a finished v1 looks like.*

## Status: v1 complete ✓ (with post-v1 revisions)

> **Post-v1 updates:** the metric layout was reorganised. **B2 cognitive load
> moved to A3** (it is deterministic engine output, not an LLM judge). **B1 was
> split into four independent metrics**: B1 = the overall synthesis verdict,
> B2 = per-cluster coherence (was internal sub-call), B3 = oracle compliance
> (was internal sub-call, `oracle_clarity` removed), and **a new B4 =
> oracle-contradiction judge** that scores how hard the oracle was to
> understand. B4 feeds B1 as forgiveness context. An earlier "one-point
> validation" B4 was implemented and removed before this restructure — sections
> below that describe that old B4 are historical.

All deliverables below are implemented and end-to-end tested. Run:

```bash
PYTHONPATH=. python scripts/run_eval.py --scenarios scenarios/*.json --out reports/<timestamp>/
```

Inside `reports/<timestamp>/` you'll find `summary.md` (human-readable, with a metric
cheat-sheet and per-scenario breakdown) and `results.jsonl` (one record per scenario,
machine-readable). Five scenarios cover all termination codes and the main operation types.

## Deliverables (with dependency order)

### 1. A2 termination coding — *prerequisite, ~10 lines*

[`f_next_best_step.py`](../src/engine/f_next_best_step.py) currently emits a single
`state_snapshot.reason = "max_turns_or_load_reached"` that conflates two outcomes
the spec wants split. Replace with two codes — `converged` and
`cognitive_overload` — and write `state_snapshot.cognitive_load_driver` alongside
so the runner can bucket overloads by `turns` / `tokens` / `clusters` without
post-hoc string matching. (An earlier revision split this into three codes
including `max_turns_reached`; the A3 redesign folded the turn-count signal
into `cognitive_overload` with a driver, removing the redundant code.) Touches
one file. **Blocks the runner's A2 reporting**; do this first.

### 2. B4 one-point validation — *new function + prompt*

- New: [`src/engine/f_validate_point.py`](../src/engine/f_validate_point.py).
  Signature `f_validate_point(point: DataPoint, cluster: Cluster, state:
  ChatSessionState) -> {endorsed: bool, confidence: float, reasoning: str}`.
  Same logging pattern as `f_eval`.
- New: [`prompts/f_validate_point.txt`](../prompts/f_validate_point.txt).
  Provide point text, cluster name/description/representatives, ask for the three
  fields above.
- **Why before the runner**: the runner needs to *call* it. Easier to wire it once
  the function exists than to stub then refill.

### 3. The runner — *the headline piece, `scripts/run_eval.py`*

Out-of-band. Mirrors the test-harness pattern already proven in
[`scripts/test_e2e.py`](../scripts/test_e2e.py) — drive the live HTTP API rather
than calling engine internals — but the *purpose* is metric collection, not
correctness checks. One scenario → one session → one record.

**Per-scenario flow:**
1. Load scenario JSON (oracle turns, settings).
2. `POST /sessions` → `POST /clusters/{sid}` → cluster the dataset.
3. Iterate the scripted oracle turns through `POST /turns` until the script ends
   or the session reaches `action=stop`.
4. Pull A1 trend from `logs/clustering_runs.jsonl` (entries scoped by `session_id`).
5. Read all turns via `GET /turns?session_id=…`; derive A2 (turn count, weighted by
   `FeedbackEntry.type`), termination code from final turn's `state_snapshot.reason`,
   and B2 trend (`cognitive_load_score` per turn).
6. End-of-session: call `f_eval(state, total_points)` → B1.
7. Sample N (e.g. 5) points per active cluster; for each call `f_validate_point` → B4.
8. Tear the session down (`DELETE /sessions/{sid}/delete`).
9. Write one JSONL record to `results.jsonl`.

**Output record schema (one line of `results.jsonl`):**
```jsonc
{
  "scenario": "sentiment_split.json",
  "session_id": "uuid...",
  "k_initial": 5, "k_final": 6,
  "A1": {"silhouette_initial": 0.31, "silhouette_final": 0.34, "trend": [...]},
  "A2": {"turns": 7, "weighted_turns": 5.4, "termination": "converged"},
  "A3": {"cognitive_load_by_turn": [1,2,1,3,1,2,1], "mean_cognitive_load": 1.6},
  "B1": {"overall_score": 0.78, "notes": "..."},
  "B2": {"coherence_mean": 0.72, "coherence_min": 0.55, "per_cluster": [...]},
  "B3": {"compliance_score": 0.81, "notes": "..."},
  "B4": {"contradiction_score": 0.15, "notes": "...", "examples": [...]},
  "wall_time_s": 48.2
}
```

After all scenarios run, an aggregator emits `summary.md` with means/medians per
metric, broken down by scenario and by `FeedbackEntry.type`.

### 4. Scenarios — *the corpus of v1 runs*

`scenarios/*.json`, each a small object: `{name, dataset, k_initial,
oracle_turns: [{raw_text, feedback_type, target_cluster_ids?, target_point_ids?},
...]}`. Initial v1 corpus:

- **`sentiment_split.json`** — straightforward refinement: oracle asks to split a
  mixed cluster by sentiment, rename, accept. Should converge cleanly.
- **`topic_merge.json`** — oracle asks to merge over-fragmented topic clusters.
- **`high_load_oracle.json`** — oracle floods with many small, conflicting tweaks;
  expects termination=`cognitive_overload`.
- **`stable_oracle.json`** — oracle approves initial clustering with minimal
  changes; expects fast `converged` termination, high B1.

Four scenarios exercise every termination code without exploding LLM cost
(~$0.50–$1.00 per full eval run on Gemini Flash, by the test-harness cost we
observed).

### 5. A2 type-weights — *small constants file*

`src/engine/eval_weights.py`:
```python
FEEDBACK_TYPE_WEIGHTS = {"global": 2.0, "cluster": 1.0, "point": 0.5, "instructional": 0.0}
```
Imported by the runner. Frozen for v1; revisit if a scenario surfaces a weight that
doesn't match intuition.

## Out of scope for v1

- **Human-rater study** — the spec calls for N≈5–10 human raters scoring B1/B4 to
  validate the judge. Out of scope for the implementation work; we can start
  collecting human scores once the runner is producing comparable judge scores.
- **Wiring `f_eval` into the live turn path** — the spec is explicit: *"for v1 the
  eval harness calls the judge B1 out-of-band at end-of-session"*. So we will NOT
  add f_eval to `backend/routers/turns.py`.

## Ordering rationale

1. Termination codes (smallest, blocks the runner).
2. B4 (new function; runner needs it before integration is clean).
3. Runner skeleton with two scenarios + JSONL output (minimum viable evaluation).
4. Remaining scenarios + `summary.md` aggregator.
5. Type-weights (trivial; can land anywhere after the runner exists).

Each step is independently shippable: stop after step 3 and we already have real,
comparable numbers — every later step is incremental refinement.

## Open decisions — resolved

- **B4 sampling size**: top-3 most uncertain points per cluster (lowest soft-assignment
  probability as proxy for highest uncertainty). Makes B4 a stress test of boundaries.
- **Scenario reuse**: fresh sessions always — each row of `results.jsonl` is independent.
- **Failure semantics in the runner**: `B1=null` on judge error, run continues. Confirmed
  working: `f_eval` now uses `extract_json_text` to strip markdown fences before parsing,
  which was the root cause of the original `JSONDecodeError` in first-run testing.

---

## What was actually built (plain-language summary)

*If you haven't been involved in the evaluation work and want to understand what exists now, read this section.*

### The idea in one sentence

We want to be able to run a script and get back a report that tells us: "how well did the agent guide the oracle to a good clustering?" — with specific numbers for each quality dimension we care about (quality of the clusters, how many feedback turns it took, whether the cognitive load was reasonable, etc.).

### What the three main pieces do

**1. Termination codes (tiny change, big impact on measurement)**

The agent's planner (`f_next_best_step.py`) decides when to stop a session. Before this change it would write `"max_turns_or_load_reached"` in both cases where it stops: hitting the maximum number of allowed turns, OR detecting that the cognitive load (a score of how complex the feedback is) got too high. That single string made it impossible to tell the two situations apart in the report.

We split it into two separate codes: `cognitive_overload` and `max_turns_reached`. A healthy session where the oracle is satisfied ends with `converged` (that code was already defined in the spec; the runner treats "ran through all oracle turns without hitting any stop condition" as converged). Now the report can say "3 out of 5 runs converged, 1 hit the turn limit, 1 hit cognitive overload" — which tells you something real about the agent's behaviour.

**2. One-point judge (`f_validate_point.py` + `prompts/f_validate_point.txt`)**

Imagine the oracle has finished guiding the agent and we have a final set of clusters. We want to spot-check: does each data point actually belong in the cluster the agent put it in?

We do this by asking an independent LLM judge (not the same one running the agent) to look at one data point at a time and decide:
- Does this text belong in this cluster? (yes/no = "endorsed")
- How confident are you? (0.0 to 1.0)
- Why? (one or two sentences)

To make the test meaningful, we deliberately pick the *most uncertain* points — the ones the agent's soft-assignment model is least sure about — not random ones. If even those get endorsed at a high rate, the clustering is solid.

The judge is also shown the full oracle feedback history so it can judge "fit" according to what the oracle actually wanted, not just the cluster's name.

This is called the B4 metric in the quality spec.

**3. The evaluation runner (`scripts/run_eval.py`)**

This is the script that ties everything together. You give it one or more scenario files, and it:

1. Reads each scenario (a JSON file describing what the oracle will say, step by step).
2. Starts a fresh session by calling the live API (the same API the front-end uses).
3. Runs the initial clustering.
4. Sends each oracle turn to the agent one by one, just like a real user would. Waits for the agent's response before sending the next one.
5. After all turns, collects all the metrics:
   - **A1**: did the silhouette score (a measure of cluster quality) go up or stay stable?
   - **A2**: how many turns did it take, what kind of feedback was it (global/point/etc.), and did it converge?
   - **A3**: the cognitive load score from each turn — was the agent overwhelmed at any point?
   - **B2**: calls the coherence judge to rate each final cluster 0–1 on internal focus.
   - **B3**: calls the compliance judge to rate how faithfully the system carried out each oracle request.
   - **B4**: calls the contradiction judge on the oracle's feedback history to flag how hard the oracle was to understand.
   - **B1**: calls the synthesis judge with B2 + B3 + B4 as input; emits one overall verdict.
6. Deletes the eval session so the database stays clean.
7. Writes the results: a `results.jsonl` file (one line per scenario, machine-readable) and a `summary.md` (human-readable, with means and medians across all scenarios).

To run it:
```bash
PYTHONPATH=. python scripts/run_eval.py --scenarios scenarios/*.json --out reports/my-run/
```
The server needs to be running first (`PYTHONPATH=. python scripts/serve_ui.py`).

### The scenario files

Each scenario is a small JSON file in `scenarios/`. It says: "use this dataset, start with this many clusters, and here is what the oracle will say, turn by turn." The oracle's lines are scripted — this is not a real human typing, it is a fixed script designed to test a specific behaviour.

Two scenarios exist right now:
- `stable_oracle.json`: the oracle basically approves the initial clustering and makes no structural changes. We expect fast convergence and a high coherence score.
- `sentiment_split.json`: the oracle asks to split a cluster by sentiment, rename the results, then accepts. We expect the split and rename operations to fire correctly.

Two more are planned (`topic_merge`, `high_load_oracle`) to cover merging and cognitive-overload termination respectively.

### What you get at the end

After a run you will have a folder like `reports/20260527-143201/` containing:
- `results.jsonl` — every scenario's numbers in one line each (good for scripting or importing into a spreadsheet).
- `summary.md` — a readable report with one section per scenario and an aggregate table at the top showing mean coherence, mean endorsement rate, how many runs converged vs. hit limits, etc.

The numbers are comparable across runs: if you change something in the engine and re-run the eval, you can diff the two `summary.md` files and see whether quality went up or down.