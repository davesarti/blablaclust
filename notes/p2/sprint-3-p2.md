# Sprint 3 — P2 (Data & Embeddings)

## What I built

Sprint 3 for P2 covered three issues against the clustering code plus one
sprint-omnibus task. All work landed entirely within P2-owned files — no
cross-team edits.

### 1. Centralised cluster naming — one LLM call instead of N

`src/engine/cluster_naming.py` + `prompts/cluster_naming.txt`.

`name_clusters` previously issued **one LLM call per cluster**: with k clusters
that meant k separate prompts, each unaware of the others. This was slow and
let names drift inconsistent across clusters (one cluster called "Shipping
problems", another "Delivery issues" for the same kind of theme).

It now builds a **single prompt** describing every cluster — each block tagged
`[Cluster id: ...]` with its representative reviews — and makes **one call**.
The response is a JSON object keyed by cluster id (`{ id → {name, description} }`),
which the LLM produces with all clusters in view at once, so labels come out
more consistent and latency drops from k round-trips to one.

Naming stays **best-effort**, same contract as before:
- a failed/unparseable call → all clusters keep their `Cluster N` placeholder;
- a partial or malformed response (missing id, non-dict entry) → only those
  clusters keep the placeholder, the rest are named.

The public signature of `name_clusters` is unchanged, so the two existing
callers (`backend/routers/clusters.py` and P3's `f_apply_operations.py`)
needed no modification.

### 2. Configurable split arity — `k` parameter on `split_cluster`

`src/engine/cluster_operations.py`.

`split_cluster` hardcoded `k=2` when calling `initial_clustering`, and had no
`k` parameter at all — so callers could never split a cluster into more than
two sub-clusters, even though `initial_clustering` already supports arbitrary k.

Added `k: int = 2` to the signature and wired it through. The function now:
- rejects `k < 2` up front with a clear `ValueError` (before any DB query);
- requires the cluster to hold at least `k` points (the old "need at least 2"
  check is now "need at least k");
- passes `k` straight to `initial_clustering`.

The default of 2 keeps every existing caller behaving exactly as before.

### 3. Naming as a first-class step in merge and split

`src/engine/cluster_operations.py`.

Naming was previously bolted on by the caller *after* a cluster was created
(P3 explicitly called `name_clusters` in `f_apply_operations` after each
split, and a merged cluster was never named at all). The "first-class naming"
issue asked to push naming into the operations themselves.

Added `auto_name: bool = True` to both `merge_clusters` and `split_cluster`.
When True (the default), the operation calls `name_clusters` internally on
the cluster(s) it creates, using the pooled points (for merge) or the
k-means subset (for split) as representative examples. Best-effort: a failed
LLM call keeps the generic placeholder (`"Merge of A + B"` /
`"<parent> - part N"`), the operation never aborts on naming.

This filled the real gap on merge (no naming before) and let P3 remove
the duplicate `name_clusters` call from `f_apply_operations`.

The issue's other half — using cluster names to *drive* split/merge
*decisions* — lives in P4's `prompts/f_output.txt` and P3's orchestration,
outside my files. P3 and P4 picked up their slices (commits `b334364` and
`64fd820`); the prompt now instructs the model to use cluster name +
description as signals (e.g. only merge when "redundant across ALL meaningful
dimensions"). With those three slices the issue is closed across the team.

### 4. Structured logging of every clustering run (Sprint 3 task)

`src/engine/clustering_log.py` (new) + `src/engine/initial_clustering.py`
+ `tests/conftest.py` (new).

Sprint 3 asked for one JSONL line per clustering run with `seed`, `k`,
`backend`, `silhouette`, `n_points`. New helper `log_clustering_run` (mirrors
P5's `log_llm_call` shape) appends to `logs/clustering_runs.jsonl`;
`initial_clustering` calls it at the end, computing silhouette on the spot
from `model.labels_` (None when `k < 2` or `k >= n_points`). Logging is
best-effort — an `OSError` is swallowed so a broken log file can never abort
a clustering run.

Covers both initial clustering and the runs triggered internally by
`split_cluster` (which goes through `initial_clustering`). Diagnostic
functions (`silhouette_for_k`, `sweep_k`) stay silent — they're for
k-selection, not real runs.

The helper lives in `src/engine/clustering_log.py` rather than `src/logger.py`
because the central logger is P5's file. The shape is intentionally identical
to `log_llm_call` so P5 can fold it in trivially — follow-up issue dropped
in `notes/p2/issue-for-p5-logger.md`.

A new `tests/conftest.py` autouse fixture redirects the log path to a
per-test tmp file, so the suite never pollutes `logs/clustering_runs.jsonl`.

## Results

| Check | Result |
|---|---|
| `test_cluster_naming.py` (12 tests) | all pass |
| `test_cluster_operations.py` (25 tests: 18 pre-existing + 7 new) | all pass |
| `test_clustering_log.py` (7 tests, new) | all pass |
| `test_f_apply_operations.py` (P3, uses changed P2 functions) | still passes |
| Single LLM call regardless of cluster count | verified (mock asserts 1 call) |
| `split_cluster(k=3)` on a 6-point cluster | 3 children, parent dissolved |
| `split_cluster(k=1)` / `k` exceeding point count | raise `ValueError` |
| `merge_clusters` / `split_cluster` self-name children | verified (mock asserts call) |
| `initial_clustering` writes all 5 required log fields | verified |
| Test suite leaves real `logs/clustering_runs.jsonl` untouched | verified |

Full P2-relevant suite: 93 tests pass. 7 tests in `test_f_next_state.py` fail
in the local env with `ModuleNotFoundError: tiktoken` (imported by
`harness_openrouter.py`) — pre-existing environment gap, unrelated to these
changes.

## What I changed in other people's files

Nothing. Every piece of work was solvable inside P2-owned files. Where the
issue genuinely required cross-file work (naming-driven split/merge
*decisions* in P3/P4 territory, or moving the new logger into `src/logger.py`),
I flagged it as a follow-up rather than touching another P's file.

## What I need from others

- **P5** — absorb `src/engine/clustering_log.py` into `src/logger.py` and
  update the one import in `initial_clustering.py`. Full instructions in
  `notes/p2/issue-for-p5-logger.md`.
- **P3** — `f_apply_operations` calls `split_cluster` without passing `k`,
  so it currently always splits in two. The *capability* for k > 2 now
  exists in P2; wiring it to an oracle request like "split into three" is
  a separate change in P3's file. Heads-up if end-to-end variable-arity
  split is wanted for the demo.
- **P4** — `prompts/cluster_naming.txt` changed shape this sprint (now
  takes `{clusters_block}` and returns an id-keyed JSON object). If P4
  keeps a prompt-versioning registry, the hash for this prompt needs
  refreshing.
- **Cosmetic, P3/P4** — `f_output.txt` shows merge ops can include a
  `"new_name"` field, but `f_apply_operations` doesn't pass it to
  `merge_clusters` (which now auto-names from pooled content anyway).
  Either drop the field from the prompt or honour it — currently the LLM
  produces a name that is silently discarded.

## Commits

- `5e0f300` — refactor: name all clusters in a single LLM call
- `2fdb8f6` — feat: add k parameter to split_cluster
- `af10bdd` — feat: auto-name clusters created by merge and split
- `7f1002f` — docs: issue for P5 to absorb clustering_log into src/logger.py
- `6bc7cfd` — structured logging of every clustering run

All pushed to `origin/main`.
