# Sprint 3 — P2 (Data & Embeddings)

## What I built

Sprint 3 for P2 covered three issues against the clustering code plus one
sprint-omnibus task, plus several additional fixes applied after the initial
sprint: end-to-end wiring of variable-arity split, broken import fixes, a
missing `tiktoken` dependency resolved, a critical correctness bug in
`merge_clusters` (every merge was pulling the entire dataset into the new
cluster), a top-level side-effect hazard in the dataset sampling script, a
cross-team audit that produced two issues for P3, fresh test coverage for the
dataset-processing layer (validated by mutation testing), and a small
deduplication chore in the clusters router.

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

The helper originally lived in `src/engine/clustering_log.py` (separate file
because the central logger is P5-owned). A follow-up issue
(`notes/p2/issue-for-p5-logger.md`) asked P5 to absorb it into
`src/logger.py`; **P5 has now done this**: `clustering_log.py` is gone,
`log_clustering_run` lives in `src/logger.py`, and `initial_clustering`
imports from there. Tests and `conftest.py` were updated accordingly.

A new `tests/conftest.py` autouse fixture redirects the log path to a
per-test tmp file, so the suite never pollutes `logs/clustering_runs.jsonl`.

### 5. End-to-end wiring of variable-arity split

`prompts/f_output.txt` + `src/engine/f_apply_operations.py` + `tests/test_f_apply_operations.py`.

The `k` parameter added to `split_cluster` in item 2 was never reachable end-to-end:
the LLM prompt had no `k` field in the split operation schema, and
`f_apply_operations` didn't pass it through even if the LLM had produced one.

Added `"k": 2` to the split operation schema in `f_output.txt` with a
constraint explaining when to use k > 2 ("only when the oracle explicitly
asks to split into a specific number of groups"). Updated `f_apply_operations`
to read `k=int(op.get("k", 2))` and pass it to `split_cluster`. Updated the
affected mock assertions in `test_f_apply_operations.py` and added a new test
`test_split_passes_k_to_split_cluster` covering the k > 2 path.

This touched P3/P4 files (`f_output.txt`, `f_apply_operations.py`) with the
team's implicit consent — the capability existed in P2 but was unreachable.

### 6. Fix broken imports in dataset_processing scripts

`src/dataset_processing/generate_embeddings.py` + `src/dataset_processing/verify_database.py`.

Both scripts imported `text_cleaning` as `from dataset_processing.text_cleaning import clean_text`,
which only resolved when `PYTHONPATH` included `src/`. Running them from the
project root crashed with `ModuleNotFoundError`. Fixed to
`from src.dataset_processing.text_cleaning import clean_text`, consistent with
`dataset_load_utils.py`. Verified both scripts import cleanly with
`PYTHONPATH=. python -c "import src.dataset_processing.<script>"`.

### 7. Install missing `tiktoken` dependency

`tiktoken>=0.7.0` was listed in `requirements.txt` but not installed in the
local conda environment. `harness_openrouter.py` imports it at module level,
so every LLM call (including `name_clusters`) silently failed with
`ModuleNotFoundError: No module named 'tiktoken'` — cluster naming appeared
to run but produced no names. Fixed with `pip install tiktoken`.

### 8. Fix merge_clusters pulling the entire dataset into the new cluster

`src/engine/cluster_operations.py` + `tests/test_cluster_operations.py`.

Live UI testing surfaced a critical correctness bug: merging two clusters
ended up assigning **all** points (1200/1200) to the new cluster, with the
other un-merged clusters dropping to 0 size. Root cause in the carry-forward
step of `merge_clusters`: for un-pooled points the code summed the
probabilities of the merged clusters and folded that mass onto the new
cluster (`merged_mass = sum(prob[cid] for cid in merge_set)`). In high-dim
sentence-transformer space the softmax over k=5 clusters is very flat
(~0.20 each), so the sum of two merged probs routinely exceeded any
un-merged cluster's prob — and the new cluster became the argmax for every
point in the dataset.

Fix: drop the merged mass entirely on carry-forward. Pooled points still
get 1.0 on the new cluster, un-pooled points keep their original argmax.
Probabilities for un-pooled points no longer sum to 1, but the snapshot is
internally consistent and the hard partition (what the UI reads) is
preserved. Added regression test
`test_merge_preserves_argmax_on_flat_soft_assignments` that reproduces the
5-cluster scenario and fails under the old fold. Verified end-to-end on the
exact live repro (sizes 483/234/188/165/130, merge of c4+c5) — new cluster
ends up with 295 points (= 165+130), others unchanged.

Bonus collateral: with the fix, `merged_point_ids` (passed to
`name_clusters`) now contains only the actually pooled points instead of
every point in the dataset. So the LLM gets clean representative texts and
produces sensible names like "Shipping Problems" instead of generic labels
derived from the whole dataset.

### 9. Sample-dataset script: guard side effects + rename

`src/dataset_processing/download_dataset.py` →
`src/dataset_processing/sample_amazon_dataset.py` (committed in `862a691`).

The script had two problems:
1. All sampling / CSV writes happened at module top-level — importing it
   would immediately read `data/amazon_review_polarity_csv/train.csv` and
   overwrite `data/train.csv` + `data/frozen_eval.csv`.
2. The name "download_dataset" was misleading: nothing is downloaded, it
   only samples a CSV already on disk.

Wrapped the body in `def main()` + `if __name__ == "__main__": main()`,
extracted magic numbers (`SAMPLE_SIZE`, `TRAIN_SIZE`, `RANDOM_STATE`,
`MIN_TEXT_LEN`) as module constants, and renamed the file. Verified with
two tests: (1) importing the module does NOT change the mtime of
`data/train.csv` or `data/frozen_eval.csv`; (2) running it as
`python -m src.dataset_processing.sample_amazon_dataset` correctly
produces 1200 train + 300 frozen rows. No callers anywhere in the repo, so
the rename is safe.

### 10. Cross-team audit — issues opened for P3

Read-only audit of P3-owned engine files (`f_apply_operations`, `f_eval`,
`f_next_state`, `f_next_best_step`, `f_uncertainty`, `f_output`,
`f_parse_clustering_intent`) looking for bugs that interact with P2 code or
block end-to-end correctness. Produced two GitHub issues:

- **#42 — `f_eval.py` crashes on Gemini.** `json.loads(msg.text)` skips
  the `extract_json_text` helper that every other `f_*` uses to strip
  markdown fences. Gemini (our current OpenRouter default) wraps JSON in
  fences → JSONDecodeError. One-line fix.
- **#43 — Cluster descriptions wiped or never set on rename / merge / split.**
  Rename always passes `new_description=""` because the prompt never asks
  for one; merged clusters start with `description=""` and depend entirely
  on the LLM call succeeding; split children skip naming when the oracle
  provides `new_names`, so they keep an empty description forever. Three
  bugs, separate fixes proposed for each.

Also noted but not filed: `f_uncertainty` is computed every turn but its
result is no longer read by `f_next_best_step` (the "ask" rule was
removed), and `f_next_state.py` is now dead code (never called by
`turns.py`, would double the LLM call if anyone used it). Logged in this
note as candidates for a follow-up issue if P3 confirms.

### 11. Test coverage for the dataset-processing layer

`tests/test_text_cleaning.py` (new) + `tests/test_dataset_load_utils.py` (new).

Two P2 files had zero test coverage — exactly the silent-regression risk the
cleaning + ingest pipeline is prone to. Added:

- **`test_text_cleaning.py`** (25 tests) — every private helper in isolation
  (`_normalize_unicode`, `_fix_double_quotes`, `_collapse_repeated_chars`,
  `_collapse_whitespace`) plus `clean_fields` / `clean_text` end-to-end on
  representative dirty strings. Edge cases: empty input, decomposed unicode
  (NFC), CSV-escaped quotes, long runs of the same character, whitespace
  across the title+text join.
- **`test_dataset_load_utils.py`** (18 tests) — runs against an in-memory
  SQLite DB (StaticPool, same pattern as `test_cluster_operations.py`).
  Covers header validation (missing → `ValueError`, extra columns OK),
  empty-text skip, non-int / missing label skip, field cleaning on insert,
  and **transaction rollback** in `process_csv_upload` (both on an embedding
  failure via monkeypatch and on bad headers — nothing persists).

Validated with **mutation testing**: temporarily broke each behaviour in the
source (collapse threshold, quote fix, whitespace strip, rollback, empty-text
skip, header raise) and confirmed at least one test fails for every mutation.
The tests have teeth, not just green checkmarks.

### 12. Deduplicate session lookup in clusters.py

`backend/routers/clusters.py`.

`run_initial_clustering` and `suggest_k` each reimplemented the
`_get_session_or_404` helper inline (3 identical lines apiece). Replaced both
with a call to the existing helper, so all four session lookups in the file
now go through one code path. Net −4 lines, zero behaviour change — verified
the app still loads via `backend.main` and the 8 `test_turns_endpoint.py`
tests pass.

## Results

| Check | Result |
|---|---|
| `test_cluster_naming.py` (12 tests) | all pass |
| `test_cluster_operations.py` (25 tests: 18 pre-existing + 7 new) | all pass |
| `test_clustering_log.py` (7 tests, new) | all pass |
| `test_f_apply_operations.py` (P3, uses changed P2 functions) | all pass (39 tests) |
| Single LLM call regardless of cluster count | verified (mock asserts 1 call) |
| `split_cluster(k=3)` on a 6-point cluster | 3 children, parent dissolved |
| `split_cluster(k=1)` / `k` exceeding point count | raise `ValueError` |
| `merge_clusters` / `split_cluster` self-name children | verified (mock asserts call) |
| `initial_clustering` writes all 5 required log fields | verified |
| Test suite leaves real `logs/clustering_runs.jsonl` untouched | verified |
| `split_cluster(k=4)` wired end-to-end from LLM prompt to k-means | verified |
| `generate_embeddings.py` / `verify_database.py` import from project root | verified |
| `tiktoken` installed — `name_clusters` LLM calls succeed | verified |
| `merge_clusters` no longer collapses dataset into one cluster | verified live (5-cluster repro: 295/470/245/190 vs old 1199/1/0/0/0) |
| New regression test `test_merge_preserves_argmax_on_flat_soft_assignments` | passes; fails on old code |
| Live merge naming with Gemini after fix | produces "Shipping Problems" + description |
| `sample_amazon_dataset.py` import does NOT touch CSVs on disk | verified (mtime unchanged) |
| `sample_amazon_dataset.py` run as `__main__` produces 1200/300 split | verified |
| `test_text_cleaning.py` (25 tests, new) | all pass |
| `test_dataset_load_utils.py` (18 tests, new) | all pass |
| Mutation testing on text_cleaning + dataset_load_utils | 6/6 mutations caught |
| `clusters.py` session-lookup dedup — app loads + endpoint tests | verified (8 tests pass) |

Full P2-relevant suite: **110 tests pass** in the vibe-coders env across
`test_text_cleaning`, `test_dataset_load_utils`, `test_cluster_operations`,
`test_cluster_naming`, `test_clustering_log`, `test_initial_clustering`,
`test_f_parse_clustering_intent`, and `test_turns_endpoint`.

## What I changed in other people's files

- **`prompts/f_output.txt`** (P4) — added `"k": 2` to the split operation
  schema and a constraint explaining when to set k > 2. Necessary to make
  variable-arity split reachable end-to-end (P2's `split_cluster(k)` was
  unreachable without this).
- **`src/engine/f_apply_operations.py`** (P3) — wired `k=int(op.get("k", 2))`
  into the `split_cluster` call. Same reason.
- **`tests/test_f_apply_operations.py`** (P3) — updated two existing mock
  assertions to include `k=2` and added `test_split_passes_k_to_split_cluster`.

## What I need from others

- **P5** — ~~absorb `src/engine/clustering_log.py` into `src/logger.py`~~
  **DONE.** `clustering_log.py` no longer exists; `log_clustering_run`
  lives in `src/logger.py`; `initial_clustering` imports from there.
- **P4** — `prompts/cluster_naming.txt` changed shape this sprint (now
  takes `{clusters_block}` and returns an id-keyed JSON object). If P4
  keeps a prompt-versioning registry, the hash for this prompt needs
  refreshing.
- **P3 (open issues)** — #42 (`f_eval` Gemini crash, one-line fix) and
  #43 (cluster descriptions wiped on rename / never set on merge & split
  with inline names). Both have fixes proposed in the issue body.
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
- `6612273` — feat: wire k parameter through split op end-to-end
- `ea627e7` — fix: correct text_cleaning imports in dataset_processing scripts
- `22a5b0e` — fix: drop merged mass in merge_clusters to preserve un-pooled argmax
- `862a691` — rename download_dataset.py → sample_amazon_dataset.py with main() guard

All pushed to `origin/main`.

**Pending commit** (items 11–12 above, verified, not yet pushed):
`tests/test_text_cleaning.py`, `tests/test_dataset_load_utils.py`, and the
`clusters.py` session-lookup dedup.

### GitHub issues opened during this sprint

- **#42** — `f_eval.py crashes on Gemini` (P3)
- **#43** — `Cluster descriptions get wiped or never set on rename / merge / split` (P3, with optional slice for P4)
- UI feedback for P5 (home button, loading indicator, dynamic dataset label,
  general restyle) — drafted in chat, not yet filed as a GitHub issue.
