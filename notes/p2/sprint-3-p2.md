# Sprint 3 — P2 (Data & Embeddings)

## What I did this sprint

Hardened the clustering layer, added a second dataset, and covered P5's UI slice
while she was on evaluation.

**Clustering & naming**
- **One LLM call for naming** (`cluster_naming.py`) — `name_clusters` built one
  prompt per cluster (k round-trips, inconsistent labels across clusters). Now a
  single prompt describes every cluster (`[Cluster id: ...]` blocks) and makes
  **one call** returning `{id → {name, description}}`. Same best-effort contract,
  same public signature (callers unchanged).
- **Configurable split arity** (`cluster_operations.py`) — added `k` to
  `split_cluster` (was hardcoded k=2); rejects `k<2` up front, requires ≥k points,
  passes k to `initial_clustering`. Default 2 keeps existing callers unchanged.
- **First-class naming in merge/split** — added `auto_name=True` to
  `merge_clusters`/`split_cluster` so they name the clusters they create from the
  pooled/subset points. Filled the real gap on merge (never named before) and let
  P3 drop the duplicate `name_clusters` call.
- **Structured logging of clustering runs** — `log_clustering_run` (one JSONL line
  per run: `seed`, `k`, `backend`, `silhouette`, `n_points`), called from
  `initial_clustering`, best-effort. A `conftest.py` autouse fixture redirects the
  log to a tmp file so the suite never pollutes the real log. (P5 later absorbed
  the helper into `src/logger.py`.)
- **Critical fix — `merge_clusters` collapsed the whole dataset** — live testing
  showed a merge assigning all 1200 points to the new cluster. Root cause: the
  carry-forward summed the merged clusters' probabilities onto the new cluster, and
  in flat high-dim softmax (~0.20 each at k=5) that sum beat every other cluster's
  prob → new cluster became the argmax for every point. Fix: drop the merged mass
  on carry-forward (pooled points get 1.0, un-pooled keep their argmax). Regression
  test reproduces the 5-cluster case; verified live (295 points, not 1199).
- **Silhouette computed once, not twice** — `POST /clusters` fit k-means twice
  (once for the log, once via `silhouette_for_k` for the response). Now
  `initial_clustering` returns the silhouette it already computed; router reads it
  directly. One run per clustering, single source of truth.

**Second dataset — 20 Newsgroups (adaptivity proof)**
- `sample_20newsgroups.py` + CSVs. Curated 6 well-separated newsgroups
  (graphics, autos, baseball, med, space, guns), headers/footers/quotes stripped,
  1200 train + 300 frozen. `title` left empty so the category never leaks into the
  embedding. Verified live: k=6 recovers all six topics with topic-appropriate
  names and **87.9% purity** vs ground truth. Amazon untouched; datasets isolated
  by `dataset_name`.

**Fixes & hygiene**
- Fixed broken `text_cleaning` imports in `generate_embeddings.py` /
  `verify_database.py` (crashed from project root).
- Renamed misleading `download_dataset.py` → `sample_amazon_dataset.py` and
  wrapped its top-level side effects in `main()` (importing it used to overwrite
  the CSVs on disk).
- Deduplicated the session lookup in `clusters.py` (4 lookups → one
  `_get_session_or_404`).
- **Test coverage for the data layer** — `test_text_cleaning.py` (25 tests) +
  `test_dataset_load_utils.py` (18 tests, in-memory SQLite, covers rollback).
  Validated by **mutation testing**: 6/6 deliberate mutations caught.

**UI (covering P5's slice, `ui/index.html`)** — dynamic dataset label from
`GET /datasets` (was hardcoded "1,500 reviews"), Inter font + typography refresh.
Deliberately skipped the palette/accent rework (out of chosen scope).

## What blocked me

- **`tiktoken` missing in the local env** — listed in `requirements.txt` but not
  installed; `harness_openrouter.py` imports it at module level, so every LLM call
  (including naming) silently failed with `ModuleNotFoundError`. Naming *appeared*
  to run but produced no names. Fixed with `pip install tiktoken`.
- **Cross-team dependencies** — making variable-arity split reachable end-to-end
  needed edits to P4's `f_output.txt` (split-op `k` field) and P3's
  `f_apply_operations.py` (pass `k` through); done with the team's consent. Using
  cluster names to *drive* merge/split decisions was P3/P4's slice (commits
  `b334364`, `64fd820`).

## What I'm doing next

- The generalization mapping function (#52) — codify a finished clustering into a
  reusable assignment for held-out items.
- Coordinate with P3 on the `f_uncertainty` / `f_next_state` dead-code findings
  from the audit (noted, not yet filed).
- Follow up that P5 absorbs `clustering_log` into `src/logger.py` (issue filed).

## Commits

- `5e0f300` — refactor: name all clusters in a single LLM call
- `2fdb8f6` — feat: add `k` parameter to `split_cluster`
- `af10bdd` — feat: auto-name clusters created by merge and split
- `6bc7cfd` — feat: structured logging of every clustering run
- `7f1002f` — docs: issue for P5 to absorb `clustering_log` into `src/logger.py`
- `6612273` — feat: wire `k` parameter through the split op end-to-end
- `ea627e7` — fix: correct `text_cleaning` imports in dataset_processing scripts
- `22a5b0e` — fix: drop merged mass in `merge_clusters` to preserve un-pooled argmax
- `862a691` — refactor: rename `download_dataset.py` → `sample_amazon_dataset.py` + `main()` guard

## Issues

- **Opened & resolved by me:** #30 (guard `silhouette_score` against pathological
  inputs), #31 (silhouette computed twice in `POST /clusters`), #32 (use
  `_get_session_or_404` consistently), #33 (unit tests for `text_cleaning` +
  `dataset_load_utils`), #34 (`download_dataset.py` side-effect + rename), #37 (UI
  improvements), #49 (verify the loop on a non-Amazon dataset), #52 (generalization
  mapping function).
- **Resolved by me (opened by a teammate):** #20 (centralize cluster naming — P1),
  #21 (naming conventions on split/merge — P1), #22 (surface silent/cached failures
  — P1), #23 (broken imports in dataset scripts — P1), #25 (`split_cluster` always
  splits into 2 — P1), #36 (naming prompt in merge/split — P1), #40 (`merge_clusters`
  folds merged mass — P3).
- **Opened by me, assigned to a teammate:** #26 (cluster naming drives split/merge
  — P3/P4), #29 (move `log_clustering_run` into `src/logger.py` — P5), #42 (`f_eval`
  crashes on Gemini — P3), #43 (descriptions wiped on rename/merge/split — P3), #48
  (make prompts dataset-agnostic — P4), #50 (eval scenarios for 20NG — P5), #51
  (expose per-dataset record counts — P1).
- **Opened by me, resolved in sprint 4:** #47 (wire token/cost/cognitive load to
  the UI).
