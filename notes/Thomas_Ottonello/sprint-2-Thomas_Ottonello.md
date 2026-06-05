# Sprint 2 — P2 (Initial Clustering & Cluster Naming) - Thomas Ottonello

## What I did this sprint

Built the initial-clustering layer and the cluster operations the engine needs.

- **Initial clustering** (`src/engine/initial_clustering.py`) —
  `initial_clustering(data_points, k, session_id, turn_number)` runs k-means on
  the embedding matrix and returns `(clusters, soft_assignments)` as DB objects
  ready to persist. Never touches the DB session — the caller owns the
  transaction (per `AGENTS.md`).
- **Soft assignments** — each point gets a probability per cluster via
  `softmax(-squared_distance_to_centroid)`, summing to 1 per point (what
  `SoftAssignment` and P3's `f_uncertainty` expect).
- **k tuning** — `silhouette_for_k` and `sweep_k` score cluster separation to
  help the oracle pick a sensible k.
- **Cluster naming** (`src/engine/cluster_naming.py` + `prompts/cluster_naming.txt`)
  — `name_clusters()` sends each cluster's representative points to the LLM and
  gets back name + description. Best-effort: a failed/unparseable call keeps the
  `Cluster N` placeholder rather than aborting.
- **Endpoints** (`backend/routers/clusters.py`, a P2-owned router):
  `POST /clusters/{session_id}` (cluster + name + persist, 409 if it already
  exists) and `GET /clusters/{session_id}/suggest-k` (silhouette sweep).
- **requirements.txt** — added `scikit-learn` and `numpy`.

Verified on the demo DB (1200 embedded points, k=5): 6000 soft assignments,
probabilities sum to 1 (±1e-4) and lie in [0,1], hard sizes sum to 1200, error
paths (k<1, k>n) raise `ValueError`. Silhouette is low (~0.03–0.04) — expected
for sentence-transformer review embeddings; the relative recommended-k still works.

**Cluster operations** (commit `06e032f`) — new `src/engine/cluster_operations.py`
with the three functions P3's `f_apply_operations` needs: `merge_clusters`,
`split_cluster` (real k-means, k=2), `rename_cluster`. Each writes a *complete*
soft-assignment snapshot at `turn_number` (touched points reassigned, all others
carried forward) and stages on `db` without committing, so a multi-op turn stays
atomic. 11 unit tests, all passing (in-memory SQLite, real k-means, no LLM).

## What blocked me

- **FastAPI / `anthropic` not installed locally** — couldn't test the endpoints
  or the LLM naming call end-to-end; verification was on the pure clustering
  logic against the demo DB.
- **Turn-number convention with P1** — initial clustering originally wrote soft
  assignments at `turn_number=1` without a `Turn` row. Resolved with P1: it now
  records at `turn_number=0` (the pre-oracle state); the `soft_assignments`
  constraint was relaxed `>= 1` → `>= 0` in `models.py` (P1's file, with
  permission). Commit `b3b72b1`.

## What I'm doing next

- Point-level `reassign` operation (move points between clusters) — the
  merge/split/rename trio is done; reassign is the remaining op.
- Coordinate with P3 on wiring `cluster_operations` into `f_apply_operations`
  and on the turn-number convention (one op per turn vs. per op).
- Add structured logging of clustering runs via `src/logger.py`.
- Evaluate HDBSCAN as an alternative to k-means given the low silhouette scores.

## Commits

- `b3b72b1` — fix: initial clustering records at `turn_number=0`; relax
  `soft_assignments` constraint to `>= 0` (with P1).
- `06e032f` — feat: `cluster_operations.py` (merge / split / rename) + 11 tests.

## Issues

- **Opened & resolved by me:** #7 (initial clustering + soft assignments), #8
  (make clustering respond to oracle feedback — the cluster-operations work).
- **Resolved by me (opened by a teammate):** #14 (implement
  `merge_clusters` / `split_cluster` / `rename_cluster` — opened by P3).
