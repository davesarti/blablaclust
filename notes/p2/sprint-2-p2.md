# Sprint 2 — P2 (Initial Clustering & Cluster Naming)

## What I built

- **Initial clustering**: `src/engine/initial_clustering.py` — `initial_clustering(data_points, k, session_id, turn_number=1)` takes DataPoints with embeddings, runs k-means on the embedding matrix, and returns `(clusters, soft_assignments)` as DB model objects ready to persist. `k` is parametric (chosen by the user). The function never touches the DB session — the caller owns the transaction, per the `AGENTS.md` rule.
- **Soft assignments**: each data point gets a probability for every cluster, computed as `softmax(-squared_distance_to_centroid)`. Probabilities sum to 1 per point — this is what the `SoftAssignment` table expects and what P3's `f_uncertainty` needs to find boundary points.
- **k tuning with silhouette score**: `silhouette_for_k(data_points, k)` and `sweep_k(data_points, k_min, k_max)` — diagnostics that score how well-separated the clusters are, to help the oracle pick a sensible k before clustering.
- **Cluster naming via LLM**: `src/engine/cluster_naming.py` + `prompts/cluster_naming.txt` — `name_clusters()` sends the 8 most representative points of each cluster to the LLM (via the harness) and gets back a name + description. Best-effort: a failed LLM call or unparseable output leaves the `Cluster N` placeholder instead of aborting.
- **Endpoints** (`backend/routers/clusters.py` — a dedicated P2-owned router, registered in `backend/main.py`):
  - `POST /clusters/{session_id}` — body `{k, generate_names}`. Runs clustering + naming, persists clusters and soft_assignments, returns sizes + silhouette score. Returns 409 if clustering already exists.
  - `GET /clusters/{session_id}/suggest-k` — silhouette sweep, returns a recommended k.
- **requirements.txt**: added `scikit-learn` and `numpy` (dependencies introduced by this layer).

## Results

Verified against the demo DB (1200 real data points, all embedded), k=5:

| Check | Result |
|---|---|
| Clusters created | 5 |
| Soft assignments created | 6000 (1200 × 5) |
| Probabilities sum to 1 per point | yes (within 1e-4) |
| All probabilities in [0, 1] | yes |
| Hard cluster sizes sum to total | 165+234+483+188+130 = 1200 |
| Error paths (k<1, k>n) | raise ValueError |

Note: silhouette scores are low (~0.03–0.04) for this dataset. This is expected for sentence-transformer embeddings of reviews — they don't separate into sharp clusters. The recommended-k mechanism still works in relative terms.

## Challenges

The soft-assignment design needed thought: k-means gives hard labels, but the `SoftAssignment` table wants probabilities. Settled on softmax over negative squared distances to centroids. Also could not test the FastAPI endpoints or the LLM naming call end-to-end locally — `fastapi` and `anthropic` aren't installed in the available env — so verification was done on the pure clustering logic against the demo DB.

## What I need from others

- **P1**: ~~integration mismatch to resolve — `initial_clustering` writes soft_assignments at `turn_number=1`, but creates no `Turn` row.~~ **RESOLVED** — initial clustering now records at `turn_number=0` (the pre-oracle state), consistent with `build_session_state` returning turn 0 when no `Turn` rows exist. Oracle turns keep starting at 1, so there is no collision. The `soft_assignments` turn_number constraint was relaxed from `>= 1` to `>= 0` accordingly (changed in `models.py` with P1's agreement). Commit `b3b72b1`.
- **P1**: aware that `requirements.txt` now includes `scikit-learn` and `numpy`.
- **P4**: cluster naming currently uses the existing harness `call_llm` — no longer blocked. If P4 wants a dedicated naming prompt convention, `prompts/cluster_naming.txt` can be adjusted.

## Follow-up after the sprint

- **Turn-number fix shipped** (commit `b3b72b1`): initial clustering moved to
  `turn_number=0`. Touched `initial_clustering.py` and `clusters.py` (P2) plus a
  one-line constraint change in `models.py` (P1's file, done with permission).
  The empty `soft_assignments` table was recreated with the new constraint; the
  1200 embedded `data_points` were preserved.
- **Cluster operations shipped** (commit `06e032f`): new module
  `src/engine/cluster_operations.py` with the three functions P3's
  `f_apply_operations` needs to execute oracle-requested changes:
  - `merge_clusters` — pools the points of several clusters, dissolves them,
    creates one new cluster.
  - `split_cluster` — runs real k-means (k=2) on a cluster's members to split it.
  - `rename_cluster` — updates name and description only.

  Each operation writes a *complete* soft-assignment snapshot at `turn_number`
  (touched points reassigned, every other point carried forward) so
  `f_uncertainty` and `hard_cluster_stats` keep reading a full picture. The
  functions stage their changes on the `db` session but never commit — the
  caller owns the transaction, so a multi-operation turn stays atomic; the
  target `turn_number` must be strictly greater than the latest snapshot turn.
  11 unit tests in `tests/test_cluster_operations.py`, all passing (in-memory
  SQLite, real k-means, no LLM calls).

## Next steps

- Point-level `reassign` operation (move individual points between clusters) —
  the merge/split/rename trio is done; reassign is the remaining cluster op.
- Coordinate with P3 on wiring `cluster_operations` into `f_apply_operations`,
  and on the `turn_number` convention (one operation per oracle turn, or an
  incrementing turn per operation).
- Add structured logging to the clustering runs via `src/logger.py`.
- Evaluate HDBSCAN as an alternative to k-means (handles noise / variable-density clusters), given the low silhouette scores.
- Confirm with P3 that `f_uncertainty` can now read the populated `soft_assignments` table.
