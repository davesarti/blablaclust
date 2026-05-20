# Sprint 2 — P2 (Initial Clustering & Cluster Naming)

## What I built

- **Initial clustering**: `src/engine/initial_clustering.py` — `initial_clustering(data_points, k, session_id, turn_number=1)` takes DataPoints with embeddings, runs k-means on the embedding matrix, and returns `(clusters, soft_assignments)` as DB model objects ready to persist. `k` is parametric (chosen by the user). The function never touches the DB session — the caller owns the transaction, per the `AGENTS.md` rule.
- **Soft assignments**: each data point gets a probability for every cluster, computed as `softmax(-squared_distance_to_centroid)`. Probabilities sum to 1 per point — this is what the `SoftAssignment` table expects and what P3's `f_uncertainty` needs to find boundary points.
- **k tuning with silhouette score**: `silhouette_for_k(data_points, k)` and `sweep_k(data_points, k_min, k_max)` — diagnostics that score how well-separated the clusters are, to help the oracle pick a sensible k before clustering.
- **Cluster naming via LLM**: `src/engine/cluster_naming.py` + `prompts/cluster_naming.txt` — `name_clusters()` sends the 8 most representative points of each cluster to the LLM (via the harness) and gets back a name + description. Best-effort: a failed LLM call or unparseable output leaves the `Cluster N` placeholder instead of aborting.
- **Endpoints** (`backend/routers/sessions.py`):
  - `POST /sessions/{id}/clustering` — body `{k, generate_names}`. Runs clustering + naming, persists clusters and soft_assignments, returns sizes + silhouette score. Returns 409 if clustering already exists.
  - `GET /sessions/{id}/clustering/suggest-k` — silhouette sweep, returns a recommended k.
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

- **P1**: integration mismatch to resolve — `initial_clustering` writes soft_assignments at `turn_number=1`, but creates no `Turn` row. `read_session_state` derives the current turn from `max(Turn.turn_number)`, so it would see turn 0 and not display the freshly created clusters. Need to agree: either the initial clustering creates a `Turn` 1, or `read_session_state` handles "clustering without turns".
- **P1**: aware that `requirements.txt` now includes `scikit-learn` and `numpy`.
- **P4**: cluster naming currently uses the existing harness `call_llm` — no longer blocked. If P4 wants a dedicated naming prompt convention, `prompts/cluster_naming.txt` can be adjusted.

## Next steps

- Coordinate with P1 on the `turn_number` / `Turn` row issue.
- Evaluate HDBSCAN as an alternative to k-means (handles noise / variable-density clusters), given the low silhouette scores.
- Confirm with P3 that `f_uncertainty` can now read the populated `soft_assignments` table.
