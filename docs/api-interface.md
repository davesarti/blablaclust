# API Interface

HTTP/JSON interface served by the FastAPI app in
[`backend/main.py`](../backend/main.py). Routers live under
[`backend/routers/`](../backend/routers/); request/response shapes that are
shared across endpoints are defined as Pydantic models in
[`src/schemas.py`](../src/schemas.py).

The app is plain FastAPI — no auth, no API key. Run it with `uvicorn
backend.main:app`; the interactive OpenAPI schema is available at `/docs` and
`/redoc` when the server is running.

## Conventions

- All payloads are JSON unless explicitly noted (`POST /datasets/upload` is the
  only multipart endpoint).
- IDs are UUID strings (`str`).
- A "turn number" is a non-negative integer; turn 0 is the initial k-means
  snapshot written by `POST /clusters/{session_id}`, and each oracle exchange
  via `POST /turns` advances it by 1.
- Errors follow FastAPI's standard `{"detail": "..."}` body. Status codes used:
  `400` (validation at the API boundary), `404` (entity missing), `409`
  (state conflict — e.g. clustering already exists, session already closed),
  `422` (semantic/engine validation), `500` (commit failure), `502` (engine /
  LLM call failure).

## Routers

| Prefix | Tag | File |
|---|---|---|
| `/datasets` | `datasets` | [`routers/datasets.py`](../backend/routers/datasets.py) |
| `/sessions` | `sessions`, `umap` | [`routers/sessions.py`](../backend/routers/sessions.py), [`routers/umap.py`](../backend/routers/umap.py) |
| `/clusters` | `clusters` | [`routers/clusters.py`](../backend/routers/clusters.py) |
| `/turns`    | `turns` | [`routers/turns.py`](../backend/routers/turns.py) |

---

## Datasets

A dataset is a named collection of `DataPoint` rows plus their embeddings.
Datasets whose name ends in `_frozen` are held-out evaluation splits and are
hidden from `GET /datasets`.

### `GET /datasets`

List selectable datasets with point/embedding counts. Excludes any dataset
whose name ends in `_frozen`.

**Response** — `200 OK`, array of:

| Field | Type | Notes |
|---|---|---|
| `dataset_id` | `str` | UUID. |
| `dataset_name` | `str` | |
| `n_points` | `int` | Total `DataPoint` rows. |
| `has_embeddings` | `int` | Count of rows with a non-null embedding. |
| `description` | `str` | LLM-generated or user-supplied; `""` when absent. |

### `POST /datasets/upload`

Multipart upload of a CSV. Inserts new `Dataset` + `DataPoint` rows and
optionally generates embeddings synchronously.

**Form fields**

| Field | Type | Notes |
|---|---|---|
| `file` | file | CSV. |
| `dataset_name` | `str` | Required and non-empty. |
| `generate_embeddings` | `bool` | Default `true`. |

**Response** — `200 OK`, `DatasetUploadResponse`:

| Field | Type |
|---|---|
| `dataset_id` | `str` |
| `dataset_name` | `str` |
| `inserted` | `int` |
| `skipped` | `int` |
| `embeddings_generated` | `int` |
| `description` | `str` |

`400` when `dataset_name` is empty, the file is missing, or
`process_csv_upload` raises `ValueError` (bad CSV).

### `GET /datasets/{dataset_id}/preview?limit=20`

First `limit` points of the dataset (no ordering guarantee beyond DB insertion
order).

**Response** — `200 OK`:

```json
{
  "dataset_id": "…",
  "dataset_name": "…",
  "description": "…",
  "points": [{"id": "…", "text": "…", "has_embedding": true}]
}
```

`404` when the dataset does not exist.

### `DELETE /datasets/{dataset_id}`

Cascades to data points and sessions at the DB level
([data-model.md → Cascade deletes](data-model.md)).

**Response** — `200 OK`: `{"dataset_id": "…", "dataset_name": "…"}`.
`404` when not found.

---

## Sessions

A session is one oracle's clustering conversation over one dataset. Sessions
move through three statuses: `active` → `converged` (oracle stopped
voluntarily) or `closed` (cognitive overload / forced).

### `GET /sessions`

List all sessions across all datasets.

**Response** — `200 OK`, array of:

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | UUID. |
| `name` | `str \| null` | |
| `dataset_id` | `str` | |
| `dataset_name` | `str` | `""` if the dataset row has been deleted. |
| `embedding_model` | `str` | Currently always `"default"`. |
| `status` | `"active" \| "converged" \| "closed"` | |
| `oracle_kind` | `"human" \| "persona"` | |
| `persona_snapshot` | `object \| null` | Frozen persona dict when `oracle_kind="persona"`. |

### `POST /sessions`

Create a new session over a dataset. Does **not** run clustering — call
`POST /clusters/{session_id}` after creation.

**Body** — `CreateSessionRequest`:

| Field | Type | Notes |
|---|---|---|
| `dataset_id` | `str` | Required. |
| `name` | `str \| null` | Optional display name. |
| `oracle_kind` | `"human" \| "persona"` | Default `"human"`. |
| `persona_snapshot` | `object \| null` | Required when `oracle_kind="persona"`. |

**Response** — `200 OK`, same shape as one element of `GET /sessions`.

`422` when the dataset doesn't exist, has no data points, or when
`oracle_kind="persona"` was sent without a `persona_snapshot`.

### `GET /sessions/{session_id}/state`

Resolved view of the session combining cluster geometry, oracle feedback
history, and the rolling preference summary. Built by
[`backend/session_state.py`](../backend/session_state.py).

**Response** — `200 OK`, `ChatSessionState` (see `src/schemas.py:29`):

| Field | Type |
|---|---|
| `session_id` | `str` |
| `turn_number` | `int` (≥ 0) — last persisted oracle turn |
| `dataset_name` | `str` |
| `embedding_model` | `str` |
| `status` | `"active" \| "converged" \| "closed"` |
| `clusters` | `Cluster[]` — only active (non-dissolved) clusters |
| `feedback_history` | `FeedbackEntry[]` |
| `oracle_preference_summary` | `str \| null` — rolling 3–5 bullet distillation |

`Cluster` fields: `id`, `session_id`, `name`, `description`, `created_at_turn`,
`dissolved_at_turn`, `size`, `representative_points`.

`404` when the session does not exist.

### `PATCH /sessions/{session_id}/state`

The only mutable field is `status` — used by the UI to force-close a session.

**Body**: `{"status": "active" | "converged" | "closed"}`.

**Response** — `200 OK`, the refreshed `ChatSessionState`. `400` when no
fields are provided, `404` when the session does not exist.

### `GET /sessions/{session_id}/eval`

Return the latest cached eval response (computed by `POST .../eval`). Always
sets `cached=true`. `404` when no cached eval exists.

### `POST /sessions/{session_id}/eval?force=false`

Run the seven-judge evaluation pipeline (`A1`–`A3` mathematical metrics,
`B1`–`B4` LLM judges) and persist the result to `eval_cache`. When `force` is
false (default), returns the cached row if its fingerprint matches the current
state and judges.

**Response** — `200 OK`, `EvalResponse`:

| Field | Type | Notes |
|---|---|---|
| `session_id` | `str` | |
| `k_final` | `int` | Active cluster count at evaluation time. |
| `cached` | `bool` | `true` when served from `eval_cache`, `false` when freshly computed. |
| `A1` | `A1Metrics` | `silhouette_initial`, `silhouette_final`, `trend[]`. |
| `A2` | `A2Metrics` | `turns`, `weighted_turns`, `termination` (`"converged"` \| `"cognitive_overload"`). |
| `A3` | `A3Metrics` | Per-turn cognitive load score, driver, final breakdown. |
| `B1` | `B1Metrics` | Overall synthesis verdict. |
| `B2` | `B2Metrics` | Per-cluster coherence (mean, min, per-cluster). |
| `B3` | `B3Metrics` | Oracle-request compliance. |
| `B4` | `B4Metrics` | Oracle contradiction / ambiguity. |

Sub-model fields are defined inline in
[`routers/sessions.py`](../backend/routers/sessions.py). `404` when the
session does not exist.

### `DELETE /sessions/{session_id}/delete`

Hard-deletes the session row (cascades to turns, clusters, soft assignments,
eval cache). **Response** — `200 OK`: `{"id": "…", "status": "deleted"}`.
`404` when the session does not exist.

### `GET /sessions/{session_id}/umap?geometry_aware=false`

2-D UMAP projection of every point in the session's dataset plus the cluster
assignments for every persisted turn. Cached in process memory per dataset
(and per `(session, turn)` when `geometry_aware=true`). Always sets
`Cache-Control: no-store` so the browser never serves a stale projection after
a new turn.

Set `geometry_aware=true` to additionally compute a second UMAP on the
hybrid `(D+1)` re-embed space for every `semantic_reembed` turn — the response
gains a `geometry_aware` field; the original layout stays intact.

`404` when the session has no data points or has not been clustered yet
(`project_session` raises `ValueError`).

---

## Clusters

The clustering lifecycle: `POST /clusters/{session_id}` writes turn 0;
subsequent merge/split/reembed operations are issued via `POST /turns` and
shift the per-turn `SoftAssignment` rows.

### `POST /clusters/{session_id}`

Run the initial k-means on the session's dataset embeddings, persist clusters
and soft assignments, and optionally name each cluster with an LLM. Idempotency
guard: once clusters exist for a session, returns `409`.

**Body** — `ClusteringRequest`:

| Field | Type | Notes |
|---|---|---|
| `k` | `int` | `2 ≤ k ≤ 50`. Ignored when `oracle_intent` is provided. |
| `generate_names` | `bool` | Default `true`. When `false`, skips the cluster-naming LLM call. |
| `oracle_intent` | `str \| null` | Free-text description; Claude extracts `k` (and an axis label) via `f_parse_clustering_intent`. |

**Response** — `200 OK`:

```json
{
  "session_id": "…",
  "k": 5,
  "silhouette_score": 0.42,
  "clusters": [{"id": "…", "name": "…", "description": "…", "size": 17}],
  "assignments_created": 84
}
```

`silhouette_score` may be `null` when `k < 2` or `k ≥ n_points`. `409` when
clusters already exist; `422` when the dataset has no points or k-means
rejects `k`.

### `GET /clusters/{session_id}/suggest-k?k_min=2&k_max=10`

Sweep k-means over the requested range and report the silhouette score for
each. Diagnostic only — does not write anything.

**Response** — `200 OK`:

```json
{
  "session_id": "…",
  "scores": {"2": 0.31, "3": 0.40, "…": 0.0},
  "recommended_k": 3
}
```

`422` when the dataset has no points or the requested range cannot be
evaluated.

### `GET /clusters/active?session_id=...`

Active (non-dissolved) clusters for a session, ordered by `created_at_turn`.

**Response** — `200 OK`, `Cluster[]` (see schema in `GET .../state`).

### `GET /clusters/history?session_id=...`

All clusters that ever existed for the session, including dissolved ones
(`dissolved_at_turn` is non-null on those).

**Response** — `200 OK`, `Cluster[]`.

### `GET /clusters/{cluster_id}`

Fetch one cluster by id.

**Response** — `200 OK`, a single `Cluster`. `404` when not found.

### `GET /clusters/{cluster_id}/points`

Hard members of the cluster at the latest snapshot turn, ordered by
descending soft-assignment probability. A point belongs to its arg-max cluster
across the session's clusters at that turn.

**Response** — `200 OK`, `ClusterPointsResponse`:

| Field | Type |
|---|---|
| `cluster_id` | `str` |
| `session_id` | `str` |
| `turn_number` | `int \| null` |
| `points` | `ClusterPoint[]` — `{id, text, probability}` |

`turn_number` is `null` and `points` is `[]` when the session has no
soft-assignment rows yet. `404` when the cluster does not exist.

---

## Turns

A turn is one oracle ↔ system exchange. The handler in `POST /turns` runs the
full engine pipeline (intent classification, operations, boundary repair,
planner, preference-summary update) under a single DB transaction.

### `GET /turns?session_id=...&limit=50`

Paginated list of turns ordered by ascending `turn_number`. `limit` clamps to
`[1, 200]`.

**Response** — `200 OK`, `TurnRead[]`. `404` when the session does not exist.

### `GET /turns/{session_id}`

All turns for the session, ordered by ascending `turn_number`. No pagination.

**Response** — `200 OK`, `TurnRead[]`. `404` when the session does not exist.

### `GET /turns/{session_id}/{turn_number}`

Fetch one turn.

**Response** — `200 OK`, `TurnRead`. `404` when the turn does not exist.

### `POST /turns`

Submit one oracle message and run the engine. Persists exactly one `Turn` row
on success.

**Body** — `InputOracle` (see `src/schemas.py:62`):

| Field | Type | Notes |
|---|---|---|
| `session_id` | `str` | |
| `raw_text` | `str` | Free-form oracle utterance. |
| `feedback_type` | `"global" \| "cluster" \| "point" \| "instructional"` | |
| `target_cluster_ids` | `str[]` | Optional; validated against active clusters. |
| `target_point_ids` | `str[]` | Optional. |
| `metadata` | `object` | Optional opaque dict. |

**Response** — `201 Created`, `TurnRead`:

| Field | Type |
|---|---|
| `session_id` | `str` |
| `turn_number` | `int` |
| `oracle_input` | `InputOracle` |
| `system_output` | `SystemTurn` |

`SystemTurn` carries `action` (`"show"` \| `"ask"` \| `"stop"`),
`clusters_updated`, `display.content` (the prose the UI renders),
`cognitive_load_score`, the `state_snapshot` dict (includes the executed
`operations` list and the planner's `reason`), and the per-turn `token_usage`
/ `cost_usd` accumulated across every LLM call inside the turn.

Error codes:

| Code | Meaning |
|---|---|
| `404` | Session not found. |
| `409` | Session is `closed`/`converged`, or no active clusters yet. |
| `422` | Unknown `target_cluster_ids`, invalid operation, missing `axis_label` on `semantic_reembed`, or `semantic_reembed` failure surfaced from the engine. |
| `500` | DB commit failed (`builder.commit()` / `db.commit()`). |
| `502` | LLM call failed or returned malformed JSON. |

Two soft-failure paths are surfaced as `201 Created` with `action="ask"`
instead of an HTTP error:

- `AxisNotDiscriminativeError` from `f_semantic_reembed` (the requested axis
  doesn't vary enough across the dataset / cluster) — the turn number is
  **not** advanced and the oracle is asked for a different axis.
- The clarify path: when `f_output` returns `action="clarify"`, the engine
  stages a `clarify_pending` op in the `state_snapshot` but performs no
  clustering changes. The next turn auto-confirms when `raw_text` is a short
  affirmation (`"yes"`, `"ok"`, `"sì"`, …), short-circuiting back into
  `semantic_reembed`.
