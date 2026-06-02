# Data Model

The persistence layer for the conversational clustering system. Defined as
SQLAlchemy ORM models in [`src/models.py`](../src/models.py); SQLite in dev,
PostgreSQL in production. The schema is smoke-tested in
[`tests/test_data_model_smoke.py`](../tests/test_data_model_smoke.py) — that test
is the executable companion to this doc, so if the two ever disagree, trust the
test and fix the doc.

## Overview

```
ChatSession ──< Cluster ──< SoftAssignment >── DataPoint ──< Dataset
     │                          (turn N)
     └──< Turn
```

A **session** is one oracle's clustering conversation over a dataset — it is
the central entity. Within a session, **clusters** are created and dissolved
over time, and a **soft assignment** records the probability that a given
**data point** belongs to a given cluster *at a specific turn*. **Turns** are
the conversation log.

A **dataset** is a named, described collection of data points. Data points are
**dataset-scoped, not session-scoped**: they exist independently of any session
(they're seeded and embedded once) and are referenced by soft assignments across
sessions. Sessions reference a dataset via `dataset_id`; deleting a dataset
cascades to its data points and sessions at the DB level.

## Tables

### `datasets` — `Dataset`

A named, described collection of data points.

| Column | Type | Notes |
|---|---|---|
| `id` | `String(36)` | PK (UUID). |
| `name` | `String(255)` | Unique. Human-readable dataset name. |
| `description` | `Text` | LLM-generated or manually supplied; empty string by default. |

Relationship: `data_points` (one-to-many, `delete-orphan`). Deleting a dataset
also cascades to its sessions at the DB level (`ondelete="CASCADE"` on the FK in
`sessions`), but that relationship is not surfaced as an ORM collection on
`Dataset`.

### `data_points` — `DataPoint`

A single record from the source dataset, plus its embedding.

| Column | Type | Notes |
|---|---|---|
| `id` | `String(36)` | PK (UUID). |
| `dataset_id` | `String(36)` | Indexed. FK → `datasets.id` (`ondelete="CASCADE"`). |
| `data` | `JSON` | The raw record, e.g. `{"text": "..."}`. |
| `embedding` | `JSON`, nullable | Vector as a JSON list; `NULL` until embeddings are generated. |

Relationships: `dataset` (many-to-one), `soft_assignments` (one-to-many, `delete-orphan`).

### `sessions` — `ChatSession`

One clustering conversation.

| Column | Type | Notes |
|---|---|---|
| `id` | `String(36)` | PK (UUID). |
| `name` | `String(255)`, nullable | Optional human-readable label. |
| `dataset_id` | `String(36)` | Indexed. FK → `datasets.id` (`ondelete="CASCADE"`). |
| `embedding_model` | `String(255)` | Model used to embed the points. |
| `status` | `String(32)` | Default `active`. **Constraint** `ck_sessions_status`: one of `active`, `converged`, `closed`. |
| `oracle_kind` | `String(16)` | Default `human`. **Constraint** `ck_sessions_oracle_kind`: one of `human`, `persona`. |
| `persona_snapshot` | `JSON`, nullable | Persona config frozen at session creation; `NULL` for human sessions. |

Relationships: `dataset` (many-to-one), `clusters` and `turns` (one-to-many,
both `delete-orphan` — see [Cascade](#cascade-deletes)).

### `clusters` — `Cluster`

A cluster within a session. Clusters are not deleted when they go away; they are
**dissolved** by stamping the turn at which they ceased to exist, preserving
history.

| Column | Type | Notes |
|---|---|---|
| `id` | `String(36)` | PK (UUID). |
| `session_id` | `String(36)` | FK → `sessions.id`. |
| `name` | `String(255)` | LLM-generated or `Cluster N` placeholder. |
| `description` | `Text` | LLM-generated; may be empty. |
| `created_at_turn` | `Integer` | Turn the cluster first appeared. |
| `dissolved_at_turn` | `Integer`, nullable | `NULL` = still active; otherwise the turn it was dissolved. |

**Constraint** `ck_clusters_turn_order`: `dissolved_at_turn IS NULL OR
dissolved_at_turn >= created_at_turn` — a cluster cannot be dissolved before it
was created.

Relationships: `session` (many-to-one), `soft_assignments` (one-to-many,
`delete-orphan`).

### `soft_assignments` — `SoftAssignment`

The probability that a data point belongs to a cluster **at a given turn**. This
is the heart of the snapshot model.

| Column | Type | Notes |
|---|---|---|
| `data_point_id` | `String(36)` | PK part, FK → `data_points.id`. |
| `cluster_id` | `String(36)` | PK part, FK → `clusters.id`. |
| `turn_number` | `Integer` | PK part. **Constraint** `ck_soft_assignments_turn_number`: `>= 0`. |
| `probability` | `Float` | **Constraint** `ck_soft_assignments_probability`: in `[0.0, 1.0]`. |

The composite primary key `(data_point_id, cluster_id, turn_number)` is what lets
each turn hold a complete, independent snapshot of the clustering.

Relationships: `data_point` and `cluster` (both many-to-one).

### `turns` — `Turn`

One exchange in the oracle conversation.

| Column | Type | Notes |
|---|---|---|
| `session_id` | `String(36)` | PK part, FK → `sessions.id`. |
| `turn_number` | `Integer` | PK part. **Constraint** `ck_turns_turn_number`: `>= 0`. |
| `oracle_input` | `JSON` | What the oracle said. |
| `system_output` | `JSON` | What the system replied (display payload). |

Relationship: `session` (many-to-one).

## The turn / snapshot model

`turn_number` is the spine of the whole model, and it means two related things:

- **`Turn.turn_number` starts at 1** in practice — oracle turns are 1, 2, 3, …
  The `ck_turns_turn_number` constraint only enforces `>= 0`, matching the
  soft-assignment floor; the "first oracle turn is 1" rule lives in the
  `create_turn` endpoint, not the schema.
- **`SoftAssignment.turn_number` starts at 0** (`ck_soft_assignments_turn_number`).
  Turn **0** is the *pre-oracle* state: the initial k-means clustering recorded
  before any conversation. Oracle-driven re-clusterings are written at turns 1+.

A soft-assignment snapshot at turn *N* is **complete**: every data point has a
probability for each cluster it belongs to at that turn. Readers such as
`f_uncertainty` and `hard_cluster_stats` take `max(turn_number)` and expect a
full picture. Cluster operations (merge / split / rename / move) therefore write
a fresh full snapshot at a turn strictly greater than the latest — touched points
get new probabilities, every other point is carried forward. A point's **hard**
cluster is the `argmax` of its probabilities at that turn.

Because assignments are soft (softmax over distances to centroids), every cluster
keeps a small non-zero probability on every point. So a cluster is never
*literally* empty — losing all of its hard (argmax) members does **not** dissolve
it; it stays active as long as it holds any soft mass. Dissolution is an explicit
act (e.g. merge/split set `dissolved_at_turn`), not a side effect of mass dropping
to zero.

## Cascade deletes

`ChatSession` owns its `clusters` and `turns`, and `Cluster` owns its
`soft_assignments`, all via `cascade="all, delete-orphan"`. Deleting a session
therefore removes its clusters, turns, and (transitively) their soft assignments.

`Dataset` owns its `data_points` via ORM `cascade="all, delete-orphan"`. It also
cascades to sessions at the **DB level** (`ondelete="CASCADE"` on both
`data_points.dataset_id` and `sessions.dataset_id`), so deleting a dataset wipes
its points and all sessions (plus their clusters/turns/assignments transitively).

`DataPoint` rows survive session deletion — they're dataset-scoped. Deleting a
data point cascades to its own soft assignments only.

> SQLite does not enforce foreign keys unless `PRAGMA foreign_keys=ON` is set; the
> smoke test enables it on connect so FK behaviour matches production.

## Conventions

- IDs are UUID strings (`String(36)`), generated by the caller.
- Engine functions never touch the DB session — the **caller owns the
  transaction**. Operations stage rows on the session and let the caller commit,
  so a multi-operation oracle turn stays atomic.
- JSON columns (`data`, `embedding`, `oracle_input`, `system_output`) round-trip
  as native dict / list.
