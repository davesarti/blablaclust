# Sprint 3 — P2 (Data & Embeddings)

## What I built

Sprint 3 for P2 was a focused pass on two issues against the existing
clustering code — both resolved entirely within P2-owned files, no cross-team
edits.

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

## Results

| Check | Result |
|---|---|
| `test_cluster_naming.py` (new, 12 tests) | all pass |
| `test_cluster_operations.py` (21 tests, 18 pre-existing + 3 new) | all pass |
| `test_f_apply_operations.py` (P3, uses both changed functions) | still passes |
| Single LLM call regardless of cluster count | verified (mock asserts 1 call) |
| `split_cluster(k=3)` on a 6-point cluster | 3 children, parent dissolved |
| `split_cluster(k=1)` / `k` exceeding point count | raise `ValueError` |

The 3 new split tests use a `db_big_cluster` fixture (six well-separated points
in one cluster) so real k-means can be exercised for k > 2; the naming tests
mock `call_llm`, so no API key is needed to run them.

Suite-wide note: 7 tests in `test_f_next_state.py` fail in the local env with
`ModuleNotFoundError: tiktoken` (imported by `harness_openrouter.py`). This is a
pre-existing environment gap, unrelated to these changes.

## What I changed in other people's files

Nothing. Both issues were fully solvable inside P2-owned files
(`cluster_naming.py`, `cluster_operations.py`, `prompts/cluster_naming.txt`,
and the two test files).

## What I need from others

- **P3** — `f_apply_operations.py` calls `split_cluster(...)` without passing
  `k`, so it currently always splits in two. The *capability* for k > 2 now
  exists in `split_cluster`; wiring it to an oracle request like "split into
  three" is a separate change in P3's file. Not a defect — just a heads-up if
  end-to-end variable-arity split is wanted for the demo.
- **P4** — `prompts/cluster_naming.txt` changed shape (now takes a
  `{clusters_block}` variable and returns an id-keyed JSON object). If P4 keeps
  a prompt-versioning registry, the hash for this prompt needs refreshing.

## Commits

- `5e0f300` — refactor: name all clusters in a single LLM call
- `2fdb8f6` — feat: add k parameter to split_cluster

Both pushed to `origin/main`.
