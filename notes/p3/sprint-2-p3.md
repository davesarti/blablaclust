# Sprint 2 — P3 (Core Engine)

## What I built

### Engine functions (all real implementations, no stubs)

- **f_output**: builds the prompt from `f_output.txt`, calls the LLM via a provider-agnostic wrapper, returns `(raw_dict, usage)` as a tuple so callers can track token costs. Registers oracle input and model response in the `ConversationContext` for multi-turn memory.

- **f_next_state**: calls `f_output`, rebuilds the cluster list from the model's response (falls back to current clusters if Claude returns nothing), checks for contradictions, appends the oracle turn to feedback history, increments `turn_number`, and returns a new immutable `ChatSessionState`. Updated to handle `f_output`'s tuple return and P1's rename of `target_cluster_id` → `target_cluster_ids`.

- **f_uncertainty**: reads `SoftAssignment` rows at the latest turn from the DB, computes `uncertainty_score = 1 - max(probability)` per data point, returns a `BoundaryPoint` list sorted descending by score. A score near 1.0 means the point is genuinely ambiguous between clusters. Returns `[]` safely when no clustering has run yet.

- **f_next_best_step**: rule-based show/ask/stop decision — no LLM call needed. Stops if cognitive load ≥ 4 or turn count > 20; asks if any boundary point has uncertainty ≥ 0.4 (surfaces the top 5 to the oracle); otherwise shows the current clustering state.

- **f_parse_clustering_intent**: takes free-text oracle input ("separate positive from negative reviews"), calls Claude with a structured extraction prompt, returns `{k, axis, reasoning}`. Falls back to `k=5, axis="semantic similarity"` on any parse or API failure. Used by the clustering endpoint to let the oracle describe what they want instead of picking k manually.

- **f_apply_operations**: dispatches Claude's structured operations (merge, split, rename) to P2's `cluster_operations.py` functions that execute real k-means-based reassignment. Each snapshot-writing operation (merge, split) consumes one `turn_number`; rename needs no snapshot. Returns the final `turn_number` used. Does not commit — caller owns the transaction.

### Supporting work

- **prompts/parse_clustering_intent.txt**: extraction prompt for `f_parse_clustering_intent` — explains how to infer k and clustering axis from natural language.
- **backend/routers/clusters.py**: added optional `oracle_intent` field to `ClusteringRequest` — when provided, `f_parse_clustering_intent` derives k automatically and the explicit `k` field is ignored. Backward compatible.
- **Guided clustering proposal**: proposed and implemented a two-mode clustering flow — automatic (k-means with suggested k) or guided (oracle describes intent in natural language, Claude extracts k and axis). Discussed with the team and confirmed as the right direction.

### Tests (48 total across P3 files, all passing)

| File | Tests |
|---|---|
| `test_f_next_state.py` | 7 |
| `test_f_uncertainty.py` | 9 |
| `test_f_next_best_step.py` | 11 |
| `test_f_parse_clustering_intent.py` | 9 |
| `test_f_apply_operations.py` | 12 |

## Challenges

- **Schema drift**: P1 renamed `target_cluster_id` → `target_cluster_ids` (singular optional → plural list) in both `InputOracle` and `FeedbackEntry`. Caught by existing tests, fixed in `f_next_state.py` and test fixtures.
- **f_output tuple return**: P4 changed `f_output` to return `(raw, usage)` for token tracking. Updated `f_next_state` to unpack correctly with `raw, _ = f_output(...)`.
- **Rebase conflict**: during a team rebase, `clusters.py` was overwritten by P2's version and our `oracle_intent` changes were lost. Re-applied manually after identifying the issue.
- **f_apply_operations existed as a stub**: a previous version of the file used the old architecture (Claude inventing soft assignments). Replaced entirely with the new version that delegates to P2's embedding-based functions.
- **Architectural bug identified**: the original design had Claude returning a full cluster list with invented data point assignments. Opus analysis flagged this as the most critical bug. Resolved by: (1) P4 updating the prompt to return structured operations, (2) P2 implementing real merge/split/rename functions, (3) P3 writing `f_apply_operations` as the dispatcher.

## What I need from others

- **P1**: wire up `f_apply_operations` in `turns.py` — after storing the Turn, call `f_apply_operations(raw.get("operations", []), session_id, turn_number, db)` then `db.commit()`. This is the last step to close the oracle feedback loop.
- **f_eval**: still a stub — not blocking anything in the current flow, but will need implementing before the final demo.

## Architecture note

Claude is the translator, not the clustering algorithm. The division of roles is:
- **K-means** → decides where every data point goes (based on embedding distances)
- **Claude** → understands natural language feedback and decides which structural operation to perform (merge, split, rename)
- **f_apply_operations** → bridges the two: takes Claude's decision and triggers the real k-means-based execution via P2's functions

Claude never touches data point assignments directly. All probability values in `SoftAssignment` come exclusively from k-means.
