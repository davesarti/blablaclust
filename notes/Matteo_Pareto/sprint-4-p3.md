# Sprint 4 — P3 (Core Engine)

## What I built

Sprint 4 was split between two issues (#68 and #70) and a set of follow-up fixes that surfaced during testing. The main delivery is the `cluster_reembed` operation — single-cluster semantic re-embedding — along with a soft-assignment correctness fix that was silently breaking uncertainty detection after any merge or split.

---

### Issue #68 — Soft-assignment renormalisation after merge/split

**Problem.** After a merge or split operation, the dissolved clusters' probability mass was dropped from the snapshot but non-affected points were never renormalised. A point that originally had `{A:0.5, B:0.3, C:0.2}` after dissolving A and B ended up with `{C:0.2}` — a distribution that summed to 0.2, not 1.0. This made `f_cluster_uncertainty` useless after turn 1: the 0.30 overlap threshold was never triggered on sub-1.0-sum distributions, so boundary warnings and split suggestions stopped firing correctly on subsequent turns.

**Fix** (`src/engine/cluster_operations.py`). Added `_renormalize(distribution)` helper and called it at the end of both the merge and split snapshot update loops. Merged/moved points that intentionally collapse to `{new_cluster: 1.0}` are unaffected — renormalisation only touches non-subset points whose residual mass must sum to 1. This restores meaningful soft probabilities for uncertainty detection on every turn after the first.

---

### Issue #70 — Single-cluster semantic re-embedding (`cluster_reembed`)

**The feature.** When an oracle says *"split the electronics cluster by battery performance"*, the old system had no option between a plain geometric split (k-means on raw embeddings, no semantic guidance) and a full global re-embed (dissolves every cluster). `cluster_reembed` fills the gap: it extracts the hard-assigned subset of one named cluster, projects those points into a hybrid axis-weighted embedding space, clusters them semantically, and dissolves only the parent — all other clusters are untouched.

**Implementation** (`src/engine/cluster_operations.py` — `semantic_reembed_cluster()`):
1. Extract subset points via `_hard_cluster` on the current snapshot.
2. Call `reembed_for_axis(subset_points, axis_hint)` — reuses the existing hybrid embedding pipeline (cosine pole scoring → LLM fallback → standardised axis dimension appended to original embeddings).
3. Fit GMM on the hybrid matrix (same pattern as `initial_clustering`).
4. Update snapshot: subset points get new GMM soft probabilities; non-subset points have the dissolved parent's mass renormalised away.
5. Dissolve parent, register k new clusters, auto-name via `name_clusters` with `axis_hint`.

**Wired through:**
- `f_apply_operations.py`: new `cluster_reembed` op type handler
- `turns.py`: included in the boundary repair trigger (same as merge/split)
- `prompts/f_output.txt`: new op type + disambiguation rules (specific cluster + axis → `cluster_reembed`; whole dataset + axis → `semantic_reembed`; specific cluster, no axis → `split`)

---

### Follow-up fixes discovered during testing

**GMM instead of k-means in `semantic_reembed_cluster`.** Initial implementation used `_fit_kmeans` directly, inconsistent with `initial_clustering` which defaults to GMM (`USE_GMM=True`). Switched to `_fit_gmm` with k-means fallback on convergence failure.

**Auto axis_weight selection** (`src/engine/f_semantic_reembed.py`). The hybrid embedding previously used a hardcoded `axis_weight=0.7` for all axes. For sentiment-type axes (orthogonal to the original embedding space), 0.7 was too low — k-means still found topical clusters rather than positive/negative ones. Fixed by returning the scoring strategy from `reembed_for_axis` and auto-selecting the weight: cosine strategy (axis already in embedding space) → 0.5; LLM fallback strategy (axis orthogonal to embeddings) → 0.9. `semantic_reembed_cluster` passes `axis_weight=None` to get auto-selection; the global `semantic_clustering` still passes an explicit 0.7 to preserve existing behaviour.

**Dict-wrapped LLM scores** (`src/engine/f_semantic_reembed.py`). The LLM fallback scorer was returning all 5.0 (neutral fallback) on every batch, causing `std=0.00` and an `AxisNotDiscriminativeError`. Root cause: `extract_json_text` only matches `{...}` objects, not `[...]` arrays. When the model wrapped its scores in `{"scores": [...]}`, the outer dict was parsed, `isinstance(raw, list)` failed, and every batch silently fell back to 5.0. Fixed by unwrapping common object wrappers (`scores`, `results`, `values`, `data` keys) before the isinstance check.

**Directional scoring via poles** (`prompts/semantic_reembed.txt`). The LLM scoring prompt asked "how much does this text relate to `{axis}`?" — a relevance question. Every music review relates to sentiment, so all scores clustered near 7 with std ≈ 0.3. Fixed the prompt to score *position* between two concrete pole examples (0 = resembles the low pole, 10 = resembles the high pole). The poles were already being generated by `_generate_axis_poles` for the cosine path but were discarded for LLM fallback; they are now threaded through `_llm_axis_scores` and `_llm_score_sample` into the prompt. This produces a genuine spread across the 0–10 scale for sentiment, tone, formality, etc.

**`AxisNotDiscriminativeError` handling in `cluster_reembed`** (`backend/routers/turns.py`). The existing error handler for structural ops caught `(ValueError, KeyError)`, which included `AxisNotDiscriminativeError` (a subclass of ValueError). This showed up as a cryptic 422 "Engine produced an invalid operation" message. Added a specific `except AxisNotDiscriminativeError` branch that returns a user-friendly oracle message ("The axis doesn't vary enough within that cluster — try a different one"), matching the behaviour of the global semantic reembed handler.

**Name-to-cluster matching for oracle-supplied split names** (`src/engine/f_apply_operations.py`). Oracle-supplied `new_names` on split operations were mapped positionally to children (`new_names[0] → child[0]`). GMM returns children in arbitrary order, so the names were frequently swapped (e.g., a cluster of film reviews got named "Books"). Fixed with `_match_names_to_clusters()`: encodes oracle names with the sentence transformer, computes each child cluster's centroid embedding, and greedily matches by cosine similarity. Falls back to positional assignment if the ST model fails. This fix applies to all splits that carry oracle-supplied names, not just semantic ones.

**f_output prompt gaps for `cluster_reembed`** (`prompts/f_output.txt`). Three places in the prompt omitted `cluster_reembed` from their enumeration, causing the LLM to fall back to `split` even for axis-guided requests: (1) the action decision list in the INTENT PRIORITY section; (2) the "Do NOT emit SEMANTIC_REEMBED when oracle names a specific cluster" rule, which previously steered toward structural ops rather than `cluster_reembed`; (3) the op type enum in CONSTRAINTS. All three fixed.

---

### Evaluation fixes (own eval tooling)

**Oracle view point IDs** (`src/eval/oracle_view.py`). Example data points in the oracle's cluster panel showed only text snippets — the point IDs were added to the legal-ID set but never rendered. The LLM oracle had no UUIDs to reference, making `target_point_ids` effectively unusable and causing `boundary_pedant`-class errors (oracle passing text content as IDs). Fixed by rendering each example as `[uuid] text snippet`.

**Oracle prompt — pinning section** (`prompts/llm_oracle.txt`). Added a PINNING DATA POINTS AND CLUSTERS section explaining the drag-and-drop UI metaphor in terms the LLM oracle can act on: put a point id in `target_point_ids` and quote its text in `raw_text` to reference it; use this for move requests and boundary corrections.

**`--dataset` override flag** (`scripts/run_persona_eval.py`). Added `--dataset <name>` argument that overrides `persona.dataset` for all personas at runtime, enabling eval runs against any loaded dataset (e.g. `--dataset imdb_train`) without editing persona JSON files.

---

### Tests

Added `tests/test_semantic_reembed_cluster.py` — 6 tests covering:
- Parent cluster dissolution and bystander preservation
- Non-subset point renormalisation (probabilities sum to 1.0 after op)
- Subset soft-assignment correctness (distributions over new children only)
- Performance bound (<2 s with mocked reembed, verifying GMM + snapshot logic is not unexpectedly slow)
- Name-to-cluster matching by embedding similarity (Books → book-centroid cluster, Films → film-centroid cluster)
- End-to-end dispatch through `f_apply_operations` for the `cluster_reembed` op type

Fixed two pre-existing tests broken by API changes: `test_f_semantic_reembed` (reembed_for_axis now returns a tuple) and `test_f_apply_operations` (split with inline names now uses `_match_names_to_clusters`).

Full suite: 343 tests, all passing.

---

## Challenges

**Tracing std=0.00 to a silent parse failure.** The symptom was `AxisNotDiscriminativeError` with `std=0.00`, which could mean the LLM was failing or returning uniform scores. The LLM calls appeared in the log (so requests were going through), but scores were all 5.0. Eventually traced to `extract_json_text` only matching `{...}` — the model's `{"scores": [...]}` response was being deserialized as a dict, failing the `isinstance(raw, list)` check, and silently returning `[5.0] * n`. Hard to find because the error was silent and logged calls don't carry the actual response body.

**Positive/negative vs. extreme/moderate split.** First test of the sentiment split produced "Extreme Sentiment Reviews" and "Mixed Sentiment Reviews" instead of "Positive" and "Negative". The root cause was the scoring prompt asking about relevance (how much does this text relate to sentiment?) rather than direction (is this positive or negative?). The poles already existed for the cosine path but weren't being used for LLM fallback. Fixing this required threading the poles all the way through three function layers into the prompt.

**GMM geometry in 385 dimensions.** Even with `axis_weight=0.9`, the 10% original-embedding component could pull k-means centroids off the axis for datasets where topical structure is strong. GMM handles this better than k-means for boundary points because it models the distribution shape rather than just centroid distance — but the improvement only materialized after the axis_weight and scoring fixes were also in place.

---

## What I changed in other people's files

| Owner | File | Change |
|---|---|---|
| P1 | `backend/routers/turns.py` | Added `AxisNotDiscriminativeError` handler for `cluster_reembed`; added `cluster_reembed` to boundary repair trigger |
| P4 | `prompts/f_output.txt` | New `cluster_reembed` op type, updated INTENT PRIORITY decision tree, added disambiguation rules |
| P4 | `prompts/semantic_reembed.txt` | Rewrote scoring scale to use pole-anchored directional scoring |
| P4 | `prompts/llm_oracle.txt` | Added PINNING section |

---

## What's left / known gaps

- **NN propagation misclassifications.** LLM scoring samples 200 points per cluster; the rest inherit scores from their nearest neighbour in the original embedding space. Nearest-neighbour by topic ≠ nearest-neighbour by sentiment axis, producing a ~10–15% misclassification rate for boundary points. The oracle can correct these with explicit move operations. The fix (score all points) doubles LLM calls — deferred.

- **`semantic_clustering` (global reembed) still uses k-means.** The global `semantic_clustering.py` path was not changed this sprint; it still uses `_fit_kmeans` with softmax probabilities rather than GMM. Bringing it in line with `semantic_reembed_cluster` is a future clean-up.

- **Wall-time gap (2–3 min).** One session showed a 2:49 gap between the last LLM scoring call and `cluster_naming`, with no concurrent sessions running. Root cause is not yet identified — GMM fitting in 385 dimensions and embedding deserialisation from SQLite are the main candidates. Needs timing instrumentation.
