# P3 — Core Engine: everything you need to know for the oral

---

## The big picture

The engine is a **Planner/Executor** agentic system. The oracle talks in free natural language. The system interprets it, executes changes, and decides what to say next — without the oracle ever needing to know the structure of the system. There is no fixed ground truth; the oracle's judgment IS the objective.

The turn loop in order:
1. **f_output** — LLM reads oracle text, emits structured JSON operations
2. **f_apply_operations** — dispatches each op to the clustering engine via TurnBuilder
3. **f_boundary_repair** — LLM corrects geometrically misplaced boundary points
4. **TurnBuilder.commit()** — writes everything atomically to the DB
5. **f_cluster_uncertainty** — reads soft assignments, detects overlap and diffuse clusters
6. **f_next_best_step** — decides: show, ask, or stop
7. **f_update_preferences** — updates rolling oracle preference summary
8. **f_eval** — end-of-session quality judge (not your code, P4)

---

## TurnBuilder

**What it is:** An in-memory staging layer. All ops read and write a Python dict `{point_id: {cluster_id: probability}}`. New clusters and dissolutions are staged in memory. One `commit()` at the end writes everything atomically.

**Why it exists:** Before it, each operation flushed to the DB immediately so the next op could read it. This caused **snapshot-axis drift** — the snapshot turn number advanced ahead of the conversation turn number. Two ops in the same turn could write inconsistent state. TurnBuilder fixes both by keeping the whole turn in memory until it's complete.

**Key rule:** No engine function ever calls `db.commit()`. Only the router does. The router starts a transaction, runs the turn, commits once if everything succeeded, rolls back if anything failed.

---

## GMM and soft assignments

**What:** Initial clustering uses a Gaussian Mixture Model (`sklearn`, `covariance_type='diag'`) instead of k-means.

**Why diag covariance:** Full covariance needs 384² ≈ 148,000 parameters per cluster. With ~100-400 points per cluster you can't estimate that reliably — the matrix becomes singular. Diagonal covariance assumes feature independence, only needs 384 parameters per cluster.

**Why GMM over k-means:** GMM gives native posterior probabilities from EM. K-means only gives hard assignments; to get soft assignments you'd need a softmax-of-distances approximation, which is an ad-hoc heuristic. The posteriors matter because:
- `f_cluster_uncertainty` needs them to detect overlap and diffuse clusters
- `f_boundary_repair` ranks points by uncertainty margin
- Generalization evaluator uses cluster geometry from GMM state
- K-means fallback still runs when GMM fails to converge

**The renormalization bug (issue #68):** After a merge dissolving clusters A and B, non-affected points had their A and B probability mass dropped but the remaining probabilities were never renormalized. A point `{A:0.5, B:0.3, C:0.2}` became `{C:0.2}` — summing to 0.2 not 1.0. This silently broke `f_cluster_uncertainty` from turn 2 onward because the 0.30 overlap threshold was never hit on sub-1.0 distributions. Fix: `_renormalize()` pass after every merge and split.

---

## f_output in detail — how natural language becomes operations

### What goes into the LLM call

`f_output` is a single LLM call. The input is built from five pieces:

1. **Session context** — session ID, turn number, total data points
2. **Current clusters** — numbered list `[1] id=<uuid> name="..." size=N description="..."`. The numbers let the oracle say "cluster 2" without knowing UUIDs; the LLM resolves the number back to a UUID
3. **Oracle input** — the raw text the oracle typed, plus `feedback_type` (global/cluster/point/instructional) and any pinned cluster/point IDs from the UI
4. **Conversation history** — all prior oracle turns and system responses in the session, so the LLM can see what was already asked and done
5. **Oracle preference summary** — rolling 3–5 bullet summary of the oracle's revealed preferences (e.g. "prefers fewer clusters", "wants sentiment-based grouping"), updated by `f_update_preferences` each turn

All of this is injected into `f_output.txt` via `render_prompt()`, which is just a `.format()` on the template file. The populated string becomes the **system prompt**. The conversation history becomes the **message list**. Together they go into `call_llm()`.

### What comes back

The LLM returns a single JSON object:
```json
{
  "action": "merge",
  "operations": [
    {"type": "merge", "cluster_ids": ["uuid-1", "uuid-2"], "new_name": "Media Reviews"}
  ],
  "display": "I merged the two clusters into Media Reviews because they covered overlapping topics."
}
```

- `action` is the top-level intent (merge / split / rename / move / semantic_reembed / cluster_reembed / clarify / no_change / explain / end)
- `operations` is the list of structured ops to execute — each has a `type` and the relevant IDs/parameters
- `display` is the plain-English explanation shown to the oracle

`loads_llm_json()` parses the response, tolerating markdown fences and stray quotes that LLMs sometimes add.

### How the LLM decides what op to emit

The prompt contains an **INTENT PRIORITY** section that instructs the LLM to:
1. Read the oracle's text first and decide the action from the text alone
2. Use pinned cluster/point IDs only if the chosen action actually needs them
3. Never emit a "move" just because point IDs were pinned — only if the text asks for a move

Key disambiguation rules baked into the prompt:
- "cluster by X" / "group by X" → `semantic_reembed` (whole dataset)
- "split [cluster name] by X" / "zoom into [cluster] using X" → `cluster_reembed` (single cluster + axis)
- bare noun phrase with no operation verb ("sentiment", "tone") → `clarify` (ask first)
- short affirmation after a clarify → `semantic_reembed` directly (confirmation handshake)
- "split cluster 2" with no axis → `split` (structural, no semantic guidance)
- "delete / remove cluster X" → rewritten as `merge` into nearest neighbour (no delete op exists)

### What happens to "cluster 2"

The prompt lists clusters as `[1] ... [2] ... [3] ...`. When the oracle says "merge cluster 2 and 4", the LLM resolves the ordinal to actual UUIDs and emits them in `cluster_ids`. The oracle never needs to know or type UUIDs.

### The clarify → confirm handshake

If the oracle says a bare axis ("sentiment"), the LLM emits `action: clarify` with an `axis_label`. The router stores this as a `clarify_pending` op in the snapshot. On the next turn, if the oracle says "yes" or any short affirmation, the router short-circuits — skips the LLM entirely — and fires `semantic_reembed` with the stored axis directly. Avoids an unnecessary LLM call for a simple confirmation.

### The ConversationContext

`ConversationContext` is a rolling message buffer. Every oracle turn is added with `add_oracle_turn()` and every system response with `add_system_turn()`. `build_messages()` returns the `{role, content}` list for the LLM call. If the context grows too large it's trimmed from the oldest end to stay within the token cap. The LLM sees the full conversation history because it's maintained in memory as the session progresses — not re-read from the DB each time.

---

## f_output and operation dispatch (short version)

**f_output:** Calls the LLM with current session state (clusters, conversation history, oracle preference summary) and the system prompt `f_output.txt`. LLM returns JSON: `{"action": "...", "operations": [...], "display": "..."}`.

**Operations supported:** `merge`, `split`, `rename`, `move`, `cluster_reembed`, `semantic_reembed` (handled separately in router).

**UUID fuzzy repair:** `_normalize_cluster_ids` runs before dispatch. Uses difflib to correct single-character transcription errors in LLM-emitted UUIDs. If the error is too large to uniquely match, it's left as-is and fails loudly with a 422.

**Constraints enforced in the prompt:**
- `semantic_reembed` is exclusive — can't be mixed with other ops
- Cluster IDs must be UUIDs from the current cluster list
- Count arithmetic: split-only to increase k, merge-only to decrease k
- No forward references (can't reference a cluster created by an earlier op in the same list)

**Error handling:** Errors propagate — no silent skipping. The router catches `ValueError/KeyError`, rolls back, returns HTTP 422 with the error message. This makes bugs visible rather than hiding them.

---

## Uncertainty detection

`f_cluster_uncertainty` reads soft-assignment posteriors from the DB and produces two signals:

- **Overlap:** two clusters have ≥30% probability mass for the same point simultaneously → merge candidate
- **Low cohesion:** mean winner-probability for a cluster is below 0.60 → split candidate

`f_next_best_step` uses these to decide whether to ask the oracle a question ("these two clusters seem similar — should I merge them?") or just show the result.

This replaced a per-point uncertainty function that didn't scale — surfacing 1200 individual uncertain points to an oracle is useless. Cluster-level signals give the oracle one actionable question per turn.

---

## Boundary repair

**Why it's needed:** GMM draws boundaries based on embedding geometry. The oracle's intent is semantic. Even with a hybrid embedding, topical structure can dominate and misplace points that are semantically wrong despite being geometrically close.

**How it works:**
1. After any structural op (merge, split, cluster_reembed), sample the most uncertain points from each affected cluster
2. "Most uncertain" = smallest margin between top-2 soft-assignment probabilities
3. Sample size: 10% of cluster size, capped at 30
4. One LLM call: "given the oracle's intent, does this point belong in this cluster?"
5. Misplaced points → `batch_move_points` → TurnBuilder
6. Then `TurnBuilder.commit()` — repair is folded into the same atomic snapshot

**Why 10% not fixed 10:** Fixed 10 = 2.5% of a 400-point cluster. 10% scales with cluster size so large clusters get proportionally more candidates reviewed without blowing up the LLM call count (capped at 30 = one batch).

---

## Semantic re-embedding

**The hybrid space:**
```
X = [orig_emb × √(1-w),   axis_score × √w]     shape: (N, D+1)
```

The oracle says "cluster by sentiment" → the system appends one axis score dimension to the original 384-dim MiniLM embedding, with weight w controlling how much the axis dominates.

**Two scoring strategies:**
1. **Cosine** (free, fast): generate LLM pole texts ("this product is amazing" / "this product is terrible"), score each point as `cosine(emb, pole_pos) - cosine(emb, pole_neg)`. Used when cosine variance across the dataset is above threshold — meaning the embedding already captures this axis.
2. **LLM fallback**: when cosine variance is too low (axis orthogonal to embeddings, e.g. sentiment on Amazon), LLM batch-scores 200 sampled points on 0–10 scale anchored to the poles. Remaining points inherit via nearest-neighbor in original embedding space.

**Auto-selected axis weight:**
- Cosine strategy → w = 0.5 (embeddings already partially capture the axis)
- LLM fallback → w = 0.9 (axis is orthogonal, need to override topical geometry)

This was the fix for the positive/negative split bug: at w=0.7, the 30% topical signal produced extreme-vs-moderate clusters instead of positive-vs-negative.

**Limitation:** One scalar dimension vs 384. An instruction-tuned embedding model would reorient all dimensions toward the axis — fundamentally richer. Deferred due to 8× CPU cost.

---

## cluster_reembed (issue #70)

**What:** Oracle zooms into one cluster and re-clusters it along a semantic axis without affecting any other cluster.

**How it works:**
1. Extract points hard-assigned to that cluster (`_hard_cluster` = argmax of soft distribution)
2. Run `reembed_for_axis` on that subset → hybrid matrix
3. Fit GMM on the hybrid subset
4. Dissolve only the parent cluster
5. Renormalize non-affected points
6. Auto-name sub-clusters with `axis_hint` context
7. Match oracle-supplied names to children by sentence-transformer centroid cosine similarity — not positional (avoids name swaps)

**Difference from global semantic_reembed:** Global dissolves everything. `cluster_reembed` is surgical — one cluster replaced by k sub-clusters, everything else unchanged.

**When axis_weight matters:** Same auto-selection logic. "Split the music cluster by sentiment" → LLM scoring strategy → w=0.9.

---

## Cognitive load

`f_cognitive_load` produces a deterministic score 1–5 from three signals:
- Turn count vs cap (20 turns)
- Pre-trim prompt token size vs cap (16,000 tokens)
- Active cluster count vs cap (25 clusters)

Score = max of three per-signal subscores. Planner stops at 5. The caps are engineering estimates, not empirically validated.

`f_next_best_step` reads this score as one input to its show/ask/stop decision. At score 5 it stops regardless of oracle intent. At score 4 it starts warning the oracle. Below 4 it's purely informational.

---

## Key numbers to know

- GMM diag covariance: 384 parameters/cluster vs 384² for full
- Boundary repair: 10% of cluster size, cap 30
- Semantic reembed axis_weight: 0.5 (cosine) or 0.9 (LLM)
- Cognitive load cap: 20 turns / 16,000 tokens / 25 clusters → stops at 5
- Overlap threshold in f_cluster_uncertainty: 0.30
- Cohesion threshold: 0.60 mean winner-probability
- Eval: 63 persona sessions (21 × 3 datasets), B3 pooled = 0.885, 84% oracle_satisfied

---

## P3 — Core Engine: oral presentation prep

## The turn loop

My main job was making the conversational loop work correctly end-to-end. Each oracle turn follows a fixed pipeline: `f_output` calls the LLM to classify the oracle's intent and emit structured operations, `f_apply_operations` dispatches those operations to the clustering engine, `f_boundary_repair` runs a post-op correction pass, `f_cluster_uncertainty` reads the updated state, and `f_next_best_step` decides what to show the oracle next. The router owns the DB transaction — it calls `TurnBuilder.commit()` exactly once per turn so that turn number, cluster lifecycle (`created_at_turn`, `dissolved_at_turn`), and soft assignments are always in lockstep. No engine function ever calls `db.commit()` directly.

---

## The TurnBuilder

Before this existed, each operation wrote its snapshot to the DB immediately so the next operation could read it back. This caused snapshot-axis drift (the snapshot turn number would get ahead of the conversation turn number), and it meant two operations in the same turn could produce inconsistent state. TurnBuilder is an in-memory staging layer: all operations during a turn read and write a Python dict (`point_id → {cluster_id: probability}`), new clusters and dissolutions are staged in memory, and a single `commit()` at the end writes everything atomically. This is the main architectural decision for P3.

---

## Soft assignments and GMM

Initial clustering uses a diagonal-covariance GMM rather than plain k-means. The reason is that GMM produces native posterior probabilities from the EM algorithm — each data point gets a genuine probability distribution over clusters rather than an approximation. This matters because the downstream uncertainty detection (`f_cluster_uncertainty`) and boundary repair both rely on those probabilities being meaningful. K-means required a softmax-of-distances approximation which was less accurate for boundary points.

**Bug fixed (issue #68):** after a merge or split, the dissolved cluster's probability mass was dropped from non-affected points but never renormalized. A point that was `{A:0.5, B:0.3, C:0.2}` after dissolving A and B would become `{C:0.2}`, summing to 0.2 instead of 1.0. This silently broke `f_cluster_uncertainty` after turn 1 because the overlap threshold (0.30) would never trigger on sub-1.0 distributions. The fix was a `_renormalize()` pass at the end of each merge and split operation.

---

## Uncertainty detection

`f_cluster_uncertainty` reads soft assignments from the DB and produces two signals: cluster-pair overlap (two clusters share significant probability mass — merge candidate) and low cohesion (a cluster's mean winner-probability is below threshold — split candidate). `f_next_best_step` uses these signals to decide whether to surface a suggestion to the oracle or just show the updated state. This replaced a per-point uncertainty function that didn't scale to real sessions.

---

## f_next_best_step and cognitive load

The planner has three possible outputs: show (display the result), ask (surface an uncertainty-driven question), and stop (end the session). Stop fires on two conditions: convergence (oracle signalled satisfaction) or cognitive overload. Cognitive load is estimated per turn by `f_cognitive_load`, with caps defined in `cognitive_load_caps.py` — if the session has run too many turns, used too many tokens, or has too many active clusters, the system stops rather than continuing into a degraded LLM quality region.

---

## f_output and operation dispatch

`f_output` calls the LLM with the current session state and oracle input and gets back a JSON object with an action type and an operations list. The prompt (`f_output.txt`) defines the grammar of valid operations and their constraints — for example, a `semantic_reembed` is exclusive (cannot be combined with other ops), and cluster IDs must be UUIDs from the current cluster list. `f_apply_operations` routes each operation type to the right clustering function. Unknown operation types are silently skipped so future protocol extensions don't crash old sessions.

---

## Boundary repair

After any structural operation (merge, split, or cluster_reembed), `f_boundary_repair` samples the most uncertain boundary points from each affected cluster — 10% of cluster size, capped at 30 (uncertainty is measured as the margin between the top-2 soft-assignment probabilities) and asks the LLM whether each point belongs where GMM placed it, given the oracle's stated intent. Points flagged as misplaced are moved via `batch_move_points` before the snapshot is committed. This corrects the geometric errors that arise when GMM draws a boundary based on distance rather than semantic meaning.

---

## cluster_reembed (if asked)

I built the mechanism that lets the oracle zoom into a single cluster and re-cluster it along a semantic axis without touching the rest. The key engineering decision was reusing the existing `reembed_for_axis` pipeline (which was already there for global reembedding) but scoping it to the cluster's hard-assigned subset, dissolving only the parent, and renormalizing the non-affected points. I also added auto-selection of `axis_weight` based on the scoring strategy: if the axis is already captured by the original embeddings (cosine strategy) → 0.5; if it's orthogonal (LLM fallback strategy, e.g. sentiment) → 0.9. If asked deeper questions about the reembedding algorithm itself (hybrid space construction, pole generation) — that's pre-existing infrastructure, not P3.
