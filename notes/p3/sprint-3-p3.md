# Sprint 3 — P3 (Core Engine)

## What I built

Sprint 3 was mostly a stabilisation pass: the loop existed end-to-end after sprint 2 but several bugs only surfaced once the UI was driving real OpenRouter/Groq traffic against the engine. I fixed those, made engine failures visible, and cleaned up cross-team integration points that were blocking real demos.

### Engine fixes (own area)

- **Multi-operation snapshot integrity** (`src/engine/f_apply_operations.py`). When one oracle turn emitted two snapshot-writing operations (e.g. two splits, or a merge + split), the second op called `_load_latest_snapshot` and read a stale view of the DB because the session is configured with `autoflush=False`. In the failure mode I reproduced, the second split wrote 966 carry-forward rows under the *dissolved* "Low Quality Films" cluster while its actual sub-clusters got zero. Fixed with a `db.flush()` between operations so every op sees the previous one's results.

- **Name children after split** (`src/engine/f_apply_operations.py`). `split_cluster` hardcodes child names to `"<parent> - part 1/2"`. After the flush, I now load the freshly written soft assignments for the two new clusters and call `name_clusters` on them — so split children show up with semantic LLM-generated names instead of "part 1/2" placeholders. This is the first half of issue #21 (merge still needs the same treatment).

- **No more silent error swallowing** (`src/engine/f_apply_operations.py` + `tests/test_f_apply_operations.py`). The engine no longer wraps merge/split/rename in `try/except`. A bad cluster_id, an already-dissolved cluster, or a missing required field all raise `ValueError`/`KeyError` instead of being swallowed. Updated `test_value_error_from_merge_propagates` and added `test_missing_required_field_raises_key_error` to lock in the propagation contract. Closes the P3 half of issue #22.

### Cross-team integration fixes (with caveats)

- **Engine errors surface as HTTP 422** (P1 territory — `backend/routers/turns.py`). Wrapped the `f_apply_operations` call in `try/except (ValueError, KeyError)`, rolled back the transaction, and re-raised as `HTTPException(422, detail=...)` so the original exception message reaches the UI banner instead of a bare "500 Internal Server Error". This is the visible-failure layer that pairs with the engine fix above.

- **Provider-agnostic conversation context** (P4 territory — `src/harness.py`).
  - `ConversationContext.build_messages()` called Anthropic's `count_tokens` unconditionally, so every session with `LLM_PROVIDER != claude` crashed on turn 2 with "ANTHROPIC_API_KEY missing". Now the trim only runs when the provider is Claude; other providers have large enough context windows that the hard 8 k token cap is unlikely to bite.
  - `ConversationContext.add_system_turn()` stored the `Display` dict directly as message `content`, but every chat-completions API requires `content` to be a plain string. Extract `content` from the dict (with string + JSON fallbacks) before appending. This was the bug behind the "400 — content must be a string" error on OpenRouter.
  - New `extract_json_text()` helper that strips ```` ```json ``` ```` fences before `json.loads()`. Used in `f_output`, `cluster_naming`, and `f_parse_clustering_intent`. Gemini Flash insists on wrapping JSON in fences even when the prompt forbids it.

- **Prompt constraint on forward-references** (P4 territory — `prompts/f_output.txt`). Added two lines: "NEVER use a cluster name as a cluster_id" and "you cannot reference a cluster created earlier in the same operations list". This addresses the failure mode where the LLM emitted `{"type": "merge", "new_name": "Music"}` followed by `{"type": "split", "cluster_id": "Music"}` — the second op had no real UUID to point at.

- **Auto-seed on startup** (P5 territory — `scripts/serve_ui.py`). New `_auto_seed()` runs before uvicorn, loads `data/train.csv` as `amazon_reviews`, and generates the sentence-transformer embeddings. Idempotent: every run after the first prints "DB already populated — skipping." First-run experience is now zero-step.

- **UI: ambiguous-point list rendered as text** (`ui/index.html`). `handleSystemTurn()` was doing `items.join('\n')` on an array of dicts, producing a column of `[object Object]` lines in the chat history. Now formats boundary points (truncated text + uncertainty score) and cluster summaries (name + size) explicitly.

## Challenges

- **Diagnosing the multi-op snapshot bug.** The symptom was "Low Quality Films - part 1: 0 points / part 2: 0 points" in the UI after a successful split. Spent some time digging through `hard_cluster_stats` thinking it was a read-side bug before checking the actual `soft_assignments` table — found 966 rows assigned to the dissolved parent cluster at the wrong turn, which traced back to `autoflush=False` and the second op's `_load_latest_snapshot` reading pre-op DB state.

- **Knowing where to stop being defensive.** The first version of the error-handling fix wrapped every op in `try/except` and silently skipped bad ones. The user (rightly) pointed out that this hides real bugs — "we don't need the code to pass all the tests while silently skipping where it does not work, so that this stuff comes back to bite us later." Reverted to strict propagation with HTTP 422 as the visible layer.

- **Cross-team file edits.** This sprint touched P1, P2, P4, and P5 files (see below). Each change is small and scoped, but I should make sure the owners see what changed.

## What I changed in other people's files

Heads-up to the team:

| Owner | File | Change |
|---|---|---|
| P1 | `backend/routers/turns.py` | Added `try/except` around `f_apply_operations` to surface engine errors as HTTP 422 with detail. Rolls back the transaction on failure. |
| P2 | `src/engine/cluster_naming.py` | One-line change: `json.loads(extract_json_text(response.text))` instead of `json.loads(response.text)`. |
| P4 | `src/harness.py` | Three fixes: `extract_json_text` helper, `add_system_turn` handles Display dict, `build_messages` is provider-aware. |
| P4 | `prompts/f_output.txt` | Added two constraints about cluster_id being a UUID and no forward-references. |
| P5 | `scripts/serve_ui.py` | Added `_auto_seed()` function — replaces the manual curl upload step on first run. |

## What I need from others

- **P2**: pick up the merge half of issue #21 — calling `name_clusters` on the merged cluster in `f_apply_operations` (or moving the naming responsibility into `merge_clusters` itself). Mirrors what I did for split.
- **P4**: silent fallbacks in `cluster_naming.py` (`except Exception: continue`) and `f_parse_clustering_intent.py` (`except Exception: return defaults`) still exist. Issue #22 originally wanted log lines at every such point. I left them as-is because both have visible indicators (placeholder name, default k=5) and I didn't want to expand scope, but P4 might want to add `log.warning` calls there.
- **`f_eval`**: still a stub. Now that the loop is visibly stable, this becomes the next real piece of work for the demo.

## Notes on what tests still don't cover

`f_apply_operations` is unit-tested with `MagicMock` for the DB session. The new code paths (`db.flush()`, the `name_clusters` follow-up call) don't fail under mocks because every mocked query returns another mock that's truthy. The real exercise is the live UI loop — which is now my main integration confidence. An integration test against an in-memory SQLite session that actually executes the split + name flow would be valuable; not in scope for this sprint.
