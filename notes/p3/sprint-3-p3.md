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

---

## Sprint 3 continuation — engine fixes and evaluation scaffold

*The following work was done after the initial sprint-3 write-up above, continuing into the same sprint as scope expanded.*

### Engine fixes

- **Inline rename on merge** (`src/engine/f_apply_operations.py`). The prompt already specified a `new_name` field on merge operations but the applier ignored it. Fixed: after `merge_clusters` returns, read `op.get("new_name")` and call `rename_cluster` on the new cluster if the oracle supplied a name. `auto_name` always runs first (for the description), then the inline name overrides only the name field. Closes the "merge and call it X" feature.

- **Inline rename on split** (`src/engine/f_apply_operations.py`). Same pattern as merge: `op.get("new_names")` is a list aligned to split children. After `split_cluster` + `db.flush()`, zip children with the list and call `rename_cluster` for each non-empty name. Name-to-child mapping is positional (best-effort; k-means returns children in no oracle-meaningful order — documented caveat, oracle can fix misalignment in the next turn).

- **Description preservation on rename** (`src/engine/f_apply_operations.py`). `rename_cluster` unconditionally overwrites both `name` and `description`. When the LLM omits `new_description`, passing `""` blanked the auto-generated description. Fixed by looking up `existing.description` before the call and falling back to it when the op has no description.

- **`auto_name=False` stripping descriptions** (`src/engine/f_apply_operations.py`). An earlier optimisation skipped the `name_clusters` LLM call when the oracle had supplied an inline name (`auto_name=False`). This also skipped description generation. Fixed by always running `auto_name=True` (default) and only overriding the name field via `rename_cluster` afterward.

- **Verbatim oracle names** (`prompts/f_output.txt`). Gemini was rewriting oracle-chosen names — "miss"/"piss" became "Movies with tonal issues"/"Poorly Made". Added an explicit constraint: *"copy oracle-supplied names VERBATIM — character-for-character — do NOT rephrase, expand, abbreviate, censor, or sanitise."* Confirmed fixed via `scripts/dump_split_state.py` which showed the LLM emitting the oracle's exact strings after the change.

- **ASK_THRESHOLD raised** (`src/engine/f_next_best_step.py`). Threshold for triggering an `ask` action raised from 0.4 to 0.45 to reduce spurious clarification requests (issue #24).

- **Termination reason split** (`src/engine/f_next_best_step.py`). The single `"max_turns_or_load_reached"` code was replaced with two distinct codes: `"cognitive_overload"` (when `cognitive_load_score >= 5`) and `"max_turns_reached"` (when `turn_number > MAX_TURNS`). Threshold kept at 5 (the maximum) per a colleague's change: load=4 is "heavy but manageable", only load=5 should auto-stop. Required by the A2 metric in the quality spec.

- **Logging on silent fallbacks** (`src/engine/cluster_naming.py`, `src/engine/f_parse_clustering_intent.py`). Replaced bare `except Exception: pass` / `return defaults` with `log.warning(...)` calls so LLM failures are visible in the log instead of silently producing placeholders.

- **`f_eval` JSON parsing** (`src/engine/f_eval.py`). `f_eval` called `json.loads(msg.text)` directly. When the LLM returned markdown-fenced JSON (which Gemini does even when the prompt forbids it), this raised `JSONDecodeError`. Fixed to use `extract_json_text(msg.text)` consistent with every other engine function.

### DB schema fix

- **`sessions.name` column missing** (`data/demo_database.db`). A colleague added a `name` field to the `ChatSession` model but the existing SQLite file wasn't migrated. The server crashed on startup with `no such column: sessions.name`. Fixed with `ALTER TABLE sessions ADD COLUMN name VARCHAR(255)` without wiping data.

### Evaluation scaffold (new)

Built the full v1 evaluation framework described in `docs/evaluation-plan.md`:

- **`src/engine/f_validate_point.py`** — new B4 judge function. Takes a data point, its assigned cluster, and the full session state; asks an independent LLM whether the point belongs in the cluster given the oracle's feedback history. Returns `{endorsed: bool, confidence: float, reasoning: str}`.

- **`prompts/f_validate_point.txt`** — judge prompt for B4. Provides cluster name/description/representatives and oracle feedback history; asks for the three-field JSON. Uses the same `extract_json_text` parsing pattern.

- **`scripts/run_eval.py`** — the eval runner. Drives scenario JSON files through the live HTTP API, collects all six metrics (A1–A2, B1–B4), writes `results.jsonl` + `summary.md`. B4 samples the 3 most uncertain points per cluster (lowest soft-assignment probability = highest uncertainty, making it a boundary stress test). Always tears down the eval session in `finally`. Usage: `PYTHONPATH=. python scripts/run_eval.py --scenarios scenarios/*.json --out reports/`.

- **Five scenario files** (`scenarios/`):
  - `stable_oracle.json` — approves initial clustering, no structural changes; expects fast `converged`, high B1.
  - `sentiment_split.json` — split + rename + accept; exercises split and rename ops in one session.
  - `topic_merge.json` — starts with k=7 (over-fragmented), oracle merges related clusters down; tests merge ops and coherence improvement.
  - `contradictory_oracle.json` — oracle splits then un-splits then re-splits; tests agent robustness under flip-flopping intent. Note: the built-in B3 detector requires `target_cluster_ids` to overlap between turns; since scenario files use empty arrays (IDs aren't known at authoring time), B3 reads 0 even though the intent is contradictory. Documented in the scenario's description field.
  - `high_load_oracle.json` — 15 turns of relentless small nudges; designed to hit `cognitive_overload` termination at turn 14 (when `round(14/20 * 5) = 4`).

- **`docs/evaluation-plan.md`** — new planning and reference doc covering the measurement approach, deliverable list, scenario corpus, and a plain-language "what was built" section for non-experts.

### End-to-end validation

Ran the full five-scenario suite against the live API. Results:

| Scenario | Termination | B1 coherence | B4 endorsement |
|---|---|---|---|
| stable_oracle | converged | 0.85 | 27% |
| sentiment_split | converged | 0.75 | 67% |
| topic_merge | converged | 0.90 | — |
| contradictory_oracle | converged | 0.75 | — |
| high_load_oracle | **cognitive_overload** | 0.75 | — |

`high_load_oracle` is the only scenario to hit the cognitive_overload termination code, confirming the threshold logic works. B4 endorsement rates are intentionally low (boundary stress test); `topic_merge` coherence of 0.90 is the highest across runs, consistent with the oracle performing meaningful structural improvements.

### What's left / known gaps

- **B3 with real cluster IDs**: contradiction detection only fires when `target_cluster_ids` overlap. Scripted scenarios can't pre-fill these since IDs are assigned at runtime. A future approach could use a two-pass runner (first pass discovers IDs, second pass replays with real IDs) or the runner could patch IDs dynamically.
- **Three missing scenarios**: `topic_merge`, `contradictory_oracle`, `high_load_oracle` were added — the corpus is now complete.
- **Human rater study**: deferred per spec. The eval produces comparable LLM judge scores; human validation can start once we have a baseline run to compare against.
