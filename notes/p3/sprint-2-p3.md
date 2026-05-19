# Sprint 2 — P3 (Core Engine)

## What I built:
- **f_output**: implemented real logic — builds the prompt from `f_output.txt`, calls the LLM via a provider-agnostic wrapper, and returns the raw JSON response from the model
- **f_next_state**: implemented real logic — calls `f_output`, rebuilds the cluster list from the model's response, checks for contradictions via P4's stub, appends the oracle turn to feedback history, and returns a new `ChatSessionState`
- **call_llm**: added a provider-agnostic wrapper to `src/harness.py` that routes to Claude or GPT based on the `LLM_PROVIDER` environment variable — all engine functions use this so the system works with either provider without code changes
- **Unit tests**: `tests/test_f_next_state.py` — 7 tests covering turn number increment, output type, cluster fallback when LLM returns nothing, feedback history growth, feedback entry correctness, empty cluster handling, and contradiction stub behavior. All passing with `HARNESS_DRY_RUN=true`
- **schemas.py**: added `contradictions` field to `ChatSessionState` with a default empty list — backward compatible, flagged to P1

## Challenges:
Running pytest required using `python -m pytest` instead of the system `pytest` binary since the conda environment wasn't on the system PATH. Also ran into a mismatch between the field names in `schemas.py` (`target_cluster_id`) and what I was passing in `f_next_state` (`target_id`) — caught by the tests.

## What I need from others:
- **P4**: `prompts/f_next_state.txt` — `f_next_state` currently uses `f_output.txt` as a fallback. Swapping to the real prompt is a one-line change once P4 pushes it
- **P2**: confirmation that k-means has been run and `SoftAssignment` table is populated — needed to implement `f_uncertainty`
- **P1**: `crud.get_soft_assignments(session_id, db)` in `crud.py` — needed by `f_uncertainty` to read from the DB. Also need to coordinate on `POST /sessions/{id}/turns` which is a joint P1+P3 endpoint

## Next steps:
- Implement `f_uncertainty` with real boundary point scoring from `SoftAssignment` (waiting on P2 + P1)
- Implement `f_next_best_step` show/ask/stop decision logic (waiting on `prompts/f_next_best_step.txt` from P4)
- Wire up `POST /sessions/{id}/turns` with P1