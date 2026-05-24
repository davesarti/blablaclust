# Issue for P5 (arianna): Move `log_clustering_run` into `src/logger.py`

## Context

For Sprint 3 task 3 (structured logging of every clustering run) I needed to add
a logger. The original issue said "via `src/logger.py`", but that file is yours
(P5), and the team's golden rule says I can't modify another P's file without
explicit permission. As an interim I created `src/engine/clustering_log.py` —
same shape as your `log_llm_call`, so absorbing it into `src/logger.py` is a
copy-paste.

## What's needed from you

- [ ] Move `log_clustering_run` from `src/engine/clustering_log.py` into
  `src/logger.py`, next to `log_llm_call`. The signature is already aligned
  with that pattern (timestamp, JSONL append). Keep the `OSError` swallow —
  clustering must not abort because the log file can't be written.
- [ ] Update the import in `src/engine/initial_clustering.py`: from
  `from src.engine.clustering_log import log_clustering_run` to
  `from src.logger import log_clustering_run`.
- [ ] Delete `src/engine/clustering_log.py`.
- [ ] The four direct `clustering_log` tests in `tests/test_clustering_log.py`
  (`test_log_writes_all_fields`, `test_log_appends_one_line_per_call`,
  `test_log_handles_none_silhouette`, `test_log_swallows_io_errors`) should move
  to `tests/test_logger.py` (if it exists) or stay where they are with the
  import changed. The three integration tests (`test_initial_clustering_logs_*`)
  stay in `tests/test_clustering_log.py`.
- [ ] The autouse fixture in `tests/conftest.py` that redirects
  `clustering_log._path` to a tmp file during tests needs to be updated to
  patch the equivalent path in `src.logger` (e.g. `src.logger._clustering_log_path`,
  or whatever name you pick).

## Notes

- Log file is `logs/clustering_runs.jsonl` (mirrors `logs/llm_calls.jsonl`).
- Row schema: `{timestamp, session_id, k, backend, seed, n_points, silhouette, turn_number}`.
- `silhouette` is `null` when `k < 2` or `k >= n_points` (undefined cases) —
  preserve this behaviour in the move.
