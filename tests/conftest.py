"""Project-wide pytest configuration.

Redirects clustering_log writes to a per-test tmp file so the suite never
pollutes the real ``logs/clustering_runs.jsonl``. Tests that want to inspect
the log file simply re-patch ``clustering_log._path`` themselves (the last
patch wins).

Also forces ``HARNESS_DRY_RUN=true`` for the whole test session. ``harness.py``
freezes ``DRY_RUN`` into a module-level constant at import time, so the value
depends on whether the env var was set before the first import. Setting it here
— before pytest imports any test module — makes the dry-run harness apply
deterministically regardless of test-collection order. (Previously only
test_turns_endpoint.py set it, so it lost the race when another module imported
harness first, causing real LLM calls and a spurious 422 in the full suite.)
"""

import os

os.environ.setdefault("HARNESS_DRY_RUN", "true")

import pytest

import src.logger as logger


@pytest.fixture(autouse=True)
def _isolate_clustering_log(tmp_path_factory, monkeypatch):
    target = tmp_path_factory.mktemp("clog") / "clustering_runs.jsonl"
    monkeypatch.setattr(logger, "_clustering_log_path", target)
