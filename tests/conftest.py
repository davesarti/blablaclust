"""Project-wide pytest configuration.

Redirects clustering_log writes to a per-test tmp file so the suite never
pollutes the real ``logs/clustering_runs.jsonl``. Tests that want to inspect
the log file simply re-patch ``clustering_log._path`` themselves (the last
patch wins).
"""

import pytest

from src.engine import clustering_log


@pytest.fixture(autouse=True)
def _isolate_clustering_log(tmp_path_factory, monkeypatch):
    target = tmp_path_factory.mktemp("clog") / "clustering_runs.jsonl"
    monkeypatch.setattr(clustering_log, "_path", target)
