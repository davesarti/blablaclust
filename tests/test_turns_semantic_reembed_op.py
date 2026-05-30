"""Router tests for the new semantic_reembed operation.

The LLM (via f_output) is the single intent-classification step. When it emits
{"type": "semantic_reembed", "axis_label": "..."} the router runs the
re-embedding pipeline; for any other op type the normal apply-operations path
runs. f_output is mocked here so the tests are deterministic.
"""

import os

os.environ["HARNESS_DRY_RUN"] = "true"

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app, get_db
from src.models import Base, ChatSession, Cluster, DataPoint


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    db = TestingSessionLocal()
    db.add(
        ChatSession(
            id="s1", dataset_name="ds", embedding_model="default", status="active"
        )
    )
    db.add(DataPoint(id="p1", dataset_name="ds", data={"text": "a"}, embedding=[0.1, 0.2]))
    db.add(DataPoint(id="p2", dataset_name="ds", data={"text": "b"}, embedding=[0.3, 0.4]))
    db.add(
        Cluster(id="c1", session_id="s1", name="C1", description="d", created_at_turn=0)
    )
    db.commit()
    db.close()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def _payload(text: str = "anything"):
    return {
        "session_id": "s1",
        "raw_text": text,
        "feedback_type": "global",
    }


def _f_output_returns(operations: list[dict], display: str = "ok"):
    """Build a fake f_output that returns the given operations + a usage dict."""
    return lambda *a, **kw: (
        {"action": operations[0]["type"] if operations else "no_change",
         "operations": operations,
         "display": display},
        {"input_tokens": 1, "output_tokens": 1},
    )


def test_semantic_reembed_op_triggers_clustering_pipeline(client):
    """When f_output emits semantic_reembed, the router must call semantic_clustering."""
    fake_op = {"type": "semantic_reembed", "axis_label": "battery life"}

    with patch(
        "backend.routers.turns.f_output",
        _f_output_returns([fake_op], display="re-embedding along battery life"),
    ), patch(
        "backend.routers.turns.semantic_clustering",
        return_value=([], []),
    ) as mock_sc:
        resp = client.post("/turns", json=_payload("battery life"))

    assert resp.status_code == 201, resp.text
    assert mock_sc.called, "semantic_clustering should have been invoked"
    kwargs = mock_sc.call_args.kwargs
    assert kwargs["axis_hint"] == "battery life"
    assert kwargs["session_id"] == "s1"


def test_semantic_reembed_op_works_on_a_later_turn(client):
    """The re-embed path must NOT be gated on turn_number — it triggers whenever
    f_output emits the op, on Turn 1 or any later turn."""
    fake_op = {"type": "semantic_reembed", "axis_label": "tone"}

    # Turn 1: a normal no-op, just to advance turn_number
    with patch(
        "backend.routers.turns.f_output",
        _f_output_returns([], display="no change"),
    ), patch("backend.routers.turns.semantic_clustering") as mock_sc_turn1:
        resp1 = client.post("/turns", json=_payload("hello"))
    assert resp1.status_code == 201
    assert not mock_sc_turn1.called

    # Turn 2: f_output returns the re-embed op
    with patch(
        "backend.routers.turns.f_output",
        _f_output_returns([fake_op], display="re-embedding along tone"),
    ), patch(
        "backend.routers.turns.semantic_clustering", return_value=([], [])
    ) as mock_sc_turn2:
        resp2 = client.post("/turns", json=_payload("tone"))

    assert resp2.status_code == 201, resp2.text
    assert mock_sc_turn2.called, "semantic_clustering must fire on later turns too"
    assert mock_sc_turn2.call_args.kwargs["axis_hint"] == "tone"


def test_structural_op_on_turn_one_does_not_trigger_reembed(client):
    """Regression guard: a Turn-1 message routed by f_output into a normal op
    (here an empty list / no_change is enough) must NOT take the re-embed
    branch. Before the fix the router invoked semantic_clustering on every
    Turn-1 request because the UI auto-set axis_hint."""
    with patch(
        "backend.routers.turns.f_output",
        _f_output_returns([], display="no change"),
    ), patch("backend.routers.turns.semantic_clustering") as mock_sc:
        resp = client.post("/turns", json=_payload("merge cluster 1 and cluster 2"))

    assert resp.status_code == 201, resp.text
    assert not mock_sc.called, "structural Turn-1 must not trigger semantic_clustering"
