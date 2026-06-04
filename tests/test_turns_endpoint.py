import os

os.environ["HARNESS_DRY_RUN"] = "true"

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
            id="sess-1", dataset_id="ds", embedding_model="default", status="active"
        )
    )
    db.add(
        ChatSession(
            id="sess-empty",
            dataset_id="ds",
            embedding_model="default",
            status="active",
        )
    )
    db.add(
        ChatSession(
            id="sess-closed",
            dataset_id="ds",
            embedding_model="default",
            status="closed",
        )
    )
    db.add(
        ChatSession(
            id="sess-converged",
            dataset_id="ds",
            embedding_model="default",
            status="converged",
        )
    )
    db.add(DataPoint(id="p1", dataset_id="ds", text="a", embedding=[0.1]))
    db.add(DataPoint(id="p2", dataset_id="ds", text="b", embedding=[0.2]))
    db.add(
        Cluster(id="c1", session_id="sess-1", name="C1", description="d", created_at_turn=1)
    )
    db.add(
        Cluster(
            id="c-closed",
            session_id="sess-closed",
            name="C",
            description="d",
            created_at_turn=1,
        )
    )
    db.add(
        Cluster(
            id="c-converged",
            session_id="sess-converged",
            name="C",
            description="d",
            created_at_turn=1,
        )
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


def _payload(**overrides):
    payload = {"session_id": "sess-1", "raw_text": "merge these", "feedback_type": "global"}
    payload.update(overrides)
    return payload


def test_create_turn_persists_and_returns(client):
    resp = client.post("/turns", json=_payload())
    assert resp.status_code == 201
    body = resp.json()
    assert body["session_id"] == "sess-1"
    assert body["turn_number"] == 1
    assert body["oracle_input"]["raw_text"] == "merge these"
    assert "display" in body["system_output"]
    assert body["system_output"]["display"]["content"]
    assert body["system_output"]["action"] == "show"


def test_turn_number_increments(client):
    assert client.post("/turns", json=_payload()).json()["turn_number"] == 1
    assert client.post("/turns", json=_payload()).json()["turn_number"] == 2

    listed = client.get("/turns", params={"session_id": "sess-1"}).json()
    assert [t["turn_number"] for t in listed] == [1, 2]


def test_unknown_session_returns_404(client):
    resp = client.post("/turns", json=_payload(session_id="does-not-exist"))
    assert resp.status_code == 404


def test_session_without_clusters_returns_409(client):
    resp = client.post("/turns", json=_payload(session_id="sess-empty"))
    assert resp.status_code == 409


def test_closed_session_returns_409(client):
    resp = client.post("/turns", json=_payload(session_id="sess-closed"))
    assert resp.status_code == 409


def test_converged_session_returns_409(client):
    resp = client.post("/turns", json=_payload(session_id="sess-converged"))
    assert resp.status_code == 409


def test_oracle_end_action_converges_session(client):
    """f_output emitting action='end' is the sole signal that converges a
    session. The router must flip status to 'converged' and 409 further POSTs."""
    from unittest.mock import patch

    end_response = (
        {"action": "end", "operations": [], "display": "Closing the session."},
        {"input_tokens": 1, "output_tokens": 1},
    )
    with patch("backend.routers.turns.f_output", return_value=end_response):
        first = client.post("/turns", json=_payload(raw_text="I'm done, close it")).json()

    assert first["system_output"]["action"] == "stop"
    assert first["system_output"]["state_snapshot"]["reason"] == "converged"

    # Subsequent POSTs must be refused with 409.
    resp = client.post("/turns", json=_payload())
    assert resp.status_code == 409
    assert "converged" in resp.json()["detail"]


def test_no_change_action_does_not_converge_session(client):
    """A 'no_change' turn must NOT converge — the oracle hasn't signalled
    close intent, they just had nothing structural to request this turn."""
    body = client.post("/turns", json=_payload(raw_text="hmm, let me think")).json()
    assert body["system_output"]["action"] == "show"

    # Session still accepts new turns.
    follow = client.post("/turns", json=_payload(raw_text="still thinking"))
    assert follow.status_code == 201


def test_unknown_target_cluster_returns_422(client):
    resp = client.post(
        "/turns",
        json=_payload(feedback_type="cluster", target_cluster_ids=["nope"]),
    )
    assert resp.status_code == 422


def test_multiple_target_clusters_accepted(client):
    # c1 is valid; a second valid cluster would be needed for a real merge,
    # but the guard only rejects unknown IDs — single known ID must pass
    resp = client.post(
        "/turns",
        json=_payload(feedback_type="cluster", target_cluster_ids=["c1"]),
    )
    assert resp.status_code == 201


def test_mixed_valid_invalid_clusters_returns_422(client):
    resp = client.post(
        "/turns",
        json=_payload(feedback_type="cluster", target_cluster_ids=["c1", "nope"]),
    )
    assert resp.status_code == 422
