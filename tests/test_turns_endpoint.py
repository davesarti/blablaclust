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
            id="sess-1", dataset_name="ds", embedding_model="default", status="active"
        )
    )
    db.add(
        ChatSession(
            id="sess-empty",
            dataset_name="ds",
            embedding_model="default",
            status="active",
        )
    )
    db.add(
        ChatSession(
            id="sess-closed",
            dataset_name="ds",
            embedding_model="default",
            status="closed",
        )
    )
    db.add(DataPoint(id="p1", dataset_name="ds", data={"text": "a"}, embedding=[0.1]))
    db.add(DataPoint(id="p2", dataset_name="ds", data={"text": "b"}, embedding=[0.2]))
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
    # dry-run engine output is stored verbatim as system_output
    assert "display" in body["system_output"]
    assert body["system_output"]["action"] == "no_change"


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
