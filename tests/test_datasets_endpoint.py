import os

os.environ["HARNESS_DRY_RUN"] = "true"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app, get_db
from src.models import Base, DataPoint


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
    # Mirror the production ingest pattern: rows without embeddings are
    # inserted without the embedding= kwarg (SQL NULL). Explicitly passing
    # embedding=None would store a JSON null literal instead, which
    # COUNT(embedding) treats as non-null — not what production sees.
    db.add(DataPoint(id="a1", dataset_name="alpha", data={"text": "x"}, embedding=[0.1]))
    db.add(DataPoint(id="a2", dataset_name="alpha", data={"text": "y"}, embedding=[0.2]))
    db.add(DataPoint(id="a3", dataset_name="alpha", data={"text": "z"}))
    db.add(DataPoint(id="b1", dataset_name="beta", data={"text": "q"}))
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


def test_list_datasets_returns_counts(client):
    res = client.get("/datasets")
    assert res.status_code == 200
    body = res.json()
    assert body == [
        {"dataset_name": "alpha", "n_points": 3, "has_embeddings": 2},
        {"dataset_name": "beta", "n_points": 1, "has_embeddings": 0},
    ]


def test_list_datasets_empty(client):
    # Wipe the seeded rows by deleting via the dataset DELETE endpoint, then
    # confirm the listing collapses to []. (Using a separate fixture-less call
    # would re-seed; this exercises the empty-grouping path.)
    assert client.delete("/datasets/alpha").status_code == 200
    assert client.delete("/datasets/beta").status_code == 200
    res = client.get("/datasets")
    assert res.status_code == 200
    assert res.json() == []
