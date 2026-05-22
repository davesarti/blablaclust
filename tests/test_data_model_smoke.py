"""Smoke test for the data model (``src/models.py``).

Not a deep behavioural test — it proves the schema is internally consistent and
wired the way ``docs/data-model.md`` describes:

* every table builds and a full related graph (session → cluster → data point →
  soft assignment, plus a turn) persists and round-trips through relationships;
* each ``CheckConstraint`` actually rejects the bad value it names;
* deleting a session cascades to its clusters, turns, and soft assignments.

Runs entirely against in-memory SQLite — no LLM, no network, no demo DB.
"""

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.models import (
    Base,
    ChatSession,
    Cluster,
    DataPoint,
    SoftAssignment,
    Turn,
)


@pytest.fixture
def db():
    """In-memory DB with all tables created and SQLite FK enforcement on."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # SQLite ignores foreign keys unless asked; turn them on so FK-level
    # behaviour matches the constraints declared in the model.
    @event.listens_for(engine, "connect")
    def _fk_pragma(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = TestingSession()
    yield session
    session.close()


def _seed_full_graph(db: Session) -> None:
    """Insert one of every row type, fully linked: the canonical happy path."""
    session = ChatSession(
        id="sess-1", dataset_name="ds", embedding_model="default", status="active"
    )
    point = DataPoint(id="dp-1", dataset_name="ds", data={"text": "hi"}, embedding=[0.1, 0.2])
    cluster = Cluster(
        id="cl-1",
        session_id="sess-1",
        name="Cluster 1",
        description="",
        created_at_turn=0,
    )
    assignment = SoftAssignment(
        data_point_id="dp-1", cluster_id="cl-1", turn_number=0, probability=1.0
    )
    turn = Turn(
        session_id="sess-1",
        turn_number=1,
        oracle_input={"text": "merge these"},
        system_output={"display": "done"},
    )
    db.add_all([session, point, cluster, assignment, turn])
    db.commit()


def test_full_graph_persists_and_round_trips(db):
    _seed_full_graph(db)
    db.expire_all()  # force a fresh read from the DB, not the identity map

    session = db.query(ChatSession).one()
    # Relationships navigate in both directions.
    assert {c.id for c in session.clusters} == {"cl-1"}
    assert {t.turn_number for t in session.turns} == {1}
    cluster = session.clusters[0]
    assert cluster.session is session
    assert cluster.soft_assignments[0].probability == 1.0
    assert cluster.soft_assignments[0].data_point.id == "dp-1"
    # JSON columns round-trip as dict / list.
    assert db.query(DataPoint).one().data == {"text": "hi"}
    assert db.query(Turn).one().oracle_input == {"text": "merge these"}


def test_deleting_session_cascades_to_children(db):
    _seed_full_graph(db)

    db.delete(db.query(ChatSession).one())
    db.commit()

    # The session's clusters, turns, and soft assignments are gone with it.
    assert db.query(Cluster).count() == 0
    assert db.query(Turn).count() == 0
    assert db.query(SoftAssignment).count() == 0
    # Data points are dataset-scoped, not session-scoped — they survive.
    assert db.query(DataPoint).count() == 1


@pytest.mark.parametrize(
    "make_row, why",
    [
        (
            lambda: ChatSession(
                id="bad", dataset_name="ds", embedding_model="m", status="bogus"
            ),
            "ck_sessions_status: status must be active/converged/closed",
        ),
        (
            lambda: Turn(
                session_id="sess-1", turn_number=-1, oracle_input={}, system_output={}
            ),
            "ck_turns_turn_number: turn number must be >= 0",
        ),
        (
            lambda: SoftAssignment(
                data_point_id="dp-1", cluster_id="cl-1", turn_number=0, probability=1.5
            ),
            "ck_soft_assignments_probability: probability must be in [0, 1]",
        ),
        (
            lambda: SoftAssignment(
                data_point_id="dp-1", cluster_id="cl-1", turn_number=-1, probability=0.5
            ),
            "ck_soft_assignments_turn_number: snapshot turn must be >= 0",
        ),
        (
            lambda: Cluster(
                id="bad",
                session_id="sess-1",
                name="c",
                description="",
                created_at_turn=5,
                dissolved_at_turn=2,
            ),
            "ck_clusters_turn_order: cannot dissolve before it was created",
        ),
    ],
)
def test_check_constraints_reject_bad_rows(db, make_row, why):
    # Seed the parent session/cluster/point the bad rows reference.
    _seed_full_graph(db)

    db.add(make_row())
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
