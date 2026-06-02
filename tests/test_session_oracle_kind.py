"""ChatSession schema: oracle_kind / persona_snapshot defaults + persistence."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models import Base, ChatSession


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_legacy_session_defaults_to_human():
    db = _make_session()
    s = ChatSession(
        id="s1",
        name="legacy",
        dataset_name="ds",
        embedding_model="default",
        status="active",
    )
    db.add(s)
    db.commit()

    fetched = db.query(ChatSession).filter(ChatSession.id == "s1").one()
    assert fetched.oracle_kind == "human"
    assert fetched.persona_snapshot is None


def test_persona_session_round_trips_snapshot():
    db = _make_session()
    snapshot = {"name": "p", "goal": "g", "notes": {"tone": "x"}}
    s = ChatSession(
        id="s2",
        name="persona/p",
        dataset_name="ds",
        embedding_model="default",
        status="active",
        oracle_kind="persona",
        persona_snapshot=snapshot,
    )
    db.add(s)
    db.commit()

    fetched = db.query(ChatSession).filter(ChatSession.id == "s2").one()
    assert fetched.oracle_kind == "persona"
    assert fetched.persona_snapshot == snapshot
