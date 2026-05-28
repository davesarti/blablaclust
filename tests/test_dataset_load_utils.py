"""Unit tests for src/dataset_processing/dataset_load_utils.py.

Runs against an in-memory SQLite database (same StaticPool pattern as
test_cluster_operations.py) so the real ingest + persistence path is exercised
without touching disk or loading the embedding model. Covers header
validation, label/text parsing edge cases, and transaction rollback.
"""

import io

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.dataset_processing import dataset_load_utils as dlu
from src.dataset_processing.dataset_load_utils import (
    _normalize_headers,
    _validate_headers,
    ingest_csv_path,
    process_csv_upload,
    save_upload_to_tmp,
)
from src.models import Base, DataPoint


@pytest.fixture
def db():
    """In-memory DB with the full schema created."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()
    yield session
    session.close()


def _write_csv(tmp_path, rows: str) -> str:
    """Write CSV content to a temp file and return its path."""
    path = tmp_path / "data.csv"
    path.write_text(rows, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# _normalize_headers
# ---------------------------------------------------------------------------


def test_normalize_headers_strips_and_sets():
    assert _normalize_headers([" label ", "title", "text "]) == {"label", "title", "text"}


def test_normalize_headers_none_returns_empty_set():
    assert _normalize_headers(None) == set()


def test_normalize_headers_skips_none_entries():
    assert _normalize_headers(["label", None, "text"]) == {"label", "text"}


# ---------------------------------------------------------------------------
# _validate_headers
# ---------------------------------------------------------------------------


def test_validate_headers_accepts_complete_set():
    # Should not raise.
    _validate_headers(["label", "title", "text"])


def test_validate_headers_accepts_extra_columns():
    _validate_headers(["label", "title", "text", "extra"])


def test_validate_headers_raises_on_missing():
    with pytest.raises(ValueError, match="Missing headers"):
        _validate_headers(["label", "title"])  # no "text"


def test_validate_headers_lists_all_missing_sorted():
    with pytest.raises(ValueError, match="text, title"):
        _validate_headers(["label"])  # missing both "text" and "title"


def test_validate_headers_none_raises():
    with pytest.raises(ValueError, match="Missing headers"):
        _validate_headers(None)


# ---------------------------------------------------------------------------
# ingest_csv_path — happy path + skip rules
# ---------------------------------------------------------------------------


def test_ingest_inserts_valid_rows(tmp_path, db):
    csv = "label,title,text\n1,Good,Great product\n0,Bad,Terrible item\n"
    path = _write_csv(tmp_path, csv)

    inserted, skipped = ingest_csv_path(path, "ds1", db)
    db.commit()

    assert inserted == 2
    assert skipped == 0
    rows = db.query(DataPoint).filter(DataPoint.dataset_name == "ds1").all()
    assert len(rows) == 2
    assert rows[0].data["label"] in (0, 1)
    assert rows[0].data["text"]  # cleaned, non-empty


def test_ingest_skips_empty_text_rows(tmp_path, db):
    # Second row has whitespace-only text → cleans to "" → skipped.
    csv = "label,title,text\n1,Good,Real text\n0,Title,   \n"
    path = _write_csv(tmp_path, csv)

    inserted, skipped = ingest_csv_path(path, "ds2", db)
    db.commit()

    assert inserted == 1
    assert skipped == 1


def test_ingest_skips_non_int_labels(tmp_path, db):
    csv = "label,title,text\nnot_a_number,Good,Real text\n2,Ok,Other text\n"
    path = _write_csv(tmp_path, csv)

    inserted, skipped = ingest_csv_path(path, "ds3", db)
    db.commit()

    assert inserted == 1   # only the row with label=2
    assert skipped == 1    # the "not_a_number" row


def test_ingest_skips_missing_label(tmp_path, db):
    # Empty label cell → None → skipped.
    csv = "label,title,text\n,Good,Real text\n1,Ok,Other text\n"
    path = _write_csv(tmp_path, csv)

    inserted, skipped = ingest_csv_path(path, "ds4", db)
    db.commit()

    assert inserted == 1
    assert skipped == 1


def test_ingest_cleans_fields_on_insert(tmp_path, db):
    csv = 'label,title,text\n1,Greaaaaat,Works    perfectly!!!!!\n'
    path = _write_csv(tmp_path, csv)

    ingest_csv_path(path, "ds5", db)
    db.commit()

    dp = db.query(DataPoint).filter(DataPoint.dataset_name == "ds5").first()
    assert dp.data["title"] == "Greaaat"
    assert dp.data["text"] == "Works perfectly!!!"


def test_ingest_raises_on_missing_headers(tmp_path, db):
    csv = "label,title\n1,Good\n"  # no "text" column
    path = _write_csv(tmp_path, csv)

    with pytest.raises(ValueError, match="Missing headers"):
        ingest_csv_path(path, "ds6", db)


# ---------------------------------------------------------------------------
# save_upload_to_tmp
# ---------------------------------------------------------------------------


def test_save_upload_to_tmp_writes_content():
    import os

    stream = io.BytesIO(b"label,title,text\n1,a,b\n")
    path = save_upload_to_tmp(stream)
    try:
        with open(path, "rb") as f:
            assert f.read() == b"label,title,text\n1,a,b\n"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# process_csv_upload — full flow + rollback
# ---------------------------------------------------------------------------


def test_process_csv_upload_inserts_without_embeddings(db):
    stream = io.BytesIO(b"label,title,text\n1,Good,Great product\n0,Bad,Bad item\n")
    result = process_csv_upload(stream, "up1", db, generate_embeddings=False)

    assert result["inserted"] == 2
    assert result["skipped"] == 0
    assert result["embeddings_generated"] == 0
    assert db.query(DataPoint).filter(DataPoint.dataset_name == "up1").count() == 2


def test_process_csv_upload_rolls_back_on_error(db, monkeypatch):
    """If embedding generation blows up, nothing is committed."""

    def boom(*args, **kwargs):
        raise RuntimeError("embedding model exploded")

    monkeypatch.setattr(dlu, "generate_embeddings_for_dataset", boom)

    stream = io.BytesIO(b"label,title,text\n1,Good,Great product\n")
    with pytest.raises(RuntimeError, match="embedding model exploded"):
        process_csv_upload(stream, "up2", db, generate_embeddings=True)

    # The inserted-but-not-committed row must have been rolled back.
    assert db.query(DataPoint).filter(DataPoint.dataset_name == "up2").count() == 0


def test_process_csv_upload_rolls_back_on_bad_headers(db):
    # Missing "text" header → ingest raises → rollback, nothing persisted.
    stream = io.BytesIO(b"label,title\n1,Good\n")
    with pytest.raises(ValueError, match="Missing headers"):
        process_csv_upload(stream, "up3", db, generate_embeddings=False)

    assert db.query(DataPoint).filter(DataPoint.dataset_name == "up3").count() == 0
