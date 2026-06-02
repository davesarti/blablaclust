"""CSV dataset upload helpers."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
import uuid
from typing import IO, Iterable, Tuple

from sqlalchemy.orm import Session

from src.models import DataPoint, Dataset
from src.dataset_processing.text_cleaning import clean_fields, clean_text
from src.dataset_processing.dataset_description import generate_dataset_description

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64
EXPECTED_HEADERS = {"label", "title", "text"}


def _normalize_headers(headers: Iterable[str] | None) -> set[str]:
    if not headers:
        return set()
    return {header.strip() for header in headers if header is not None}


def _validate_headers(headers: Iterable[str] | None) -> None:
    header_set = _normalize_headers(headers)
    missing = EXPECTED_HEADERS - header_set
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing headers: {missing_list}")


def save_upload_to_tmp(upload_stream: IO[bytes], suffix: str = ".csv") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        if hasattr(upload_stream, "seek"):
            upload_stream.seek(0)
        with tmp as handle:
            shutil.copyfileobj(upload_stream, handle)
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    return tmp.name


def get_or_create_dataset(name: str, db: Session) -> Dataset:
    """Look up a Dataset by name, creating it if missing.

    Raises ``ValueError`` if a Dataset with the given name already has
    DataPoints attached — used by upload to reject duplicate uploads of
    the same name without silently appending to the existing one.
    """
    existing = db.query(Dataset).filter(Dataset.name == name).one_or_none()
    if existing is not None:
        has_points = (
            db.query(DataPoint.id)
            .filter(DataPoint.dataset_id == existing.id)
            .first()
            is not None
        )
        if has_points:
            raise ValueError(
                f"A dataset named '{name}' already exists. Choose a different "
                f"name or delete the existing one first."
            )
        return existing
    ds = Dataset(id=str(uuid.uuid4()), name=name, description="")
    db.add(ds)
    db.flush()
    return ds


def ingest_csv_path(path: str, dataset_id: str, db: Session) -> Tuple[int, int]:
    inserted = 0
    skipped = 0
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _validate_headers(reader.fieldnames)
        for row in reader:
            raw_title = row.get("title") or ""
            raw_text = row.get("text") or ""
            title_clean, text_clean = clean_fields(raw_title, raw_text)
            if not text_clean:
                skipped += 1
                continue
            label_raw = row.get("label")
            try:
                label = int(label_raw) if label_raw is not None else None
            except ValueError:
                label = None
            if label is None:
                skipped += 1
                continue
            dp = DataPoint(
                id=str(uuid.uuid4()),
                dataset_id=dataset_id,
                data={
                    "label": label,
                    "title": title_clean,
                    "text": text_clean,
                },
            )
            db.add(dp)
            inserted += 1
    return inserted, skipped


def iter_generate_embeddings_for_dataset(
    dataset_id: str,
    db: Session,
    model=None,
):
    """Batch-by-batch embedding generator yielding ``(done, total)`` tuples.

    The first yield fires before the model loads (``done=0``) so callers can show
    a "preparing" state; subsequent yields fire after each ``EMBEDDING_BATCH_SIZE``
    chunk is encoded and persisted onto its DataPoint. The DB session is mutated
    in place — the caller is responsible for committing.
    """
    data_points = (
        db.query(DataPoint)
        .filter(DataPoint.dataset_id == dataset_id, DataPoint.embedding == None)
        .all()
    )
    total = len(data_points)
    yield (0, total)
    if total == 0:
        return

    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [clean_text(dp.data["title"], dp.data["text"]) for dp in data_points]
    done = 0
    for i in range(0, total, EMBEDDING_BATCH_SIZE):
        chunk_texts = texts[i : i + EMBEDDING_BATCH_SIZE]
        chunk_dps = data_points[i : i + EMBEDDING_BATCH_SIZE]
        vecs = model.encode(
            chunk_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        for dp, vec in zip(chunk_dps, vecs):
            dp.embedding = vec.tolist()
        done += len(chunk_dps)
        yield (done, total)


def generate_embeddings_for_dataset(
    dataset_id: str,
    db: Session,
    model=None,
) -> int:
    data_points = (
        db.query(DataPoint)
        .filter(DataPoint.dataset_id == dataset_id, DataPoint.embedding == None)
        .all()
    )
    if not data_points:
        return 0

    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [clean_text(dp.data["title"], dp.data["text"]) for dp in data_points]
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    for dp, vec in zip(data_points, embeddings):
        dp.embedding = vec.tolist()

    return len(data_points)


def process_csv_upload(
    upload_stream: IO[bytes],
    dataset_name: str,
    db: Session,
    generate_embeddings: bool = True,
) -> dict[str, object]:
    """Ingest a CSV under ``dataset_name``, embed, and generate a description.

    Returns ``{dataset_id, inserted, skipped, embeddings_generated, description}``.
    Rejects duplicate-name uploads via :func:`get_or_create_dataset`.
    """
    tmp_path = save_upload_to_tmp(upload_stream)
    try:
        dataset = get_or_create_dataset(dataset_name, db)
        inserted, skipped = ingest_csv_path(tmp_path, dataset.id, db)
        db.flush()
        embeddings_generated = 0
        if generate_embeddings:
            embeddings_generated = generate_embeddings_for_dataset(dataset.id, db)
        description = generate_dataset_description(dataset.id, db)
        db.commit()
        return {
            "dataset_id": dataset.id,
            "inserted": inserted,
            "skipped": skipped,
            "embeddings_generated": embeddings_generated,
            "description": description,
        }
    except Exception:
        db.rollback()
        raise
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
