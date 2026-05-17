"""CSV dataset upload helpers."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
import uuid
from typing import IO, Iterable, Tuple

from sqlalchemy.orm import Session

from src.models import DataPoint
from src.dataset_processing.text_cleaning import clean_fields, clean_text

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


def ingest_csv_path(path: str, dataset_name: str, db: Session) -> Tuple[int, int]:
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
                dataset_name=dataset_name,
                data={
                    "label": label,
                    "title": title_clean,
                    "text": text_clean,
                },
            )
            db.add(dp)
            inserted += 1
    return inserted, skipped


def generate_embeddings_for_dataset(
    dataset_name: str,
    db: Session,
    model=None,
) -> int:
    data_points = (
        db.query(DataPoint)
        .filter(DataPoint.dataset_name == dataset_name, DataPoint.embedding == None)
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
) -> dict[str, int]:
    tmp_path = save_upload_to_tmp(upload_stream)
    try:
        inserted, skipped = ingest_csv_path(tmp_path, dataset_name, db)
        db.flush()
        embeddings_generated = 0
        if generate_embeddings:
            embeddings_generated = generate_embeddings_for_dataset(dataset_name, db)
        db.commit()
        return {
            "inserted": inserted,
            "skipped": skipped,
            "embeddings_generated": embeddings_generated,
        }
    except Exception:
        db.rollback()
        raise
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
