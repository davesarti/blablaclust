"""Utilities for ingesting a CSV dataset and generating embeddings.

Intended to back a POST /datasets upload endpoint. Functions do not commit —
callers own the transaction, except process_dataset which commits for convenience.
"""

import csv
import io
import uuid
from typing import IO

from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from src.models import DataPoint
from src.text_cleaning import clean_text

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64


def ingest_csv(file: IO[str], dataset_name: str, db: Session) -> int:
    """Parse a CSV file-like object (text mode) and insert DataPoint rows.

    Expects columns: label, title, text.
    Does not commit — caller is responsible for db.commit().
    Returns the number of rows inserted.
    """
    reader = csv.DictReader(file)
    count = 0
    for row in reader:
        dp = DataPoint(
            id=str(uuid.uuid4()),
            dataset_name=dataset_name,
            data={
                "label": int(row["label"]),
                "title": row["title"],
                "text": row["text"],
            },
        )
        db.add(dp)
        count += 1
    return count


def ingest_csv_bytes(content: bytes, dataset_name: str, db: Session) -> int:
    """Convenience wrapper for raw bytes (e.g. from UploadFile.read()).

    Decodes as UTF-8 and delegates to ingest_csv.
    """
    return ingest_csv(io.StringIO(content.decode("utf-8")), dataset_name, db)


def generate_embeddings(
    dataset_name: str,
    db: Session,
    model: SentenceTransformer | None = None,
) -> int:
    """Generate and store embeddings for all DataPoints in dataset_name that lack one.

    Loads the default model if none is provided.
    Does not commit — caller is responsible for db.commit().
    Returns the number of embeddings generated.
    """
    data_points = (
        db.query(DataPoint)
        .filter(DataPoint.dataset_name == dataset_name, DataPoint.embedding == None)
        .all()
    )
    if not data_points:
        return 0

    if model is None:
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


def process_dataset(
    content: bytes,
    dataset_name: str,
    db: Session,
    model: SentenceTransformer | None = None,
) -> dict:
    """Ingest a CSV (raw bytes) and generate embeddings in one call.

    Commits the transaction on success; rolls back on error.
    Returns {"dataset_name": str, "inserted": int, "embedded": int}.
    """
    try:
        inserted = ingest_csv_bytes(content, dataset_name, db)
        db.flush()
        embedded = generate_embeddings(dataset_name, db, model=model)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return {"dataset_name": dataset_name, "inserted": inserted, "embedded": embedded}
