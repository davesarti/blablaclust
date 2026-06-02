"""Load Amazon review data points into the SQLite database."""

import csv
import uuid
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models import Base, DataPoint, Dataset

DB_PATH = "sqlite:///./data/demo_database.db"
FILES = [
    ("data/train.csv", "amazon_reviews_train"),
    ("data/frozen_eval.csv", "amazon_reviews_eval"),
]

engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)


def _get_or_create_dataset(name: str, db) -> Dataset:
    existing = db.query(Dataset).filter(Dataset.name == name).one_or_none()
    if existing is not None:
        return existing
    ds = Dataset(id=str(uuid.uuid4()), name=name, description="")
    db.add(ds)
    db.flush()
    return ds


def load_csv(filepath: str, dataset_name: str, db) -> int:
    dataset = _get_or_create_dataset(dataset_name, db)
    count = 0
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dp = DataPoint(
                id=str(uuid.uuid4()),
                dataset_id=dataset.id,
                data={
                    "label": int(row["label"]),
                    "title": row["title"],
                    "text": row["text"],
                },
            )
            db.add(dp)
            count += 1
    return count


def main():
    db = SessionLocal()
    try:
        existing = db.query(DataPoint).count()
        if existing > 0:
            print(f"DB already has {existing} data points. Skipping (run with --force to overwrite).")
            return

        total = 0
        for filepath, dataset_name in FILES:
            if not Path(filepath).exists():
                print(f"File not found: {filepath}, skipping.")
                continue
            n = load_csv(filepath, dataset_name, db)
            print(f"  Loaded {n} rows from {filepath} as '{dataset_name}'")
            total += n

        db.commit()
        print(f"Done. {total} data points inserted.")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    if "--force" in sys.argv:
        db = SessionLocal()
        db.query(DataPoint).delete()
        db.query(Dataset).delete()
        db.commit()
        db.close()
        print("Cleared existing data points and datasets.")
    main()
