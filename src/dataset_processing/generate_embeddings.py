"""Generate and store embeddings for all data points using Sentence Transformers."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer

from src.models import DataPoint

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
DB_PATH = "sqlite:///./data/demo_database.db"

engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


def main():
    db = SessionLocal()
    try:
        data_points = db.query(DataPoint).filter(DataPoint.embedding == None).all()
        print(f"Data points to embed: {len(data_points)}")

        if not data_points:
            print("Nothing to do.")
            return

        print(f"Loading model '{MODEL_NAME}'...")
        model = SentenceTransformer(MODEL_NAME)

        texts = [dp.text for dp in data_points]

        print(f"Generating embeddings in batches of {BATCH_SIZE}...")
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        for dp, vec in zip(data_points, embeddings):
            dp.embedding = vec.tolist()

        db.commit()
        print(f"Done. {len(data_points)} embeddings saved (dim={embeddings.shape[1]}).")

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
