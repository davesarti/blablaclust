"""Verify database contents and embedding correctness."""

import json
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer

from src.models import DataPoint
from dataset_processing.text_cleaning import clean_text

DB_PATH = "sqlite:///./data/demo_database.db"
MODEL_NAME = "all-MiniLM-L6-v2"

engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


def verify_database(db):
    print("=" * 50)
    print("DATABASE CHECK")
    print("=" * 50)

    total = db.query(DataPoint).count()
    print(f"Total data points : {total}")

    for name in ["amazon_reviews_train", "amazon_reviews_eval"]:
        n = db.query(DataPoint).filter(DataPoint.dataset_name == name).count()
        print(f"  {name}: {n}")

    missing_emb = db.query(DataPoint).filter(DataPoint.embedding == None).count()
    print(f"Missing embeddings : {missing_emb}")

    sample = db.query(DataPoint).first()
    print(f"\nSample data point:")
    print(f"  id           : {sample.id}")
    print(f"  dataset_name : {sample.dataset_name}")
    print(f"  label        : {sample.data['label']}")
    print(f"  title        : {sample.data['title'][:60]}")
    print(f"  text         : {sample.data['text'][:80]}...")
    print(f"  embedding dim: {len(sample.embedding)}")


def verify_embeddings(db):
    print("\n" + "=" * 50)
    print("EMBEDDING CHECK")
    print("=" * 50)

    # 1. Dimensionality and value range
    sample = db.query(DataPoint).first()
    vec = np.array(sample.embedding)
    print(f"Vector dim   : {len(vec)}")
    print(f"Value range  : [{vec.min():.4f}, {vec.max():.4f}]")
    print(f"Norm (L2)    : {np.linalg.norm(vec):.4f}  (expected ~1.0 if normalized)")

    # 2. Regenerate one embedding and compare with stored
    print("\nConsistency check (regenerate 1 embedding and compare)...")
    model = SentenceTransformer(MODEL_NAME)
    text = clean_text(sample.data["title"], sample.data["text"])
    fresh_vec = model.encode(text, convert_to_numpy=True)
    cosine_sim = np.dot(vec, fresh_vec) / (np.linalg.norm(vec) * np.linalg.norm(fresh_vec))
    print(f"  Cosine similarity stored vs regenerated: {cosine_sim:.6f}  (expected ~1.0)")

    # 3. Semantic sanity check — similar reviews should be closer than opposite ones
    print("\nSemantic sanity check...")
    positive = db.query(DataPoint).filter(DataPoint.data["label"].as_integer() == 2).limit(50).all()
    negative = db.query(DataPoint).filter(DataPoint.data["label"].as_integer() == 1).limit(50).all()

    pos_vecs = np.array([dp.embedding for dp in positive])
    neg_vecs = np.array([dp.embedding for dp in negative])

    # centroid of positives vs centroid of negatives
    pos_centroid = pos_vecs.mean(axis=0)
    neg_centroid = neg_vecs.mean(axis=0)
    cross_sim = np.dot(pos_centroid, neg_centroid) / (
        np.linalg.norm(pos_centroid) * np.linalg.norm(neg_centroid)
    )

    # avg intra-class similarity (positives among themselves)
    intra_sim = np.mean(pos_vecs @ pos_centroid / (
        np.linalg.norm(pos_vecs, axis=1) * np.linalg.norm(pos_centroid)
    ))

    print(f"  Intra-class similarity (positive reviews): {intra_sim:.4f}")
    print(f"  Cross-class similarity (pos vs neg centroid): {cross_sim:.4f}")
    print(f"  -> {'OK' if intra_sim > cross_sim else 'WARN: cross > intra, check embeddings'}")


if __name__ == "__main__":
    db = SessionLocal()
    try:
        verify_database(db)
        verify_embeddings(db)
    finally:
        db.close()
