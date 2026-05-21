"""
Run with:
    conda activate vibe-coders
    cd vibe-coders
    PYTHONPATH=. python scripts/serve_ui.py

UI:  http://localhost:8000/ui
API: http://localhost:8000/docs
"""
import pathlib
import uvicorn
from fastapi.responses import FileResponse
from backend.main import app

UI_PATH   = pathlib.Path(__file__).parent.parent / "ui" / "index.html"

# ---------------------------------------------------------------------------
# Auto-seed: Amazon Reviews dataset
#
# This block is specific to the demo dataset bundled with this repository:
#   data/train.csv  — 1 200 Amazon Electronics reviews (label, title, text)
#
# On the FIRST run the DB is empty, so we load the CSV and generate
# sentence-transformer embeddings (all-MiniLM-L6-v2, ~60 s on CPU).
# On every subsequent run the DB already has rows so we skip silently.
#
# If you want to use a DIFFERENT dataset, either:
#   (a) upload it via POST /datasets/upload (handles insert + embed in one call), or
#   (b) replace TRAIN_CSV / DATASET_NAME below and keep this auto-seed logic.
#
# The dataset_name stored in the DB ("amazon_reviews") is what the UI dropdown
# will display — it must match the value selected when creating a session.
# ---------------------------------------------------------------------------
TRAIN_CSV    = pathlib.Path(__file__).parent.parent / "data" / "train.csv"
DATASET_NAME = "amazon_reviews"   # must match what the UI sends to POST /sessions


def _auto_seed():
    """Populate the DB from train.csv if it is empty, then generate embeddings.

    This is idempotent: if the DB already contains data points (from a previous
    run or a manual upload), this function exits immediately without touching
    anything.  It only runs synchronously at startup, before uvicorn begins
    accepting requests, so there is no race condition with live traffic.
    """
    from src.models import DataPoint
    from src.dataset_processing.dataset_load_utils import (
        ingest_csv_path,
        generate_embeddings_for_dataset,
    )
    from backend.main import SessionLocal

    if not TRAIN_CSV.exists():
        # The CSV ships with the repo.  If it is missing, warn and continue —
        # the server will still start; the user must upload a dataset manually.
        print(f"[seed] {TRAIN_CSV} not found — skipping auto-seed.")
        return

    db = SessionLocal()
    try:
        if db.query(DataPoint).count() > 0:
            # Already seeded (or manually uploaded).  Nothing to do.
            print("[seed] DB already populated — skipping.")
            return

        print(f"[seed] Loading '{DATASET_NAME}' from {TRAIN_CSV} …")
        inserted, skipped = ingest_csv_path(str(TRAIN_CSV), DATASET_NAME, db)
        db.flush()   # make rows visible to the embedding query below
        print(f"[seed] {inserted} rows inserted, {skipped} skipped.")

        # Embedding generation is the slow step (~60 s on CPU for 1 200 rows).
        # The model (all-MiniLM-L6-v2) is downloaded automatically by
        # sentence-transformers on the first call and cached locally afterward.
        print("[seed] Generating embeddings (first run: ~60 s, model auto-downloads) …")
        n = generate_embeddings_for_dataset(DATASET_NAME, db)
        db.commit()
        print(f"[seed] Done — {n} embeddings stored.  Ready.")
    except Exception as exc:
        db.rollback()
        print(f"[seed] ERROR during seed: {exc}")
        raise
    finally:
        db.close()


@app.get("/ui", include_in_schema=False)
def serve_ui():
    return FileResponse(UI_PATH, media_type="text/html")


if __name__ == "__main__":
    _auto_seed()   # no-op after the first run
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
