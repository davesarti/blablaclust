import os

from fastapi import FastAPI
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from src.models import Base

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "demo_database.db")
engine = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def _run_migrations() -> None:
    """Add columns that exist in the ORM but may be missing from an older DB.

    SQLAlchemy's create_all only creates missing *tables*, not missing
    *columns* on existing tables. This function bridges the gap for the
    columns added after the initial schema was deployed. It is idempotent —
    safe to call on every startup.
    """
    with engine.connect() as conn:
        insp = inspect(engine)
        sessions_cols = {c["name"] for c in insp.get_columns("sessions")}
        if "preference_summary" not in sessions_cols:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN preference_summary TEXT"))
            conn.commit()


_run_migrations()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Router imports must come after get_db is defined so the circular reference resolves
from backend.routers.clusters import router as clusters_router
from backend.routers.datasets import router as datasets_router
from backend.routers.sessions import router as sessions_router
from backend.routers.turns import router as turns_router
from backend.routers.umap import router as umap_router

app = FastAPI()
app.include_router(datasets_router)
app.include_router(sessions_router)
app.include_router(turns_router)
app.include_router(clusters_router)
app.include_router(umap_router)
