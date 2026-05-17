import os

from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.models import Base

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "demo_database.db")
engine = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Router imports must come after get_db is defined so the circular reference resolves
from backend.routers.datasets import router as datasets_router
from backend.routers.sessions import router as sessions_router

app = FastAPI()
app.include_router(datasets_router)
app.include_router(sessions_router)
