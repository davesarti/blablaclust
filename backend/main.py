import uuid

from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Importiamo i tuoi modelli e la Base
from src.models import Base, ChatSession

# Configurazione db locale
engine = create_engine("sqlite:///./data/demo_database.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Crea le tabelle se non esistono
Base.metadata.create_all(bind=engine)

app = FastAPI()


class CreateSessionRequest(BaseModel):
    dataset_name: str

# Dependency per ottenere la sessione del DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/sessions")
def read_sessions(db: Session = Depends(get_db)):
    # Recupera tutte le sessioni dal database
    chat_sessions = db.query(ChatSession).all()
    return [
        {
            "id": session.id,
            "dataset_name": session.dataset_name,
            "embedding_model": session.embedding_model,
            "status": session.status,
        }
        for session in chat_sessions
    ]


@app.post("/sessions")
def create_session(payload: CreateSessionRequest, db: Session = Depends(get_db)):
    new_session = ChatSession(
        id=str(uuid.uuid4()),
        dataset_name=payload.dataset_name,
        embedding_model="default",
        status="active",
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return {
        "id": new_session.id,
        "dataset_name": new_session.dataset_name,
        "embedding_model": new_session.embedding_model,
        "status": new_session.status,
    }