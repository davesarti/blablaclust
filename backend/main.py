from fastapi import FastAPI, Depends
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
    chatSessions = db.query(ChatSession).all()
    return chatSessions