# Sprint 1 — P1 (Backend & Data Modeling)

## What I built:
- **Backend**: `backend/main.py` — FastAPI skeleton with endpoints to manage sessions, for now just GET and POST to create a new session. The POST endpoint generates a new UUID for the session, sets a default model and status, and saves it to the database
- **Schemas**: `src/schemas.py` — Pydantic models for request/response validation (early version, will be surely changed in the next weeks with what the team truly needs)
- **Models**: `src/models.py` — SQLAlchemy ORM definitions for database tables

## Challenges:
The fact that I had never used FastAPI made me quite slow, combined with the fact that I had little experience in backend development in general. SQLAlchemy was also new to me, but really intuitive, the main issue was about deciding the right data model and how to structure the database with the team. Pydantic models were pretty straightforward but are surely going to evolve as we understand better the data flow and the needs of the system.

## Next steps:
- Implement the logic to save and retrieve real sessions from the database in `backend/main.py`
- Add endpoints to upload and process the dataset
- Start working on the integration with the LLM layer (P4) to generate responses based on the session data