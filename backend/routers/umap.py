"""UMAP projection endpoint (P2 — Data & Embeddings).

Serves a 2-D layout of a session's dataset plus the per-turn cluster
assignments, so the frontend can animate how the clustering evolves over the
conversation. The expensive reduction runs once per dataset and is cached in
process memory (embeddings are immutable for a dataset).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.main import get_db
from src.viz.umap_projection import project_session

router = APIRouter(prefix="/sessions", tags=["umap"])

# {dataset_name: (point_ids, coords, reducer_name)} — UMAP is dataset-stable.
_coords_cache: dict = {}


@router.get("/{session_id}/umap")
def get_session_umap(session_id: str, db: Session = Depends(get_db)) -> dict:
    """2-D projection of the session's points with per-turn cluster colouring."""
    try:
        return project_session(db, session_id, coords_cache=_coords_cache)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
