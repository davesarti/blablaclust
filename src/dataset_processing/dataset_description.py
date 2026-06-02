"""LLM-generated one-sentence description for an uploaded dataset.

Mirrors the cluster_naming pattern: best-effort LLM call, never aborts the
upload. Mutates the Dataset row in place — the row must already exist.
"""

from __future__ import annotations

import random

from sqlalchemy.orm import Session

from src.harness import call_llm, render_prompt, loads_llm_json
from src.logger import log
from src.models import DataPoint, Dataset

SAMPLE_SIZE = 12
MAX_TEXT_CHARS = 400


def _point_text(dp: DataPoint) -> str:
    title = (dp.data or {}).get("title", "") or ""
    text = (dp.data or {}).get("text", "") or ""
    combined = f"{title} {text}".strip()
    if len(combined) > MAX_TEXT_CHARS:
        combined = combined[:MAX_TEXT_CHARS].rstrip() + "…"
    return combined


def generate_dataset_description(
    dataset_id: str,
    db: Session,
    sample_size: int = SAMPLE_SIZE,
) -> str:
    """Generate and persist a one-sentence description for the dataset.

    Picks a random sample of points, asks the LLM for a one-sentence
    description, and saves it onto the ``Dataset`` row (which the caller
    must have created). Returns the description on success or an empty
    string on failure — never raises.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
    if dataset is None:
        return ""

    points = (
        db.query(DataPoint)
        .filter(DataPoint.dataset_id == dataset_id)
        .all()
    )
    if not points:
        return ""

    if len(points) > sample_size:
        points = random.sample(points, sample_size)

    sample_texts = [t for t in (_point_text(dp) for dp in points) if t]
    if not sample_texts:
        return ""

    sample_block = "\n".join(f"- {t}" for t in sample_texts)
    prompt = render_prompt("dataset_description", sample_block=sample_block)

    description = ""
    response = None
    try:
        response = call_llm(
            [{"role": "user", "content": "Describe the dataset."}],
            system=prompt,
        )
        parsed = loads_llm_json(response.text)
        if isinstance(parsed, dict):
            raw = parsed.get("description")
            if raw:
                description = str(raw).strip()[:500]
    except Exception as e:
        snippet = repr(response.text[:500]) if response is not None else "<no response>"
        log.warning(
            f"dataset_description: LLM call failed or returned invalid JSON, "
            f"keeping empty description for '{dataset.name}'. Error: {e}. "
            f"Raw response: {snippet}"
        )

    dataset.description = description
    return description
