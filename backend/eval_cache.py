"""Content-hash cache for `/sessions/{id}/eval` responses.

The eval endpoint makes four LLM calls (B2/B3/B4/B1). The deterministic A1/A2/A3
metrics are cheap, but they ride in the same response so we cache the whole
record. The key is a SHA256 over the exact payloads the LLM judges see, plus
the prompt-file hashes and the model name — so any change to clusters, turns,
feedback, prompts, or model invalidates the entry automatically.
"""

import hashlib
import json

from sqlalchemy.orm import Session

from src.harness import hash_prompt
from src.harness.harness_claude import DEFAULT_MODEL
from src.models import EvalCache
from src.schemas import ChatSessionState


_PROMPT_NAMES = (
    "f_eval_coherence",
    "f_eval_compliance",
    "f_eval_contradiction",
    "f_eval_overall",
)


def compute_cache_key(
    state: ChatSessionState,
    coherence_samples: list[dict],
    compliance_turns: list[dict],
) -> str:
    """SHA256 over the canonical bundle of judge inputs + prompts + model."""
    clusters = sorted(
        (
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "size": c.size,
                "representative_points": c.representative_points,
            }
            for c in state.clusters
        ),
        key=lambda c: c["id"],
    )

    coherence_payload = sorted(
        (
            {
                "cluster_id": s["cluster"].id,
                "top_texts": s["top_texts"],
                "bottom_texts": s["bottom_texts"],
            }
            for s in coherence_samples
        ),
        key=lambda s: s["cluster_id"],
    )

    bundle = {
        "clusters": clusters,
        "feedback_history": [f.model_dump() for f in state.feedback_history],
        "compliance_turns": compliance_turns,
        "coherence_samples": coherence_payload,
        "prompt_hashes": {name: hash_prompt(name) for name in _PROMPT_NAMES},
        "model": DEFAULT_MODEL,
    }
    canonical = json.dumps(bundle, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get(db: Session, key: str) -> dict | None:
    row = db.query(EvalCache).filter(EvalCache.key == key).first()
    if row is None:
        return None
    return json.loads(row.response_json)


def put(db: Session, key: str, session_id: str, response: dict) -> None:
    row = EvalCache(
        key=key,
        session_id=session_id,
        response_json=json.dumps(response),
    )
    db.merge(row)
    db.commit()
