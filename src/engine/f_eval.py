import json

from src.schemas import ChatSessionState
from src.harness import render_prompt, call_llm, hash_prompt, estimate_cost_usd, extract_json_text
from src.logger import log_llm_call


def f_eval(
    state: ChatSessionState,
    total_points: int = 0,
) -> dict:
    total = total_points or sum(c.size for c in state.clusters)

    prompt = render_prompt(
        "f_eval",
        session_id=state.session_id,
        turn_number=state.turn_number,
        total_points=total,
        clusters_json=json.dumps([c.model_dump() for c in state.clusters]),
        feedback_history_json=json.dumps([f.model_dump() for f in state.feedback_history]),
    )

    msg = call_llm([{"role": "user", "content": prompt}], system="")
    log_llm_call(
        session_id=state.session_id,
        prompt_name="f_eval",
        prompt_hash=hash_prompt("f_eval"),
        usage=msg.usage,
        cost_usd=estimate_cost_usd(msg.usage, msg.model),
        turn_number=state.turn_number,
        model=msg.model,
    )

    return json.loads(extract_json_text(msg.text))


def f_eval_coherence(
    state: ChatSessionState,
    cluster_samples: list[dict],
) -> list[dict]:
    """B1a — per-cluster internal coherence (LLM judge, batched).

    Each item in `cluster_samples` must have keys:
        cluster (Cluster), top_texts (list[str]), bottom_texts (list[str]).
    `top_texts` are the highest-probability members, `bottom_texts` the
    lowest-probability edge cases (the stress test).

    Returns one dict per cluster {cluster_id, coherence, reasoning} in the same
    order as the input. Missing/malformed entries default to coherence=0.0.
    """
    blocks = []
    for item in cluster_samples:
        c = item["cluster"]
        top = "\n".join(f"  - {t}" for t in item["top_texts"]) or "  (none)"
        bottom = "\n".join(f"  - {t}" for t in item["bottom_texts"]) or "  (none)"
        blocks.append(
            f"ID: {c.id}\n"
            f"Name: {c.name}\n"
            f"Description: {c.description or ''}\n"
            f"Size: {c.size} points\n"
            f"top members:\n{top}\n"
            f"bottom members:\n{bottom}"
        )
    clusters_block = "\n\n".join(blocks)

    prompt = render_prompt(
        "f_eval_coherence",
        session_id=state.session_id,
        turn_number=state.turn_number,
        feedback_history_json=json.dumps([f.model_dump() for f in state.feedback_history]),
        clusters_block=clusters_block,
    )

    msg = call_llm([{"role": "user", "content": prompt}], system="", max_tokens=8192)
    log_llm_call(
        session_id=state.session_id,
        prompt_name="f_eval_coherence",
        prompt_hash=hash_prompt("f_eval_coherence"),
        usage=msg.usage,
        cost_usd=estimate_cost_usd(msg.usage, msg.model),
        turn_number=state.turn_number,
        model=msg.model,
    )

    try:
        parsed = json.loads(extract_json_text(msg.text))
        by_id = {r.get("cluster_id"): r for r in parsed.get("results", [])}
    except (json.JSONDecodeError, ValueError):
        # Truncated or malformed judge output (e.g., hit max_tokens). Don't
        # crash the whole eval — fall back to neutral scores per cluster.
        by_id = {}

    results = []
    for item in cluster_samples:
        cid = item["cluster"].id
        r = by_id.get(cid, {})
        results.append({
            "cluster_id": cid,
            "cluster_name": item["cluster"].name,
            "coherence": float(r.get("coherence", 0.0)),
            "reasoning": str(r.get("reasoning", "")),
        })
    return results


def f_eval_compliance(
    state: ChatSessionState,
    clusters_block: str,
    turns: list[dict],
) -> dict:
    """B3 — oracle compliance (LLM judge).

    Judges how faithfully the system translated each oracle request into
    operations. Oracle clarity is judged separately by f_eval_contradiction.

    Args:
        clusters_block: pre-rendered text describing the final clusters.
        turns: list of {turn, oracle_request, operations}, one per conversation turn.

    Returns {compliance_score, notes}.
    """
    prompt = render_prompt(
        "f_eval_compliance",
        session_id=state.session_id,
        turn_number=state.turn_number,
        clusters_block=clusters_block,
        turns_json=json.dumps(turns, indent=2),
    )

    msg = call_llm([{"role": "user", "content": prompt}], system="", max_tokens=2048)
    log_llm_call(
        session_id=state.session_id,
        prompt_name="f_eval_compliance",
        prompt_hash=hash_prompt("f_eval_compliance"),
        usage=msg.usage,
        cost_usd=estimate_cost_usd(msg.usage, msg.model),
        turn_number=state.turn_number,
        model=msg.model,
    )

    try:
        parsed = json.loads(extract_json_text(msg.text))
    except (json.JSONDecodeError, ValueError):
        parsed = {}
    return {
        "compliance_score": float(parsed.get("compliance_score", 0.0)),
        "notes": str(parsed.get("notes", "")),
    }


def f_eval_contradiction(state: ChatSessionState) -> dict:
    """B4 — oracle contradiction detection (LLM judge).

    Assesses how hard it would have been for the system to understand the
    oracle's intent: contradictions, drift, ambiguity, vague targets. The
    overall synthesis uses this to forgive low coherence/compliance when the
    oracle itself was the obstacle.

    Returns {contradiction_score, notes, examples}.
    """
    prompt = render_prompt(
        "f_eval_contradiction",
        session_id=state.session_id,
        turn_number=state.turn_number,
        feedback_history_json=json.dumps(
            [f.model_dump() for f in state.feedback_history]
        ),
    )

    msg = call_llm([{"role": "user", "content": prompt}], system="", max_tokens=2048)
    log_llm_call(
        session_id=state.session_id,
        prompt_name="f_eval_contradiction",
        prompt_hash=hash_prompt("f_eval_contradiction"),
        usage=msg.usage,
        cost_usd=estimate_cost_usd(msg.usage, msg.model),
        turn_number=state.turn_number,
        model=msg.model,
    )

    try:
        parsed = json.loads(extract_json_text(msg.text))
    except (json.JSONDecodeError, ValueError):
        parsed = {}
    examples = parsed.get("examples") or []
    if not isinstance(examples, list):
        examples = []
    return {
        "contradiction_score": float(parsed.get("contradiction_score", 0.0)),
        "notes": str(parsed.get("notes", "")),
        "examples": [str(e) for e in examples],
    }


def f_eval_overall(
    state: ChatSessionState,
    coherence_results: list[dict],
    coherence_mean: float,
    coherence_min: float,
    compliance: dict,
    contradiction: dict,
) -> dict:
    """B1 overall — synthesis judge.

    Combines coherence (B2), compliance (B3), and contradiction (B4) into one
    verdict. A high contradiction score forgives low coherence/compliance
    ("the system did its best with a self-contradictory oracle").

    Returns {overall_score, notes}.
    """
    coherence_block = "\n".join(
        f"- {r.get('cluster_name') or 'unnamed'}: {r['coherence']:.2f} — {r['reasoning']}"
        for r in coherence_results
    ) or "(no clusters)"

    prompt = render_prompt(
        "f_eval_overall",
        session_id=state.session_id,
        turn_number=state.turn_number,
        coherence_mean=f"{coherence_mean:.3f}",
        coherence_min=f"{coherence_min:.3f}",
        coherence_block=coherence_block,
        compliance_score=f"{compliance['compliance_score']:.3f}",
        compliance_notes=compliance["notes"],
        contradiction_score=f"{contradiction['contradiction_score']:.3f}",
        contradiction_notes=contradiction["notes"],
    )

    msg = call_llm([{"role": "user", "content": prompt}], system="")
    log_llm_call(
        session_id=state.session_id,
        prompt_name="f_eval_overall",
        prompt_hash=hash_prompt("f_eval_overall"),
        usage=msg.usage,
        cost_usd=estimate_cost_usd(msg.usage, msg.model),
        turn_number=state.turn_number,
        model=msg.model,
    )

    try:
        parsed = json.loads(extract_json_text(msg.text))
    except (json.JSONDecodeError, ValueError):
        parsed = {}
    return {
        "overall_score": float(parsed.get("overall_score", 0.0)),
        "notes": str(parsed.get("notes", "")),
    }
