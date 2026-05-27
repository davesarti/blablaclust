"""B4 — one-point validation (LLM-as-judge).

Asks an independent judge whether a specific data point belongs in a specific
cluster, given the cluster's name/description/representatives and the oracle's
feedback history. Used by the eval harness to:

  - Validate A1's soft-assignment calibration: endorsement rate should correlate
    negatively with each point's `f_uncertainty` score.
  - Spot-check end-of-session clustering quality at the point level (B4 in
    `docs/quality_specs.md`).

This is **out-of-band**: nothing in the live turn path calls it. The eval harness
([`scripts/run_eval.py`](../../scripts/run_eval.py)) samples N points per cluster
at end-of-session and aggregates the results.
"""

import json

from src.harness import (
    call_llm,
    estimate_cost_usd,
    extract_json_text,
    hash_prompt,
    render_prompt,
)
from src.logger import log_llm_call
from src.schemas import ChatSessionState, Cluster


def f_validate_point(
    state: ChatSessionState,
    cluster: Cluster,
    point_id: str,
    point_text: str,
) -> dict:
    """Ask the judge whether `point` content fits `cluster` as the oracle shaped it.

    Args:
        state: the full session state — supplies session_id, turn_number, and
            the feedback history that frames what the cluster is *supposed* to be.
        cluster: the cluster under test. Uses name, description, size, and
            representative_points (top-probability member texts).
        point_id: id of the data point being validated. Echoed back in the
            judge's reasoning; not used for clustering.
        point_text: the point's content the judge reads.

    Returns:
        Dict with three keys:
          - ``endorsed`` (bool): does this point belong in this cluster?
          - ``confidence`` (float in [0, 1]): the judge's confidence.
          - ``reasoning`` (str): one or two sentences citing the specific
            match/mismatch with the cluster's theme.

    Failure modes are handled by the caller — the LLM call may raise (network,
    rate limit, malformed JSON). The runner should record a null B4 record
    rather than aborting the whole eval batch.
    """
    representatives_block = (
        "\n".join(f"- {r}" for r in cluster.representative_points)
        or "(no representative texts available)"
    )

    prompt = render_prompt(
        "f_validate_point",
        session_id=state.session_id,
        turn_number=state.turn_number,
        cluster_id=cluster.id,
        cluster_name=cluster.name,
        cluster_description=cluster.description or "",
        cluster_size=cluster.size,
        representatives_block=representatives_block,
        point_id=point_id,
        point_text=point_text,
        feedback_history_json=json.dumps(
            [f.model_dump() for f in state.feedback_history]
        ),
    )

    msg = call_llm([{"role": "user", "content": prompt}], system="")
    log_llm_call(
        session_id=state.session_id,
        prompt_name="f_validate_point",
        prompt_hash=hash_prompt("f_validate_point"),
        usage=msg.usage,
        cost_usd=estimate_cost_usd(msg.usage),
    )

    parsed = json.loads(extract_json_text(msg.text))
    return {
        "endorsed": bool(parsed.get("endorsed", False)),
        "confidence": float(parsed.get("confidence", 0.0)),
        "reasoning": str(parsed.get("reasoning", "")),
    }
