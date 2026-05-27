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
        cost_usd=estimate_cost_usd(msg.usage),
    )

    return json.loads(extract_json_text(msg.text))
