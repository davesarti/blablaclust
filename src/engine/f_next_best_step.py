import json

from src.schemas import ChatSessionState, SystemTurn, Display
from src.harness import ConversationContext, render_prompt, call_llm, hash_prompt, estimate_cost_usd
from src.logger import log_llm_call


def f_next_best_step(
    state: ChatSessionState,
    uncertainty: list,
    context: ConversationContext,
) -> SystemTurn:
    prompt = render_prompt(
        "f_next_best_step",
        turn_number=state.turn_number,
        cognitive_load=context.get_cognitive_load_score(),
        current_state_json=json.dumps([c.model_dump() for c in state.clusters]),
        last_oracle_input=json.dumps(
            state.feedback_history[-1].model_dump() if state.feedback_history else {}
        ),
        uncertainty_scores_json=json.dumps(uncertainty),
    )

    msg = call_llm([{"role": "user", "content": prompt}], system="")
    log_llm_call(
        session_id=state.session_id,
        prompt_name="f_next_best_step",
        prompt_hash=hash_prompt("f_next_best_step"),
        usage=msg.usage,
        cost_usd=estimate_cost_usd(msg.usage),
    )

    raw = json.loads(msg.text)

    return SystemTurn(
        session_id=state.session_id,
        turn_number=state.turn_number,
        action=raw["action"],
        clusters_updated=False,
        display=Display(type="text", content=raw["display_content"]),
        contradiction_detected=False,
        contradiction_detail=None,
        cognitive_load_score=raw["cognitive_load_score"],
        state_snapshot={},
    )
