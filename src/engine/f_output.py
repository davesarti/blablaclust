import json
from src.schemas import ChatSessionState, InputOracle
from src.harness import ConversationContext, render_prompt, call_llm, hash_prompt, estimate_cost_usd, loads_llm_json
from src.logger import log_llm_call


def f_output(
    state: ChatSessionState,
    oracle_turn: InputOracle,
    context: ConversationContext,
    total_points: int,
) -> tuple[dict, dict]:
    # Build the prompt by injecting current state and oracle input into the
    # f_output.txt template. Claude receives the full picture: existing clusters,
    # what the oracle just said, and the conversation history so far.
    prompt = render_prompt(
        "f_output",
        session_id=state.session_id,
        turn_number=state.turn_number,
        total_points=total_points,
        clusters_json=json.dumps([c.model_dump() for c in state.clusters]),
        feedback_type=oracle_turn.feedback_type,
        oracle_raw_text=oracle_turn.raw_text,
        target_cluster_ids=json.dumps(oracle_turn.target_cluster_ids),
        target_point_ids=json.dumps(oracle_turn.target_point_ids),
        history_summary=json.dumps([f.model_dump() for f in state.feedback_history]),
    )

    # Register the oracle's message in the conversation memory so future turns
    # can see the full dialogue history when calling build_messages().
    context.add_oracle_turn(oracle_turn.model_dump())

    # Call whichever LLM is configured (Claude or GPT) via the provider-agnostic
    # wrapper — LLM_PROVIDER env var controls which one is used.
    msg = call_llm(context.build_messages(), system=prompt)
    log_llm_call(
        session_id=state.session_id,
        prompt_name="f_output",
        prompt_hash=hash_prompt("f_output"),
        usage=msg.usage,
        cost_usd=estimate_cost_usd(msg.usage),
    )

    raw = loads_llm_json(msg.text)

    # Register Claude's response in the conversation memory.
    context.add_system_turn(raw)

    # Return raw dict + usage so callers can persist token counts and cost.
    # f_next_state is responsible for turning raw into a proper ChatSessionState.
    return raw, msg.usage
