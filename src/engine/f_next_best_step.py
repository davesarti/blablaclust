from src.schemas import ChatSessionState, SystemTurn, Display
from src.harness import ConversationContext


def f_next_best_step(
    state: ChatSessionState,
    uncertainty: list,
    context: ConversationContext,
) -> SystemTurn:
    return SystemTurn(
        session_id=state.session_id,
        turn_number=state.turn_number,
        action="show",
        clusters_updated=False,
        display=Display(type="text", content="stub: no action taken"),
        contradiction_detected=False,
        contradiction_detail=None,
        cognitive_load_score=1,
        state_snapshot={},
    )