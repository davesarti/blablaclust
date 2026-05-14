from src.schemas import ChatSessionState, InputOracle
from src.harness import ConversationContext


def f_next_state(
    state: ChatSessionState,
    oracle_turn: InputOracle,
    context: ConversationContext,
    total_points: int,
) -> ChatSessionState:
    return state
