from src.schemas import ChatSessionState, InputOracle
from src.harness import ConversationContext


def f_output(
    state: ChatSessionState,
    oracle_turn: InputOracle,
    context: ConversationContext,
    total_points: int,
) -> dict:
    return {
        "action": "no_change",
        "clusters_updated": [],
        "display": "stub: no changes applied",
        "contradiction_detected": False,
        "cognitive_load_score": 0.0,
    }
