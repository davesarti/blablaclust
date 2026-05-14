from src.schemas import ChatSessionState


def f_eval(
    state: ChatSessionState,
) -> dict:
    return {
        "coherence_score": 0.5,
        "coverage_score": 0.5,
        "suggested_merges": [],
        "notes": "stub",
    }