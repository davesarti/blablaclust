from src.schemas import ChatSessionState, InputOracle, Cluster, FeedbackEntry
from src.harness import ConversationContext
from src.engine.f_output import f_output


def f_next_state(
    state: ChatSessionState,
    oracle_turn: InputOracle,
    context: ConversationContext,
    total_points: int,
) -> ChatSessionState:
    # Ask Claude to apply the oracle's feedback to the current clustering.
    # raw is a dict with keys: action, clusters_updated, display,
    # contradiction_detected, cognitive_load_score.
    raw = f_output(state, oracle_turn, context, total_points)

    # Rebuild the cluster list from Claude's response. Claude only knows about
    # clustering fields (name, description, size, etc.) — we fill in session_id
    # and created_at_turn ourselves since those are system-level metadata.
    # If Claude returned no updated clusters (e.g. action="explain"), keep the
    # current ones unchanged.
    updated_clusters = [
        Cluster(
            id=c["id"],
            session_id=state.session_id,
            name=c["name"],
            description=c["description"],
            size=c["size"],
            representative_points=c.get("representative_points", []),
            created_at_turn=state.turn_number,
        )
        for c in raw.get("clusters_updated", [])
    ] or state.clusters

    # Check if the oracle's new feedback contradicts something they said earlier.
    # detect_contradiction is a stub in P4's harness (always returns None) until
    # Week 3 when P4 implements the real logic. Safe to call now.
    contradiction = context.detect_contradiction(
        oracle_turn.model_dump(),
        [c.model_dump() for c in state.clusters],
    )

    # Record what the oracle said this turn so future prompts have full history.
    new_feedback = FeedbackEntry(
        turn=state.turn_number,
        type=oracle_turn.feedback_type,
        content=oracle_turn.raw_text,
        target_cluster_id=oracle_turn.target_cluster_id,
    )

    # Return a new state object — never mutate the input state.
    # turn_number advances by 1 so every turn is uniquely identified.
    return ChatSessionState(
        session_id=state.session_id,
        turn_number=state.turn_number + 1,
        dataset_name=state.dataset_name,
        status=state.status,
        clusters=updated_clusters,
        feedback_history=state.feedback_history + [new_feedback],
        contradictions=state.contradictions + ([contradiction] if contradiction else []),
    )
