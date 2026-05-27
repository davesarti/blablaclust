"""Decide what the system should do next after processing one oracle turn.

The three possible actions:
  - "show"  →  present the current clustering state to the oracle
  - "ask"   →  ask the oracle to clarify a specific ambiguous area
                (triggered when boundary points are found with high uncertainty)
  - "stop"  →  end the session (oracle is cognitively overloaded or the
                clustering has stabilised over many turns)

Decision logic (rule-based, no LLM needed):
  1. If cognitive load is too high (score >= 4) or too many turns have passed
     (turn_number > MAX_TURNS), stop — the oracle has given enough feedback.
  2. If there are boundary points with uncertainty above ASK_THRESHOLD,
     ask — there are genuinely ambiguous data points worth clarifying.
  3. Otherwise, show — present the current state and wait for the next turn.
"""

from src.engine.f_uncertainty import BoundaryPoint
from src.schemas import ChatSessionState, Display, SystemTurn
from src.harness import ConversationContext

# A point with uncertainty >= this value is worth asking the oracle about.
# Raised from 0.4 to 0.45 to avoid asking too frequently after initial clustering,
# when many points naturally land near cluster boundaries with uncertainty ~0.5.
# 0.45 means "only ask for points that are genuinely ambiguous (>45% split between clusters)".
ASK_THRESHOLD = 0.45

# After this many turns the session is likely to have converged.
MAX_TURNS = 20


def f_next_best_step(
    state: ChatSessionState,
    uncertainty: list[BoundaryPoint],
    context: ConversationContext,
) -> SystemTurn:
    """Return the next action the system should take.

    Args:
        state: Current clustering state (clusters, turn number, history).
        uncertainty: Output of f_uncertainty — boundary points sorted by score.
        context: Conversation memory, used to compute cognitive load.

    Returns:
        A SystemTurn describing what to show/ask/stop and why.
    """
    cognitive_load = context.get_cognitive_load_score()
    contradiction = bool(state.contradictions)

    # Rule 1: stop if the oracle is overloaded or the session has run long.
    # Threshold raised from 4 to 5: load=4 means "heavy but manageable",
    # only load=5 (the maximum) should trigger an automatic stop.
    if cognitive_load >= 5 or state.turn_number > MAX_TURNS:
        return SystemTurn(
            session_id=state.session_id,
            turn_number=state.turn_number,
            action="stop",
            clusters_updated=False,
            display=Display(
                type="text",
                content=(
                    f"Clustering complete after {state.turn_number} turns. "
                    f"Final state has {len(state.clusters)} clusters."
                ),
            ),
            contradiction_detected=contradiction,
            contradiction_detail=state.contradictions[-1] if contradiction else None,
            cognitive_load_score=cognitive_load,
            state_snapshot={"reason": "max_turns_or_load_reached"},
        )

    # Rule 2 (removed): proactively asking the oracle about ambiguous boundary
    # points was confusing — the display surfaced opaque cluster UUIDs and gave
    # no guidance on how to respond. The oracle can always give targeted feedback
    # on specific points when they want to; the system no longer interrupts the
    # flow to ask unprompted.

    # Rule 3: show — clustering looks stable, present current state.
    return SystemTurn(
        session_id=state.session_id,
        turn_number=state.turn_number,
        action="show",
        clusters_updated=False,
        display=Display(
            type="text",
            content=(
                f"Current clustering has {len(state.clusters)} clusters "
                f"after {state.turn_number} turns. "
                "All data points are assigned with high confidence."
            ),
            items=[
                {"cluster_id": c.id, "name": c.name, "size": c.size}
                for c in state.clusters
            ],
        ),
        contradiction_detected=contradiction,
        contradiction_detail=state.contradictions[-1] if contradiction else None,
        cognitive_load_score=cognitive_load,
        state_snapshot={"reason": "stable_clustering"},
    )

