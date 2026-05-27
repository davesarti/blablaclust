"""Decide what the system should do next after processing one oracle turn.

The three possible actions:
  - "show"  →  present the current clustering state to the oracle
  - "ask"   →  ask a targeted structural question (merge or split candidate)
  - "stop"  →  end the session (oracle is cognitively overloaded or the
                clustering has stabilised over many turns)

Decision logic (rule-based, no LLM needed):
  1. If cognitive load is too high (score >= 5) or too many turns have passed
     (turn_number > MAX_TURNS), stop.
  2. If two clusters overlap significantly, ask whether to merge them.
     If a cluster has low internal cohesion, ask whether to split it.
     Structural questions are asked one at a time, most urgent first.
  3. Otherwise, show — present the current state and wait for next turn.
"""

from src.engine.f_uncertainty import ClusterUncertainty
from src.schemas import ChatSessionState, Display, SystemTurn
from src.harness import ConversationContext

# After this many turns the session is likely to have converged.
MAX_TURNS = 20


def f_next_best_step(
    state: ChatSessionState,
    uncertainty: ClusterUncertainty,
    context: ConversationContext,
) -> SystemTurn:
    """Return the next action the system should take.

    Args:
        state: Current clustering state (clusters, turn number, history).
        uncertainty: Output of f_cluster_uncertainty — cluster-level overlap
                     and cohesion signals.
        context: Conversation memory, used to compute cognitive load.

    Returns:
        A SystemTurn describing what to show/ask/stop and why.
    """
    cognitive_load = context.get_cognitive_load_score()
    contradiction = bool(state.contradictions)

    # Rule 1: stop if the oracle is overloaded or the session has run long.
    # Threshold is 5 (the maximum): load=4 means "heavy but manageable".
    # Distinct reason codes let the eval harness bucket runs separately.
    if cognitive_load >= 5 or state.turn_number > MAX_TURNS:
        reason = (
            "cognitive_overload" if cognitive_load >= 5 else "max_turns_reached"
        )
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
            state_snapshot={"reason": reason},
        )

    # Rule 2a: ask about a merge if two clusters overlap significantly.
    # Overlap means many points sit between them with no clear home — the oracle
    # should decide if the distinction is real or if they should be one cluster.
    if uncertainty.overlaps:
        top = uncertainty.overlaps[0]
        pct = round(top.overlap_fraction * 100)
        return SystemTurn(
            session_id=state.session_id,
            turn_number=state.turn_number,
            action="ask",
            clusters_updated=False,
            display=Display(
                type="text",
                content=(
                    f"Clusters \"{top.cluster_a_name}\" and \"{top.cluster_b_name}\" "
                    f"overlap: {pct}% of data points ({top.n_overlap}) are ambiguous "
                    f"between them. Are they meaningfully distinct, or should they be merged?"
                ),
                items=[
                    {
                        "cluster_a_id": top.cluster_a_id,
                        "cluster_a_name": top.cluster_a_name,
                        "cluster_b_id": top.cluster_b_id,
                        "cluster_b_name": top.cluster_b_name,
                        "overlap_fraction": top.overlap_fraction,
                        "n_overlap": top.n_overlap,
                    }
                ],
            ),
            contradiction_detected=contradiction,
            contradiction_detail=state.contradictions[-1] if contradiction else None,
            cognitive_load_score=cognitive_load,
            state_snapshot={"overlap_count": len(uncertainty.overlaps)},
        )

    # Rule 2b: ask about a split if a cluster has low internal cohesion.
    # Low cohesion means the cluster is internally diffuse — points inside it
    # are not confidently assigned to it, suggesting sub-themes worth separating.
    if uncertainty.low_cohesion:
        worst = uncertainty.low_cohesion[0]
        pct = round(worst.mean_max_prob * 100)
        return SystemTurn(
            session_id=state.session_id,
            turn_number=state.turn_number,
            action="ask",
            clusters_updated=False,
            display=Display(
                type="text",
                content=(
                    f"Cluster \"{worst.cluster_name}\" has low internal cohesion "
                    f"(average confidence {pct}%). It may contain distinct sub-themes. "
                    f"Would you like to split it?"
                ),
                items=[
                    {
                        "cluster_id": worst.cluster_id,
                        "cluster_name": worst.cluster_name,
                        "mean_max_prob": worst.mean_max_prob,
                    }
                ],
            ),
            contradiction_detected=contradiction,
            contradiction_detail=state.contradictions[-1] if contradiction else None,
            cognitive_load_score=cognitive_load,
            state_snapshot={"low_cohesion_count": len(uncertainty.low_cohesion)},
        )

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
