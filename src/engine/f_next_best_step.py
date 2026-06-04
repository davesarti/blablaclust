"""Decide what the system should do next after processing one oracle turn.

The three possible actions:
  - "show"  →  present the current clustering state to the oracle
  - "ask"   →  ask a targeted structural question (merge or split candidate)
  - "stop"  →  end the session (overload saturated, or oracle converged)

Decision logic (rule-based, no LLM needed):
  1. If the oracle explicitly signaled close intent (f_output classified
     the turn as action="end"), stop with reason="converged". This wins
     over every other rule — an explicit close from the oracle is always
     a clean termination, never a forced shutdown.
  2. If cognitive_load.score >= 5, stop with reason="cognitive_overload".
     The driver and breakdown are written into state_snapshot so the eval
     report can group stops by which signal saturated.
  3. If two clusters overlap significantly, ask whether to merge them.
     If a cluster has low internal cohesion, ask whether to split it.
     Structural questions are asked one at a time, most urgent first.
  4. Otherwise, show — present the current state and wait for next turn.
"""

from src.engine.f_uncertainty import ClusterUncertainty
from src.schemas import ChatSessionState, CognitiveLoad, Display, SystemTurn


def f_next_best_step(
    state: ChatSessionState,
    uncertainty: ClusterUncertainty,
    cognitive_load: CognitiveLoad,
    oracle_signaled_end: bool = False,
) -> SystemTurn:
    """Return the next action the system should take.

    Args:
        state: Current clustering state (clusters, turn number, history).
        uncertainty: Output of f_cluster_uncertainty — cluster-level overlap
                     and cohesion signals.
        cognitive_load: Output of f_cognitive_load — deterministic A3 score.

    Returns:
        A SystemTurn describing what to show/ask/stop and why.
    """
    # Rule 1: stop when the oracle explicitly signaled close intent (f_output
    # emitted action="end"). This takes priority over overload — an explicit
    # oracle close is always recorded as a converged termination.
    if oracle_signaled_end:
        return SystemTurn(
            session_id=state.session_id,
            turn_number=state.turn_number,
            action="stop",
            clusters_updated=False,
            display=Display(
                type="text",
                content=(
                    f"Clustering converged after {state.turn_number} turns. "
                    f"Final state has {len(state.clusters)} clusters."
                ),
            ),
            cognitive_load_score=cognitive_load.score,
            state_snapshot={
                "reason": "converged",
                "cluster_count": len(state.clusters),
            },
        )

    # Rule 2: stop when the deterministic A3 score saturates.
    if cognitive_load.score >= 5:
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
            cognitive_load_score=cognitive_load.score,
            state_snapshot={
                "reason": "cognitive_overload",
                "cognitive_load_driver": cognitive_load.driver,
                "cognitive_load_breakdown": {
                    "turns": cognitive_load.turns_score,
                    "tokens": cognitive_load.tokens_score,
                    "clusters": cognitive_load.clusters_score,
                },
                "cognitive_load_raw": {
                    "turns_used": cognitive_load.turns_used,
                    "tokens_used": cognitive_load.tokens_used,
                    "clusters_count": cognitive_load.clusters_count,
                },
            },
        )

    # Rule 2a: ask about a merge if two clusters overlap significantly.
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
            cognitive_load_score=cognitive_load.score,
            state_snapshot={"overlap_count": len(uncertainty.overlaps)},
        )

    # Rule 2b: ask about a split if a cluster has low internal cohesion.
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
            cognitive_load_score=cognitive_load.score,
            state_snapshot={"low_cohesion_count": len(uncertainty.low_cohesion)},
        )

    # Rule 4: show — clustering looks stable, present current state.
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
        cognitive_load_score=cognitive_load.score,
        state_snapshot={"reason": "stable_clustering"},
    )
