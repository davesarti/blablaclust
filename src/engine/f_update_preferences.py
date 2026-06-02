"""Maintain a rolling oracle preference summary from feedback history.

Called after each successful turn to distil the oracle's revealed preferences
into 3–5 bullet points. The result is stored on the session and injected into
the f_output prompt so the LLM has a pre-digested view of what the oracle
cares about — not just the raw turn log.

Design choices
--------------
* Best-effort: any LLM or parse failure leaves the old summary unchanged.
* Only called when ≥ MIN_TURNS_BEFORE_SUMMARY turns exist — the first turn
  has no history to summarise.
* Uses a separate system prompt (prompts/f_update_preferences.txt) so the
  logic is versioned in the prompts directory alongside f_output.txt.
"""

from src.schemas import ChatSessionState
from src.harness import call_llm, render_prompt, loads_llm_json
from src.logger import log

# Don't spend a token budget summarising a single exchange — wait until at
# least this many oracle turns have accumulated.
_MIN_TURNS_BEFORE_SUMMARY = 1


def f_update_preferences(state: ChatSessionState) -> str | None:
    """Return a fresh oracle preference summary, or None on failure / no history.

    The return value is a newline-joined list of bullet points (each prefixed
    with '- '), ready to be inserted directly into a prompt block.  Returns
    None when there is not enough history yet or when the LLM call fails.

    Args:
        state: The session state *after* the latest turn has been persisted
               (so feedback_history already includes the turn we just completed).
    """
    if len(state.feedback_history) < _MIN_TURNS_BEFORE_SUMMARY:
        return None

    # Build a human-readable feedback log.
    feedback_lines: list[str] = []
    for f in state.feedback_history:
        line = f"Turn {f.turn} [{f.type}]: {f.content}"
        if f.target_cluster_ids:
            line += f"  (targeted clusters: {', '.join(f.target_cluster_ids)})"
        feedback_lines.append(line)

    clusters_block = "\n".join(
        f"- {c.name}: {c.description}" for c in state.clusters
    ) or "(no active clusters)"

    prompt = render_prompt(
        "f_update_preferences",
        feedback_history="\n".join(feedback_lines),
        current_clusters=clusters_block,
    )

    try:
        response = call_llm(
            [{"role": "user", "content": "Extract oracle preferences."}],
            system=prompt,
        )
        parsed = loads_llm_json(response.text)
        preferences = parsed.get("preferences", [])
        if not isinstance(preferences, list) or not preferences:
            log.warning("f_update_preferences: LLM returned empty preferences list")
            return None
        bullets = [str(p).strip() for p in preferences[:5] if str(p).strip()]
        if not bullets:
            return None
        return "\n".join(f"- {p}" for p in bullets)
    except Exception as exc:
        log.warning(
            "f_update_preferences: LLM call failed (%s), keeping previous summary", exc
        )
        return None
