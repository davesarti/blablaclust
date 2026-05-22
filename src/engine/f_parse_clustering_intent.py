"""Parse the oracle's natural language clustering intent into structured parameters.

When the oracle describes how they want to cluster their data ("I want to separate
positive from negative reviews", "group by topic into 4 clusters"), this function
asks Claude to extract two things:
  - k: the number of clusters
  - axis: the dimension to cluster on (sentiment, topic, quality, ...)

The result is passed directly to initial_clustering so that k-means runs with the
oracle's intent baked in from the start, rather than using an arbitrary default k.
"""

import json

from src.harness import call_llm, render_prompt, extract_json_text
from src.logger import log

# Safe bounds for k — k-means below 2 is degenerate, above 20 is rarely useful
K_MIN = 2
K_MAX = 20


def f_parse_clustering_intent(
    oracle_intent: str,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
) -> dict[str, int | str]:
    """Extract k and clustering axis from the oracle's free-text description.

    Calls Claude with a structured extraction prompt and returns a dict with:
      - "k": int — number of clusters (clamped to [k_min, k_max])
      - "axis": str — the clustering dimension (e.g. "sentiment", "topic")
      - "reasoning": str — Claude's one-sentence explanation of the choice

    If Claude returns unparseable output, falls back to safe defaults
    (k=5, axis="semantic similarity") so clustering can always proceed.

    Args:
        oracle_intent: Free text from the oracle describing how they want to cluster.
        k_min: Minimum allowed k (default 2).
        k_max: Maximum allowed k (default 20).
    """
    prompt = render_prompt(
        "parse_clustering_intent",
        oracle_intent=oracle_intent,
        k_min=k_min,
        k_max=k_max,
    )

    try:
        response = call_llm(
            [{"role": "user", "content": oracle_intent}],
            system=prompt,
        )
        parsed = json.loads(extract_json_text(response.text))

        # Clamp k to the allowed range regardless of what Claude returned.
        k = max(k_min, min(k_max, int(parsed["k"])))

        return {
            "k": k,
            "axis": str(parsed.get("axis", "semantic similarity")),
            "reasoning": str(parsed.get("reasoning", "")),
        }

    except Exception as e:
        log.warning(f"f_parse_clustering_intent: Failed to parse oracle intent, using fallback defaults (k=5, axis='semantic similarity'). Error: {e}")
        return {
            "k": 5,
            "axis": "semantic similarity",
            "reasoning": "fallback: could not parse oracle intent",
        }
