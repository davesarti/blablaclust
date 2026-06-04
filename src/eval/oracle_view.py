"""Build the per-turn 'oracle view' payload.

The oracle LLM sees the cluster panel through this view: name, description,
size, and a few representative point texts per active cluster. The view is
also the source of truth for which cluster_ids / point_ids the oracle is
allowed to reference — invented IDs are filtered against the same sets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


# Soft cap on example-text length so the prompt stays compact. Long-form
# point texts are truncated mid-sentence; the oracle doesn't need full text
# to decide its next move.
_EXAMPLE_TRUNCATE_CHARS = 160


@dataclass
class OracleView:
    """The cluster panel as the oracle sees it, plus the legal ID sets."""

    cluster_lines: List[str] = field(default_factory=list)
    cluster_ids: set[str] = field(default_factory=set)
    point_ids: set[str] = field(default_factory=set)

    def render(self) -> str:
        if not self.cluster_lines:
            return "(no active clusters)"
        return "\n".join(self.cluster_lines)


def _example_text(point: Mapping[str, Any]) -> str:
    """Best-effort one-line text for a representative point.

    Datasets vary: amazon_reviews puts the content under data.text; 20ng
    splits across title/text. Fall back to JSON if neither key is present so
    the oracle still has *something* to look at.
    """
    data = point.get("data") or {}
    text = (data.get("text") or data.get("title") or "").strip()
    if not text:
        text = json.dumps(data, ensure_ascii=False)
    text = " ".join(text.split())  # collapse whitespace
    if len(text) > _EXAMPLE_TRUNCATE_CHARS:
        text = text[: _EXAMPLE_TRUNCATE_CHARS - 1].rstrip() + "…"
    return text


def build_oracle_view(
    state: Mapping[str, Any],
    cluster_points: Mapping[str, List[Mapping[str, Any]]],
    n_examples: int = 3,
) -> OracleView:
    """Render the view from a ChatSessionState dict + per-cluster point lists.

    Parameters
    ----------
    state
        The /sessions/{sid}/state response body (a ChatSessionState payload).
    cluster_points
        Mapping cluster_id -> list of ClusterPoint dicts (from
        /clusters/{cid}/points). Only the first `n_examples` are shown.
    n_examples
        Max example texts per cluster.
    """
    view = OracleView()
    clusters = state.get("clusters") or []
    # Sort by id for deterministic rendering — useful in tests + diffable logs.
    for cluster in sorted(clusters, key=lambda c: c.get("id", "")):
        cid = cluster.get("id", "?")
        name = cluster.get("name") or "(unnamed)"
        description = cluster.get("description") or "(no description)"
        size = cluster.get("size")
        size_str = f"size {size}" if size is not None else "size ?"
        view.cluster_ids.add(cid)

        lines = [f'- Cluster {cid} "{name}" ({size_str})',
                 f"    description: {description}"]

        examples = list(cluster_points.get(cid, []))[:n_examples]
        if examples:
            lines.append("    examples (id | text):")
            for point in examples:
                pid = point.get("id")
                if pid:
                    view.point_ids.add(pid)
                pid_str = pid if pid else "?"
                lines.append(f"      - [{pid_str}] {_example_text(point)}")
        view.cluster_lines.extend(lines)

    return view


def render_notes(notes: Mapping[str, Any]) -> str:
    """Render a persona's notes object as a bulleted list for the prompt."""
    items: List[str] = []
    for key, value in notes.items():
        if value is None or value == "":
            continue
        items.append(f"- {key}: {value}")
    if not items:
        return "- (no notes — use a neutral, helpful tone)"
    return "\n".join(items)


def filter_invented_ids(
    cluster_ids: List[str],
    point_ids: List[str],
    view: OracleView,
) -> Dict[str, List[str]]:
    """Strip any cluster_id / point_id the oracle invented (not in the view)."""
    return {
        "target_cluster_ids": [cid for cid in cluster_ids if cid in view.cluster_ids],
        "target_point_ids":  [pid for pid in point_ids  if pid in view.point_ids],
    }
