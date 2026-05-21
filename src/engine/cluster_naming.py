"""LLM-generated names and descriptions for clusters.

Takes the clusters and soft assignments produced by `initial_clustering` and
asks the LLM (via the harness) to label each cluster from its most
representative data points. Mutates the Cluster objects in place.
"""

import json

from src.harness import call_llm, render_prompt, extract_json_text
from src.models import Cluster as DbCluster, DataPoint, SoftAssignment as DbSoftAssignment

REPRESENTATIVE_SAMPLE_SIZE = 8


def _point_text(dp: DataPoint) -> str:
    title = (dp.data or {}).get("title", "") or ""
    text = (dp.data or {}).get("text", "") or ""
    return f"{title} {text}".strip()


def name_clusters(
    clusters: list[DbCluster],
    assignments: list[DbSoftAssignment],
    data_points: list[DataPoint],
    sample_size: int = REPRESENTATIVE_SAMPLE_SIZE,
) -> list[DbCluster]:
    """Fill in name and description for each cluster using the LLM.

    For each cluster, the `sample_size` data points with the highest assignment
    probability are sent to the LLM as representative examples. If the LLM call
    fails or returns unparseable output, that cluster keeps its placeholder
    name — naming a cluster never aborts the whole operation.

    Mutates `clusters` in place and also returns the list for convenience.
    """
    text_by_id = {dp.id: _point_text(dp) for dp in data_points}

    assignments_by_cluster: dict[str, list[DbSoftAssignment]] = {}
    for a in assignments:
        assignments_by_cluster.setdefault(a.cluster_id, []).append(a)

    for cluster in clusters:
        cluster_assignments = assignments_by_cluster.get(cluster.id, [])
        cluster_assignments.sort(key=lambda a: a.probability, reverse=True)
        sample_texts = [
            text_by_id.get(a.data_point_id, "")
            for a in cluster_assignments[:sample_size]
        ]
        sample_texts = [t for t in sample_texts if t]
        if not sample_texts:
            continue  # empty cluster — keep placeholder name

        reviews_block = "\n".join(f"- {t}" for t in sample_texts)
        prompt = render_prompt("cluster_naming", reviews=reviews_block)

        # Naming is best-effort: a failed LLM call (no API key, rate limit,
        # unreachable) or an unparseable response must not abort clustering —
        # the cluster simply keeps its "Cluster N" placeholder name.
        try:
            response = call_llm(
                [{"role": "user", "content": "Name this cluster."}],
                system=prompt,
            )
            parsed = json.loads(extract_json_text(response.text))
            cluster.name = str(parsed["name"])[:255]
            cluster.description = str(parsed["description"])
        except Exception:
            continue

    return clusters
