"""LLM-generated names and descriptions for clusters.

Takes the clusters and soft assignments produced by `initial_clustering` and
asks the LLM (via the harness) to label all clusters in a single call.
Mutates the Cluster objects in place.
"""

from src.harness import call_llm, render_prompt, loads_llm_json
from src.logger import log
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
    axis_hint: str | None = None,
) -> list[DbCluster]:
    """Fill in name and description for all clusters in a single LLM call.

    All clusters are described together in one prompt, which produces more
    consistent names and reduces latency compared to one call per cluster.

    For each cluster, the `sample_size` data points with the highest assignment
    probability are selected as representative examples. Clusters with no
    usable texts are silently skipped (placeholder name kept).

    If the LLM call fails or returns unparseable output, all clusters keep
    their placeholder names. If the response omits individual cluster IDs,
    those clusters also keep their placeholder names — naming never aborts
    the overall clustering operation.

    Args:
        axis_hint: When provided, the naming prompt instructs the LLM to label
            clusters along this semantic axis (e.g. "angry tone") rather than
            purely by topic.

    Mutates `clusters` in place and also returns the list for convenience.
    """
    text_by_id = {dp.id: _point_text(dp) for dp in data_points}

    assignments_by_cluster: dict[str, list[DbSoftAssignment]] = {}
    for a in assignments:
        assignments_by_cluster.setdefault(a.cluster_id, []).append(a)

    # Build one text block per cluster and remember which clusters have data.
    cluster_blocks: list[str] = []
    nameable_ids: list[str] = []

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
        cluster_blocks.append(f"[Cluster id: {cluster.id}]\n{reviews_block}")
        nameable_ids.append(cluster.id)

    if not cluster_blocks:
        return clusters  # nothing to name

    clusters_block = "\n\n".join(cluster_blocks)

    if axis_hint:
        axis_context = (
            f"\nAXIS CONTEXT\n"
            f"These clusters were produced by re-orienting the entire dataset "
            f"along the semantic axis \"{axis_hint}\". Each cluster represents "
            f"a different POSITION on this axis, not a different topic.\n"
            f"For each cluster, apply this rule:\n"
            f"- If you can meaningfully say these texts score HIGH or LOW on "
            f"'{axis_hint}', name the cluster by its axis position using natural "
            f"words that convey degree or intensity on that axis.\n"
            f"- Only use a topic name if '{axis_hint}' is genuinely inapplicable "
            f"to these texts. When in doubt, prefer an axis-based name.\n"
        )
    else:
        axis_context = ""

    prompt = render_prompt(
        "cluster_naming",
        clusters_block=clusters_block,
        axis_context=axis_context,
    )

    # Naming is best-effort: a failed LLM call (no API key, rate limit,
    # unreachable) or an unparseable response must not abort clustering.
    response = None
    try:
        response = call_llm(
            [{"role": "user", "content": "Name all clusters."}],
            system=prompt,
        )
        parsed = loads_llm_json(response.text)

        clusters_by_id = {c.id: c for c in clusters}
        for cluster_id in nameable_ids:
            entry = parsed.get(cluster_id)
            if not isinstance(entry, dict):
                continue  # missing or malformed entry — keep placeholder
            cluster = clusters_by_id[cluster_id]
            if name := entry.get("name"):
                cluster.name = str(name)[:255]
            if description := entry.get("description"):
                cluster.description = str(description)
    except Exception as e:
        # Log a snippet of the raw response so a recurring parse failure is
        # diagnosable rather than opaque (the response itself was never logged).
        snippet = repr(response.text[:500]) if response is not None else "<no response>"
        log.warning(
            f"cluster_naming: LLM call failed or returned invalid JSON, keeping "
            f"placeholder names for all clusters. Error: {e}. Raw response: {snippet}"
        )

    return clusters
