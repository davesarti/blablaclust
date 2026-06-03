"""LLM-generated names and descriptions for clusters.

Takes the clusters and soft assignments produced by `initial_clustering` and
asks the LLM (via the harness) to label all clusters in a single call.
Mutates the Cluster objects in place.
"""

import difflib
import re

from src.harness import call_llm, render_prompt, loads_llm_json
from src.logger import log
from src.models import Cluster as DbCluster, DataPoint, SoftAssignment as DbSoftAssignment

_NAMING_PCT = 0.15   # fraction of hard-assigned points to send to the naming LLM
_NAMING_CAP = 30     # upper bound regardless of cluster size
_PLACEHOLDER_RE = re.compile(r"^Cluster \d+$")


def _point_text(dp: DataPoint) -> str:
    return dp.text or ""


def name_clusters(
    clusters: list[DbCluster],
    assignments: list[DbSoftAssignment],
    data_points: list[DataPoint],
    axis_hint: str | None = None,
) -> list[DbCluster]:
    """Fill in name and description for all clusters in a single LLM call.

    All clusters are described together in one prompt, which produces more
    consistent names and reduces latency compared to one call per cluster.

    For each cluster, 15% of its hard-assigned points (capped at 30), ranked
    by soft-assignment probability, are used as representative examples. This
    gives the naming LLM a broad enough view to avoid names that over-fit the
    single densest sub-theme. Clusters with no usable texts are silently
    skipped (placeholder name kept).

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

    # Compute hard cluster sizes (argmax across all clusters in this snapshot)
    # so the percentage-based sample is relative to actual membership, not the
    # full dataset size that appears in each cluster's assignment list.
    best_for_point: dict[str, tuple[str, float]] = {}
    for cid, cass in assignments_by_cluster.items():
        for a in cass:
            cur = best_for_point.get(a.data_point_id)
            if cur is None or a.probability > cur[1]:
                best_for_point[a.data_point_id] = (cid, a.probability)
    hard_sizes: dict[str, int] = {}
    for _, (cid, _) in best_for_point.items():
        hard_sizes[cid] = hard_sizes.get(cid, 0) + 1

    # Build one text block per cluster and remember which clusters have data.
    cluster_blocks: list[str] = []
    nameable_ids: list[str] = []
    block_by_id: dict[str, str] = {}

    for cluster in clusters:
        cluster_assignments = assignments_by_cluster.get(cluster.id, [])
        cluster_assignments.sort(key=lambda a: a.probability, reverse=True)
        cluster_size = hard_sizes.get(cluster.id, 0) or len(cluster_assignments)
        n = min(max(1, int(cluster_size * _NAMING_PCT)), _NAMING_CAP)
        sample_texts = [
            text_by_id.get(a.data_point_id, "")
            for a in cluster_assignments[:n]
        ]
        sample_texts = [t for t in sample_texts if t]
        if not sample_texts:
            continue  # empty cluster — keep placeholder name

        reviews_block = "\n".join(f"- {t}" for t in sample_texts)
        block = f"[Cluster id: {cluster.id}]\n{reviews_block}"
        cluster_blocks.append(block)
        nameable_ids.append(cluster.id)
        block_by_id[cluster.id] = block

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

    # Naming is best-effort: a failed LLM call or unparseable response must not
    # abort clustering. Retry up to 2 extra times before giving up — malformed
    # JSON from the LLM is the most common transient failure here.
    _MAX_ATTEMPTS = 3
    response = None
    last_exc: Exception | None = None
    clusters_by_id = {c.id: c for c in clusters}

    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = call_llm(
                [{"role": "user", "content": "Name all clusters."}],
                system=prompt,
            )
            parsed = loads_llm_json(response.text)

            parsed_keys = list(parsed.keys())
            for cluster_id in nameable_ids:
                entry = parsed.get(cluster_id)
                if not isinstance(entry, dict):
                    # The LLM may have mistyped one character of the UUID key.
                    # Try a fuzzy match on the parsed keys — same fix as the
                    # UUID repair in f_apply_operations for the inverse direction.
                    matches = difflib.get_close_matches(cluster_id, parsed_keys, n=1, cutoff=0.9)
                    if matches:
                        log.warning(
                            "cluster_naming: LLM mistyped cluster_id key %s -> %s, "
                            "recovering via fuzzy match", cluster_id, matches[0]
                        )
                        entry = parsed.get(matches[0])
                if not isinstance(entry, dict):
                    continue  # truly missing — keep placeholder
                cluster = clusters_by_id[cluster_id]
                if name := entry.get("name"):
                    cluster.name = str(name)[:255]
                if description := entry.get("description"):
                    cluster.description = str(description)
            break  # success — stop retrying
        except Exception as e:
            last_exc = e
            snippet = repr(response.text[:200]) if response is not None else "<no response>"
            if attempt < _MAX_ATTEMPTS - 1:
                log.warning(
                    "cluster_naming: attempt %d/%d failed (%s), retrying. Response: %s",
                    attempt + 1, _MAX_ATTEMPTS, e, snippet,
                )
            else:
                log.warning(
                    "cluster_naming: all %d attempts failed, keeping placeholder names. "
                    "Last error: %s. Last response: %s",
                    _MAX_ATTEMPTS, e, snippet,
                )

    # Detect clusters that still have a placeholder name after the main call
    # (e.g. because the LLM UUID typo exceeded the fuzzy-match cutoff) and
    # retry with a smaller, targeted prompt containing only those clusters.
    still_placeholder = [
        cid for cid in nameable_ids
        if _PLACEHOLDER_RE.match(clusters_by_id[cid].name)
    ]
    if still_placeholder:
        log.info(
            "cluster_naming: %d cluster(s) still have placeholder names, "
            "issuing targeted retry: %s",
            len(still_placeholder), still_placeholder,
        )
        retry_blocks = "\n\n".join(block_by_id[cid] for cid in still_placeholder)
        retry_prompt = render_prompt(
            "cluster_naming",
            clusters_block=retry_blocks,
            axis_context=axis_context,
        )
        retry_response = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                retry_response = call_llm(
                    [{"role": "user", "content": "Name all clusters."}],
                    system=retry_prompt,
                )
                parsed = loads_llm_json(retry_response.text)
                parsed_keys = list(parsed.keys())
                for cluster_id in still_placeholder:
                    entry = parsed.get(cluster_id)
                    if not isinstance(entry, dict):
                        matches = difflib.get_close_matches(cluster_id, parsed_keys, n=1, cutoff=0.9)
                        if matches:
                            log.warning(
                                "cluster_naming retry: LLM mistyped cluster_id %s -> %s, "
                                "recovering via fuzzy match",
                                cluster_id, matches[0],
                            )
                            entry = parsed.get(matches[0])
                    if not isinstance(entry, dict):
                        continue
                    cluster = clusters_by_id[cluster_id]
                    if name := entry.get("name"):
                        cluster.name = str(name)[:255]
                    if description := entry.get("description"):
                        cluster.description = str(description)
                break
            except Exception as e:
                snippet = repr(retry_response.text[:200]) if retry_response is not None else "<no response>"
                if attempt < _MAX_ATTEMPTS - 1:
                    log.warning(
                        "cluster_naming retry: attempt %d/%d failed (%s), retrying. Response: %s",
                        attempt + 1, _MAX_ATTEMPTS, e, snippet,
                    )
                else:
                    log.warning(
                        "cluster_naming retry: all %d attempts failed for placeholder clusters. "
                        "Last error: %s. Last response: %s",
                        _MAX_ATTEMPTS, e, snippet,
                    )

    return clusters
