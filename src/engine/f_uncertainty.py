"""Boundary-point uncertainty scoring from soft cluster assignments.

Soft assignments give every data point a probability for each cluster (they sum
to 1 per point). A point that sits right on the boundary between two clusters
will have probabilities close to 0.5/0.5 — it genuinely could belong to either.
A point deep inside a cluster will have something like 0.95/0.05 — very certain.

The uncertainty score captures this:

    uncertainty = 1 - max(probability across all clusters for that point)

  - score ≈ 0.0  →  the point belongs firmly to one cluster (max prob ≈ 1.0)
  - score ≈ 0.5  →  split between two clusters (max prob ≈ 0.5)
  - score ≈ 1 - 1/k  →  perfectly ambiguous across k clusters

f_uncertainty is called by f_next_best_step to decide when the oracle should be
asked to clarify ambiguous areas rather than getting a global summary.
"""

from dataclasses import dataclass

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.models import Cluster as DbCluster, DataPoint, SoftAssignment


@dataclass
class BoundaryPoint:
    """A data point ranked by how ambiguous its cluster membership is.

    cluster_scores maps cluster_id → probability for every cluster in the
    session. The values sum to 1.0 (they come straight from the softmax in
    initial_clustering). uncertainty_score is derived from those: the closer
    it is to 1.0, the more borderline the point is.
    """
    point_id: str
    text_preview: str           # first 200 chars of title+text, for display
    cluster_scores: dict[str, float]   # {cluster_id: probability}
    uncertainty_score: float    # 1 - max(cluster_scores.values())


def _point_text(dp: DataPoint) -> str:
    """Concatenate title and text fields from a DataPoint's JSON data blob."""
    title = (dp.data or {}).get("title", "") or ""
    text = (dp.data or {}).get("text", "") or ""
    return f"{title} {text}".strip()


def f_uncertainty(
    session_id: str,
    db: Session,
    top_n: int = 10,
) -> list[BoundaryPoint]:
    """Return the top_n most uncertain data points for a session.

    Workflow:
      1. Find all cluster IDs that belong to this session.
      2. Find the latest turn that has soft assignments (i.e. the most recent
         clustering run — either initial k-means or a model-driven update).
      3. Load all soft assignments at that turn and group them by data point.
      4. Compute uncertainty_score = 1 - max(probability) for each point.
      5. Sort descending (most ambiguous first) and return the top_n.

    Returns an empty list if no clustering has been run yet (no clusters or no
    soft assignments exist) — callers should handle this gracefully.

    Args:
        session_id: The ChatSession to score.
        db: SQLAlchemy session (injected by FastAPI's Depends).
        top_n: Maximum number of boundary points to return (default 10).
    """
    # Step 1: get every cluster ID for this session.
    # We scope all subsequent queries to these IDs so we never accidentally
    # read assignments from a different session that happens to share a cluster.
    cluster_ids = [
        cid
        for (cid,) in db.query(DbCluster.id)
        .filter(DbCluster.session_id == session_id)
        .all()
    ]
    if not cluster_ids:
        return []  # session has no clusters yet — clustering hasn't run

    # Step 2: find the most recent turn that has soft assignments.
    # SoftAssignment rows are keyed by (data_point_id, cluster_id, turn_number),
    # so new clustering runs add rows at turn_number = current turn rather than
    # replacing old ones. We always want the freshest picture.
    latest_turn = (
        db.query(func.max(SoftAssignment.turn_number))
        .filter(SoftAssignment.cluster_id.in_(cluster_ids))
        .scalar()
    )
    if latest_turn is None:
        return []  # clusters exist but initial_clustering hasn't run yet

    # Step 3: load all assignments at the latest turn.
    # Each SoftAssignment row has one (point, cluster) probability value.
    # A full point's distribution spans k rows — one per cluster.
    assignments = (
        db.query(SoftAssignment)
        .filter(
            SoftAssignment.cluster_id.in_(cluster_ids),
            SoftAssignment.turn_number == latest_turn,
        )
        .all()
    )

    # Group probabilities by data point: {point_id: {cluster_id: prob}}
    scores_by_point: dict[str, dict[str, float]] = {}
    for a in assignments:
        scores_by_point.setdefault(a.data_point_id, {})[a.cluster_id] = a.probability

    # Step 4: fetch text previews for display (best-effort; missing → empty string).
    point_ids = list(scores_by_point.keys())
    data_points = (
        db.query(DataPoint).filter(DataPoint.id.in_(point_ids)).all()
    )
    text_by_id = {dp.id: _point_text(dp) for dp in data_points}

    # Step 5: compute uncertainty score and build result objects.
    results: list[BoundaryPoint] = []
    for point_id, cluster_scores in scores_by_point.items():
        # max(cluster_scores) is the probability of the "winning" cluster.
        # Subtracting from 1 gives how much probability mass is spread elsewhere.
        uncertainty_score = 1.0 - max(cluster_scores.values())
        results.append(
            BoundaryPoint(
                point_id=point_id,
                text_preview=text_by_id.get(point_id, "")[:200],
                cluster_scores=cluster_scores,
                uncertainty_score=round(uncertainty_score, 4),
            )
        )

    # Sort most-ambiguous first so the caller can just take results[:n].
    results.sort(key=lambda p: p.uncertainty_score, reverse=True)
    return results[:top_n]
