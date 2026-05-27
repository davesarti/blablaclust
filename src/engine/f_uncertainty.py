"""Cluster-level uncertainty scoring from soft assignments.

Instead of surfacing individual borderline data points (which doesn't scale
to oracles — asking about 1200 points one by one is useless), this module
aggregates point-level soft assignments up to the *cluster* level and
produces two actionable signals:

  ClusterOverlap   — two clusters bleed into each other; ask about merge.
  ClusterCohesion  — a cluster is internally diffuse; ask about split.

Both are expressed as single numbers per cluster (pair), so the oracle gets
one structural question at a time instead of a wall of individual data points.

The old per-point BoundaryPoint type is kept as an internal helper used by
f_cluster_uncertainty() to compute the cluster-level aggregates.
"""

from dataclasses import dataclass, field

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.models import Cluster as DbCluster, SoftAssignment

# A point "overlaps" between clusters A and B when it has meaningful probability
# in both — meaning neither cluster fully owns it.
OVERLAP_PROB_THRESHOLD = 0.30

# A cluster is a split candidate when its mean winner-probability is below this.
# 0.60 means "on average the winning cluster only holds 60% of the probability
# mass — the rest leaks into neighbours".
COHESION_SPLIT_THRESHOLD = 0.60

# Only surface overlaps that affect at least this fraction of all session points.
OVERLAP_SIGNIFICANCE_THRESHOLD = 0.05


@dataclass
class ClusterOverlap:
    """Two clusters that share significant probability mass.

    overlap_fraction is n_overlap / n_total_points in the session.
    High values mean many points sit between the two clusters and a merge
    question is worth asking.
    """
    cluster_a_id: str
    cluster_b_id: str
    cluster_a_name: str
    cluster_b_name: str
    overlap_fraction: float   # overlapping points / total session points
    n_overlap: int            # raw count of overlapping points


@dataclass
class ClusterCohesion:
    """Internal tightness of a single cluster.

    mean_max_prob is the average probability of the winning cluster across
    all points whose argmax is this cluster.  Low values mean the cluster is
    diffuse — a split may clarify the structure.
    """
    cluster_id: str
    cluster_name: str
    mean_max_prob: float   # 1.0 = perfectly tight, lower = diffuse


@dataclass
class ClusterUncertainty:
    """Cluster-level uncertainty summary for one session.

    overlaps  — cluster pairs sorted by overlap_fraction descending
    low_cohesion — clusters sorted by mean_max_prob ascending (most diffuse first)
    """
    overlaps: list[ClusterOverlap] = field(default_factory=list)
    low_cohesion: list[ClusterCohesion] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.overlaps and not self.low_cohesion


def f_cluster_uncertainty(
    session_id: str,
    db: Session,
) -> ClusterUncertainty:
    """Compute cluster-level uncertainty for a session.

    Returns a ClusterUncertainty with the top overlapping cluster pair and
    the most diffuse cluster (if any breach their thresholds).  Returns an
    empty ClusterUncertainty if no clustering has run yet.
    """
    # --- load cluster metadata (id → name) ---
    clusters = (
        db.query(DbCluster)
        .filter(DbCluster.session_id == session_id, DbCluster.dissolved_at_turn.is_(None))
        .all()
    )
    if not clusters:
        return ClusterUncertainty()

    cluster_ids = [c.id for c in clusters]
    name_by_id = {c.id: c.name for c in clusters}

    # --- find latest soft-assignment snapshot ---
    latest_turn = (
        db.query(func.max(SoftAssignment.turn_number))
        .filter(SoftAssignment.cluster_id.in_(cluster_ids))
        .scalar()
    )
    if latest_turn is None:
        return ClusterUncertainty()

    # --- load all assignments at that turn ---
    assignments = (
        db.query(SoftAssignment)
        .filter(
            SoftAssignment.cluster_id.in_(cluster_ids),
            SoftAssignment.turn_number == latest_turn,
        )
        .all()
    )

    # Group by data point: {point_id: {cluster_id: prob}}
    scores_by_point: dict[str, dict[str, float]] = {}
    for a in assignments:
        scores_by_point.setdefault(a.data_point_id, {})[a.cluster_id] = a.probability

    n_total = len(scores_by_point)
    if n_total == 0:
        return ClusterUncertainty()

    # --- cluster-pair overlap ---
    # For each ordered pair (a, b) count points where both have prob >= threshold.
    overlap_counts: dict[tuple[str, str], int] = {}
    for point_scores in scores_by_point.values():
        above = [cid for cid, p in point_scores.items() if p >= OVERLAP_PROB_THRESHOLD]
        for i, a in enumerate(above):
            for b in above[i + 1:]:
                key = (min(a, b), max(a, b))   # canonical order
                overlap_counts[key] = overlap_counts.get(key, 0) + 1

    overlaps: list[ClusterOverlap] = []
    for (a, b), n in overlap_counts.items():
        fraction = n / n_total
        if fraction >= OVERLAP_SIGNIFICANCE_THRESHOLD:
            overlaps.append(ClusterOverlap(
                cluster_a_id=a,
                cluster_b_id=b,
                cluster_a_name=name_by_id.get(a, a),
                cluster_b_name=name_by_id.get(b, b),
                overlap_fraction=round(fraction, 3),
                n_overlap=n,
            ))
    overlaps.sort(key=lambda o: o.overlap_fraction, reverse=True)

    # --- cluster cohesion ---
    # For each cluster, look at points whose argmax is that cluster and
    # compute the mean of their max probability.
    sum_max: dict[str, float] = {cid: 0.0 for cid in cluster_ids}
    cnt_max: dict[str, int] = {cid: 0 for cid in cluster_ids}
    for point_scores in scores_by_point.values():
        if not point_scores:
            continue
        winner = max(point_scores, key=lambda cid: point_scores[cid])
        if winner in sum_max:
            sum_max[winner] += point_scores[winner]
            cnt_max[winner] += 1

    low_cohesion: list[ClusterCohesion] = []
    for cid in cluster_ids:
        if cnt_max[cid] == 0:
            continue
        mean_max = sum_max[cid] / cnt_max[cid]
        if mean_max < COHESION_SPLIT_THRESHOLD:
            low_cohesion.append(ClusterCohesion(
                cluster_id=cid,
                cluster_name=name_by_id.get(cid, cid),
                mean_max_prob=round(mean_max, 3),
            ))
    low_cohesion.sort(key=lambda c: c.mean_max_prob)

    return ClusterUncertainty(overlaps=overlaps, low_cohesion=low_cohesion)


# ---------------------------------------------------------------------------
# Legacy per-point API — kept for backwards compatibility with existing tests.
# New code should use f_cluster_uncertainty() instead.
# ---------------------------------------------------------------------------

@dataclass
class BoundaryPoint:
    point_id: str
    text_preview: str
    cluster_scores: dict[str, float]
    uncertainty_score: float


def f_uncertainty(
    session_id: str,
    db: Session,
    top_n: int = 10,
) -> list[BoundaryPoint]:
    """Legacy per-point uncertainty. Prefer f_cluster_uncertainty()."""
    from sqlalchemy import func as _func
    from src.models import DataPoint

    cluster_ids = [
        cid
        for (cid,) in db.query(DbCluster.id)
        .filter(DbCluster.session_id == session_id)
        .all()
    ]
    if not cluster_ids:
        return []

    latest_turn = (
        db.query(_func.max(SoftAssignment.turn_number))
        .filter(SoftAssignment.cluster_id.in_(cluster_ids))
        .scalar()
    )
    if latest_turn is None:
        return []

    assignments = (
        db.query(SoftAssignment)
        .filter(
            SoftAssignment.cluster_id.in_(cluster_ids),
            SoftAssignment.turn_number == latest_turn,
        )
        .all()
    )

    scores_by_point: dict[str, dict[str, float]] = {}
    for a in assignments:
        scores_by_point.setdefault(a.data_point_id, {})[a.cluster_id] = a.probability

    point_ids = list(scores_by_point.keys())
    data_points = db.query(DataPoint).filter(DataPoint.id.in_(point_ids)).all()
    text_by_id: dict[str, str] = {}
    for dp in data_points:
        title = (dp.data or {}).get("title", "") or ""
        text = (dp.data or {}).get("text", "") or ""
        text_by_id[dp.id] = f"{title} {text}".strip()

    results: list[BoundaryPoint] = []
    for point_id, cluster_scores in scores_by_point.items():
        uncertainty_score = 1.0 - max(cluster_scores.values())
        results.append(BoundaryPoint(
            point_id=point_id,
            text_preview=text_by_id.get(point_id, "")[:200],
            cluster_scores=cluster_scores,
            uncertainty_score=round(uncertainty_score, 4),
        ))

    results.sort(key=lambda p: p.uncertainty_score, reverse=True)
    return results[:top_n]
