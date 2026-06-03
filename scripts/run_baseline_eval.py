"""No-dialogue BASELINE arm for the clustering eval.

Answers the brief's core experimental question — *does conversational refinement
improve clustering over plain clustering?* — by measuring the quality of the
INITIAL k-means clustering with **no oracle dialogue**, using the same metrics
the conversational eval uses, reported with **bootstrap 95% CIs**.

This is the BASELINE (control) arm. The conversational arm is produced by
``run_scenario_eval.py`` / ``run_persona_eval.py``; its summary now also carries
bootstrap CIs (see ``src/eval/eval_report.py``). Compare the two arms' A1/B2 per
dataset to read the effect of dialogue.

Metrics — **A1 (silhouette)** and **B2 (coherence)** only. These are *intrinsic*
clustering-quality metrics, defined with or without dialogue, so they make a fair
baseline. B1/B3/B4 are dialogue-dependent (oracle compliance / contradiction /
the synthesis judge that forgives a hard oracle) and have no meaning for a
no-dialogue control, so they are deliberately excluded.

Design / safety:
  * Embeddings are read **read-only** from the live DB (already computed — no
    re-embedding). The clustering + eval run in a **throwaway in-memory DB**, so
    the live DB is never written to (no risk of polluting it with sessions).
  * Sampling for B2 mirrors ``backend/routers/sessions.py`` exactly (top-3 most
    representative + bottom-2 weakest-fitting per cluster, resolved via
    ``dp.text``) so the baseline B2 is measured identically to the live endpoint.
  * A1 (silhouette) is deterministic. B2 calls the LLM judge once per dataset;
    with ``HARNESS_DRY_RUN=true`` the judge is mocked to 0.0 (plumbing smoke).

Usage:
    PYTHONPATH=. python scripts/run_baseline_eval.py                     # all datasets
    PYTHONPATH=. python scripts/run_baseline_eval.py --datasets 20_newsgroups --k 6
    PYTHONPATH=. python scripts/run_baseline_eval.py --out reports/baseline
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.logger as logger  # noqa: E402
from backend.session_state import build_session_state  # noqa: E402
from src.engine.cluster_naming import name_clusters  # noqa: E402
from src.engine.f_eval import f_eval_coherence  # noqa: E402
from src.engine.initial_clustering import KMEANS_RANDOM_STATE, initial_clustering  # noqa: E402
from src.harness import DRY_RUN  # noqa: E402
from src.models import Base, ChatSession, DataPoint, Dataset, SoftAssignment  # noqa: E402

DEFAULT_DB = "data/demo_database.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BOOTSTRAP_ITERS = 10000

# Sensible default k per dataset (topic datasets want more clusters than the
# binary-sentiment ones). Override for all datasets with --k. For a fair
# baseline-vs-conversational comparison, use the same k the conversational run
# started from (its k_initial).
DEFAULT_K = {"20_newsgroups": 6, "amazon_reviews": 4, "imdb_reviews": 4}
_FALLBACK_K = 4

# B2 sampling — must match backend/routers/sessions.py so the baseline B2 is
# measured the same way as the live endpoint.
N_TOP_COHERENCE = 3
N_BOTTOM_COHERENCE = 2


# ---------------------------------------------------------------------------
# Bootstrap CI (percentile) — same method as run_generalization_stability_eval
# and src/eval/eval_report.py, for cross-report consistency.
# ---------------------------------------------------------------------------
def _bootstrap_mean(values, iters: int = BOOTSTRAP_ITERS, seed: int = 0):
    """Percentile bootstrap 95% CI for a mean. Returns (mean, lo, hi)."""
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return (0.0, float("nan"), float("nan"))
    if len(arr) == 1:
        return (float(arr[0]), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(iters, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(arr.mean()), float(lo), float(hi))


# ---------------------------------------------------------------------------
# Read the dataset's points (read-only) from the live DB
# ---------------------------------------------------------------------------
def _load_points(db_path: str, dataset_name: str, limit: int | None):
    """Return [(id, text, embedding_list), ...] for *dataset_name*, read-only.

    Never writes to the live DB. Skips points without an embedding.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    rows: list[tuple[str, str, list]] = []
    with Session() as s:
        ds = s.query(Dataset).filter(Dataset.name == dataset_name).one_or_none()
        if ds is None:
            raise SystemExit(f"[baseline] dataset '{dataset_name}' not found in {db_path}")
        q = (
            s.query(DataPoint)
            .filter(DataPoint.dataset_id == ds.id, DataPoint.embedding.is_not(None))
        )
        if limit:
            q = q.limit(limit)
        for dp in q.all():
            rows.append((dp.id, dp.text, dp.embedding))
    engine.dispose()
    return rows


# ---------------------------------------------------------------------------
# A1 — per-point silhouette for a single-session in-memory DB at turn 0
# ---------------------------------------------------------------------------
def _silhouette_per_point(db, session_id: str) -> list[float]:
    """Per-point silhouette of the turn-0 hard partition. Empty if degenerate."""
    rows = (
        db.query(SoftAssignment)
        .filter(SoftAssignment.turn_number == 0)
        .all()
    )
    snap: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for r in rows:
        snap[r.data_point_id][r.cluster_id] = r.probability
    point_ids = list(snap.keys())
    if len(point_ids) < 3:
        return []
    hard = {pid: max(dist, key=dist.get) for pid, dist in snap.items()}
    emb_by_id = {
        dp.id: dp.embedding
        for dp in db.query(DataPoint).filter(DataPoint.id.in_(point_ids)).all()
        if dp.embedding is not None
    }
    point_ids = [pid for pid in point_ids if pid in emb_by_id]
    labels = [hard[pid] for pid in point_ids]
    if len(set(labels)) < 2 or len(point_ids) <= len(set(labels)):
        return []
    X = np.asarray([emb_by_id[pid] for pid in point_ids], dtype=np.float64)
    y = LabelEncoder().fit_transform(labels)
    return [float(s) for s in silhouette_samples(X, y)]


# ---------------------------------------------------------------------------
# B2 — coherence sampling (mirrors backend/routers/sessions.py) + judge
# ---------------------------------------------------------------------------
def _coherence_samples(db, state) -> list[dict]:
    cluster_ids = [c.id for c in state.clusters]
    if not cluster_ids:
        return []
    latest_turn = (
        db.query(func.max(SoftAssignment.turn_number))
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
    best_by_point: dict[str, tuple[str, float]] = {}
    for a in assignments:
        cur = best_by_point.get(a.data_point_id)
        if cur is None or a.probability > cur[1]:
            best_by_point[a.data_point_id] = (a.cluster_id, a.probability)
    members: dict[str, list[tuple[str, float]]] = {cid: [] for cid in cluster_ids}
    for pid, (cid, prob) in best_by_point.items():
        members[cid].append((pid, prob))
    for cid in members:
        members[cid].sort(key=lambda it: it[1], reverse=True)

    plan: dict[str, tuple[list, list]] = {}
    needed: set[str] = set()
    for cid, pts in members.items():
        if not pts:
            continue
        top = pts[:N_TOP_COHERENCE]
        bottom = pts[N_TOP_COHERENCE:][-N_BOTTOM_COHERENCE:]
        plan[cid] = (top, bottom)
        for pid, _ in top + bottom:
            needed.add(pid)

    text_by_id: dict[str, str] = {}
    if needed:
        for dp in db.query(DataPoint).filter(DataPoint.id.in_(needed)).all():
            text_by_id[dp.id] = dp.text or dp.id

    cluster_by_id = {c.id: c for c in state.clusters}
    samples: list[dict] = []
    for cid, (top, bottom) in plan.items():
        samples.append({
            "cluster": cluster_by_id[cid],
            "top_texts": [text_by_id.get(pid, pid) for pid, _ in top],
            "bottom_texts": [text_by_id.get(pid, pid) for pid, _ in bottom],
        })
    return samples


def _coherence_per_cluster(db, session) -> list[dict]:
    """B2 judge over the session's clusters. Returns f_eval_coherence results."""
    state = build_session_state(db, session)
    samples = _coherence_samples(db, state)
    if not samples:
        return []
    return f_eval_coherence(state, samples)


# ---------------------------------------------------------------------------
# Evaluate ONE dataset's no-dialogue baseline
# ---------------------------------------------------------------------------
def _evaluate_dataset(dataset_name: str, k: int, db_path: str, limit: int | None) -> dict:
    print(f"\n── {dataset_name}  (k={k}) ──────────────────────────────────")
    points = _load_points(db_path, dataset_name, limit)
    print(f"   loaded {len(points)} embedded points (read-only)")

    # Throwaway in-memory DB — the live DB is never written to.
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    ds_id, sess_id = "ds-baseline", f"baseline-{dataset_name}"
    db.add(Dataset(id=ds_id, name=dataset_name, description=""))
    session = ChatSession(id=sess_id, dataset_id=ds_id,
                          embedding_model=EMBEDDING_MODEL, status="active")
    db.add(session)
    dps = [DataPoint(id=pid, dataset_id=ds_id, text=text, embedding=emb)
           for pid, text, emb in points]
    db.add_all(dps)
    db.flush()

    clusters, assignments, fit_sil = initial_clustering(
        dps, k=k, session_id=sess_id, turn_number=0
    )
    # Name the clusters (1 LLM call) so the B2 judge sees the same named clusters
    # the live eval would — keeps the baseline B2 comparable to the conversational arm.
    if not DRY_RUN:
        try:
            name_clusters(clusters, assignments, dps)
        except Exception as e:  # noqa: BLE001 — naming is best-effort
            print(f"   ⚠ naming failed ({e}); proceeding with placeholder names")
    db.add_all(clusters)
    db.add_all(assignments)
    db.commit()
    print(f"   clustered: k={k}, seed={KMEANS_RANDOM_STATE}, fit silhouette={fit_sil:.4f}")

    # A1 — per-point silhouette + bootstrap CI
    sil_points = _silhouette_per_point(db, sess_id)
    a1_mean, a1_lo, a1_hi = _bootstrap_mean(sil_points)

    # B2 — per-cluster coherence + bootstrap CI
    b2_error = None
    coh_results: list[dict] = []
    try:
        coh_results = _coherence_per_cluster(db, session)
    except Exception as e:  # noqa: BLE001 — provider error must not sink A1
        b2_error = f"{type(e).__name__}: {e}"
        print(f"   ⚠ B2 judge unavailable: {b2_error}")
    coh_vals = [float(r["coherence"]) for r in coh_results]
    b2_mean, b2_lo, b2_hi = _bootstrap_mean(coh_vals)
    b2_min = min(coh_vals) if coh_vals else None

    db.close()
    engine.dispose()

    print(f"   A1 silhouette: mean {a1_mean:.4f}  95% CI [{a1_lo:.4f}, {a1_hi:.4f}]  "
          f"({len(sil_points)} pts)")
    if b2_error:
        print(f"   B2 coherence: UNAVAILABLE ({b2_error})")
    elif coh_vals:
        print(f"   B2 coherence: mean {b2_mean:.3f}  95% CI [{b2_lo:.3f}, {b2_hi:.3f}]  "
              f"min {b2_min:.3f}  ({len(coh_vals)} clusters)")

    return {
        "arm": "baseline_no_dialogue",
        "dataset": dataset_name,
        "k": k,
        "n_points": len(points),
        "seed": KMEANS_RANDOM_STATE,
        "fit_silhouette": fit_sil,
        "A1": {
            "metric": "silhouette",
            "n": len(sil_points),
            "mean": a1_mean,
            "ci_lo": a1_lo,
            "ci_hi": a1_hi,
        },
        "B2": {
            "metric": "coherence",
            "n_clusters": len(coh_vals),
            "mean": b2_mean if coh_vals else None,
            "ci_lo": b2_lo if coh_vals else None,
            "ci_hi": b2_hi if coh_vals else None,
            "min": b2_min,
            "per_cluster": [
                {"cluster_id": r["cluster_id"],
                 "cluster_name": r.get("cluster_name", ""),
                 "coherence": float(r["coherence"])}
                for r in coh_results
            ],
            "error": b2_error,
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _write_report(records: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    lines = ["# Baseline (no-dialogue) clustering evaluation\n"]
    lines.append(f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append(
        "\nThe **control arm**: quality of the initial k-means clustering with "
        "**no oracle dialogue**. A1 (silhouette) and B2 (coherence) are measured "
        "exactly as in the live eval; B1/B3/B4 are dialogue-dependent and omitted. "
        "95% CIs are percentile bootstraps (A1 over points, B2 over clusters — so "
        "the B2 CI is coarse at small k).\n"
    )
    lines.append(
        "\nTo read the **effect of dialogue**, compare these numbers against the "
        "conversational arm's A1/B2 (with CIs) from `run_scenario_eval.py` / "
        "`run_persona_eval.py` at the same dataset and k.\n\n"
    )
    lines.append("| Dataset | k | n | A1 silhouette (mean, 95% CI) | B2 coherence (mean, 95% CI) | B2 min |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for r in records:
        a1 = r["A1"]; b2 = r["B2"]
        a1s = f"{a1['mean']:.4f} [{a1['ci_lo']:.4f}, {a1['ci_hi']:.4f}]"
        if b2.get("error"):
            b2s, b2min = f"UNAVAILABLE ({b2['error']})", "—"
        elif b2.get("mean") is not None:
            b2s = f"{b2['mean']:.3f} [{b2['ci_lo']:.3f}, {b2['ci_hi']:.3f}]"
            b2min = f"{b2['min']:.3f}"
        else:
            b2s, b2min = "(no data)", "—"
        lines.append(f"| {r['dataset']} | {r['k']} | {r['n_points']} | {a1s} | {b2s} | {b2min} |\n")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.writelines(lines)
    print(f"\nWrote {out_dir}/results.jsonl + summary.md")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_K.keys()),
                    help="dataset names to baseline (default: all three)")
    ap.add_argument("--k", type=int, default=None,
                    help="override k for ALL datasets (default: per-dataset DEFAULT_K)")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"live DB path (default {DEFAULT_DB})")
    ap.add_argument("--out", default=None,
                    help="output dir (default reports/baseline-<timestamp>/)")
    ap.add_argument("--limit", type=int, default=None, help="cap points per dataset (smoke)")
    args = ap.parse_args()

    if DRY_RUN:
        print("⚠ HARNESS_DRY_RUN=true — B2 (coherence) is MOCKED (0.0). "
              "A1 (silhouette) is real. Run without dry-run for real B2.\n")

    # Engine's clustering log → throwaway file (these are not tracked sessions).
    logger._clustering_log_path = Path(tempfile.gettempdir()) / "baseline_clustering.jsonl"

    out_dir = args.out or f"reports/baseline-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    records = []
    for name in args.datasets:
        k = args.k or DEFAULT_K.get(name, _FALLBACK_K)
        records.append(_evaluate_dataset(name, k, args.db, args.limit))

    _write_report(records, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
