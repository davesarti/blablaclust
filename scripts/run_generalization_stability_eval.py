"""Online generalization eval — does a CONVERGED clustering stay coherent as
new data arrives?

The project brief's "generalization" question, framed *operationally* and
**without ground-truth labels** (clustering is unsupervised; the oracle is the
objective). The test is not "does the clustering recover a hidden category" —
it is "once the oracle is happy, do new data points that arrive into the running
system keep the clustering coherent?". We answer it by re-evaluating the two
relevant metrics already frozen in ``docs/quality_specs.md`` at two snapshots
around an ingestion event — **no new metric**:

  1. Reach a converged session. Here we cluster the train split with the real
     engine (k-means) and treat that snapshot as the converged state. No LLM
     refinement loop is run (that needs the oracle and is the eval harness's
     job); the generalization *procedure* — freeze centroids, ingest, re-eval —
     is identical no matter how convergence was reached. To run against a real
     oracle-converged session instead, load its snapshot from the DB and pass
     its centroids to ``ingest_points`` (same call).
  2. t0 — pre-ingestion eval: A1 (silhouette) and B2 (cluster coherence) on the
     converged state.
  3. Ingest the frozen split as the stream of *new arrivals*: embed →
     ``assign_nearest`` against the frozen centroids → write a fresh full
     snapshot at ``turn + 1`` (``ingest_points``). Centroids stay frozen; the
     pre-existing points are carried forward verbatim, never re-evaluated.
  4. t1 — post-ingestion eval: A1 and B2 again, plus an A1 sub-aggregate over
     just the newly-ingested batch ("do the new points sit cleanly relative to
     the centroids?"). B2's bottom-2 stress sample naturally picks up bad new
     members.
  5. Report **paired Δ + bootstrap 95% CI** on A1 (over the common, pre-existing
     points) and B2 (over the per-cluster scores) between t0 and t1. With
     ``--batches N`` the ingest/eval loop repeats to trace a drift curve.

LABELS ARE OUT OF SCOPE. This script reads only ``title,text`` from every CSV —
never the ``label`` column. Generalization is consistency under growth, not
accuracy against a hidden category.

A1 (silhouette) is deterministic and needs no LLM. B2 (coherence) calls the
LLM judge; with ``HARNESS_DRY_RUN=true`` the judge is mocked and B2 collapses to
0.0 (use it only to smoke-test the plumbing). Run without dry-run for real B2.

Usage:
    PYTHONPATH=. python scripts/run_generalization_stability_eval.py            # 20NG, k=6
    PYTHONPATH=. python scripts/run_generalization_stability_eval.py \
        --base data/train.csv --new data/frozen_eval.csv --k 2                  # Amazon
    PYTHONPATH=. python scripts/run_generalization_stability_eval.py --batches 4  # drift curve
    PYTHONPATH=. python scripts/run_generalization_stability_eval.py --limit 400  # quick smoke
"""

from __future__ import annotations

import argparse
import collections
import csv

import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.logger as logger
from backend.session_state import build_session_state
from src.dataset_processing.text_cleaning import clean_text
from src.engine.f_eval import f_eval_coherence
from src.engine.generalization import centroids_from_snapshot, ingest_points
from src.engine.initial_clustering import KMEANS_RANDOM_STATE, initial_clustering
from src.harness import DRY_RUN
from src.models import Base, ChatSession, DataPoint, Dataset, SoftAssignment

DATASET_ID = "ds-gen-stability"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SESSION_ID = "gen-stability-eval"
BOOTSTRAP_ITERS = 10000
# Coherence sampling — mirror backend/routers/sessions.py so B2 here matches the
# live eval endpoint (top-3 most representative + bottom-2 weakest-fitting edge
# cases; the bottom-2 stress-test the cluster and catch bad new members).
N_TOP_COHERENCE = 3
N_BOTTOM_COHERENCE = 2


# ---------------------------------------------------------------------------
# Data loading — title,text ONLY (labels are out of scope)
# ---------------------------------------------------------------------------
def _read_texts(path: str, limit: int | None = None) -> list[str]:
    """Cleaned ``title + text`` per row.

    Never touches the ``label`` column — clustering is unsupervised and labels
    are out of scope for the generalization eval (see ``docs/quality_specs.md``).
    """
    texts: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = clean_text(row.get("title") or "", row.get("text") or "")
            if not text:
                continue
            texts.append(text)
            if limit is not None and len(texts) >= limit:
                break
    return texts


def _embed(texts: list[str], model) -> np.ndarray:
    return model.encode(
        texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# Bootstrap CIs (statistics, not metrics)
# ---------------------------------------------------------------------------
def _bootstrap_mean(
    values: np.ndarray, iters: int = BOOTSTRAP_ITERS, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap 95% CI for a mean. Returns (mean, lo, hi)."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(iters, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(values.mean()), float(lo), float(hi))


def _bootstrap_paired(
    diffs: np.ndarray, iters: int = BOOTSTRAP_ITERS, seed: int = 0
) -> tuple[float, float, float]:
    """Paired-difference bootstrap 95% CI. ``diffs`` are per-unit (t1 - t0)
    values for units common to both snapshots. Returns (mean_diff, lo, hi)."""
    return _bootstrap_mean(diffs, iters=iters, seed=seed)


# ---------------------------------------------------------------------------
# Snapshot helpers (read straight from the DB)
# ---------------------------------------------------------------------------
def _snapshot_at(db, turn: int) -> dict[str, dict[str, float]]:
    """``data_point_id -> {cluster_id: probability}`` at ``turn`` (single-session DB)."""
    rows = (
        db.query(SoftAssignment)
        .filter(SoftAssignment.turn_number == turn)
        .all()
    )
    snap: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for r in rows:
        snap[r.data_point_id][r.cluster_id] = r.probability
    return snap


def _hard_silhouette(db, snapshot: dict[str, dict[str, float]]) -> dict[str, float]:
    """Per-point silhouette for a snapshot, keyed by data_point_id.

    Hard label = argmax of each point's distribution; X = its stored embedding.
    Returns {point_id: silhouette}; empty if the snapshot is too small/degenerate.
    """
    point_ids = list(snapshot.keys())
    if len(point_ids) < 3:
        return {}
    hard = {pid: max(dist, key=dist.get) for pid, dist in snapshot.items()}
    emb_by_id = {
        dp.id: dp.embedding
        for dp in db.query(DataPoint).filter(DataPoint.id.in_(point_ids)).all()
        if dp.embedding is not None
    }
    point_ids = [pid for pid in point_ids if pid in emb_by_id]
    labels = [hard[pid] for pid in point_ids]
    if len(set(labels)) < 2 or len(point_ids) <= len(set(labels)):
        return {}
    X = np.asarray([emb_by_id[pid] for pid in point_ids], dtype=np.float64)
    y = LabelEncoder().fit_transform(labels)
    sil = silhouette_samples(X, y)
    return dict(zip(point_ids, (float(s) for s in sil)))


# ---------------------------------------------------------------------------
# B2 coherence — ported from backend/routers/sessions.py so the script does not
# import the FastAPI app. Same top-3/bottom-2 sampling, same judge.
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
            # DataPoint moved from a `data` JSON blob to a flat `text` column
            # (commit 0eeb6ce); mirror the live endpoint's `dp.text or dp.id`.
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


def _coherence_by_cluster(db, session) -> dict[str, float]:
    """{cluster_id: coherence} via the B2 judge on the session's latest snapshot."""
    state = build_session_state(db, session)
    samples = _coherence_samples(db, state)
    if not samples:
        return {}
    results = f_eval_coherence(state, samples)
    return {r["cluster_id"]: float(r["coherence"]) for r in results}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"


def _report_a1(s0: dict[str, float], s1: dict[str, float], new_ids: set[str]) -> None:
    common = [pid for pid in s0 if pid in s1]
    mean0 = float(np.mean(list(s0.values()))) if s0 else 0.0
    mean1 = float(np.mean(list(s1.values()))) if s1 else 0.0
    print("A1 — silhouette (cluster goodness)")
    print(f"    t0 mean silhouette: {mean0:.4f}  ({len(s0)} points)")
    print(f"    t1 mean silhouette: {mean1:.4f}  ({len(s1)} points)")
    if common:
        diffs = np.array([s1[pid] - s0[pid] for pid in common])
        m, lo, hi = _bootstrap_paired(diffs)
        print(f"    paired Δ on the {len(common)} pre-existing points: {_fmt_ci(m, lo, hi)}")
        verdict = "degraded" if hi < 0 else ("improved" if lo > 0 else "held (CI spans 0)")
        print(f"      → existing-point cohesion {verdict}")
    new_vals = [s1[pid] for pid in new_ids if pid in s1]
    if new_vals:
        m, lo, hi = _bootstrap_mean(np.array(new_vals))
        print(f"    sub-aggregate — {len(new_vals)} NEW points: mean silhouette "
              f"{m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")


def _report_b2(c0: dict[str, float], c1: dict[str, float]) -> None:
    if not c0 or not c1:
        print("B2 — coherence: skipped (no clusters / no judge output)")
        return
    common = [cid for cid in c0 if cid in c1]
    mean0, min0 = float(np.mean(list(c0.values()))), float(min(c0.values()))
    mean1, min1 = float(np.mean(list(c1.values()))), float(min(c1.values()))
    print("B2 — cluster coherence (LLM judge)")
    print(f"    t0  mean {mean0:.3f}  min {min0:.3f}  ({len(c0)} clusters)")
    print(f"    t1  mean {mean1:.3f}  min {min1:.3f}  ({len(c1)} clusters)")
    if common:
        diffs = np.array([c1[cid] - c0[cid] for cid in common])
        m, lo, hi = _bootstrap_paired(diffs)
        print(f"    paired Δ over {len(common)} clusters: {_fmt_ci(m, lo, hi)}")
        print(f"      (n_clusters is small — the CI is correspondingly coarse)")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="data/20newsgroups_train.csv",
                        help="corpus to cluster into the converged state")
    parser.add_argument("--new", default="data/20newsgroups_frozen.csv",
                        help="stream of NEW arrivals to ingest")
    parser.add_argument("--k", type=int, default=6, help="number of clusters")
    parser.add_argument("--batches", type=int, default=1,
                        help="split the new arrivals into N successive ingestions (drift curve)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap rows per split (quick smoke runs)")
    parser.add_argument("--b2-every-batch", action="store_true",
                        help="run the B2 judge after every batch, not just t0/final")
    args = parser.parse_args()

    if DRY_RUN:
        print("⚠ HARNESS_DRY_RUN=true — B2 (coherence) is MOCKED (collapses to 0.0).")
        print("  A1 (silhouette) is deterministic and unaffected. Run without dry-run for real B2.\n")

    # Engine's clustering log → throwaway file (this is not a tracked session).
    import tempfile
    from pathlib import Path
    logger._clustering_log_path = Path(tempfile.gettempdir()) / "gen_stability_clustering.jsonl"

    print("Loading + cleaning splits (title,text only — no labels)…")
    base_texts = _read_texts(args.base, args.limit)
    new_texts = _read_texts(args.new, args.limit)
    print(f"  base (→ converged): {len(base_texts)} rows")
    print(f"  new  (→ ingested):  {len(new_texts)} rows")

    print(f"Loading {EMBEDDING_MODEL} and embedding (one-time, ~60-90s on CPU)…")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    base_emb = _embed(base_texts, model)
    new_emb = _embed(new_texts, model)

    # ── Converged session in a throwaway in-memory DB ────────────────────────
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    db.add(Dataset(id=DATASET_ID, name="generalization", description=""))
    session = ChatSession(id=SESSION_ID, dataset_id=DATASET_ID,
                          embedding_model=EMBEDDING_MODEL, status="converged")
    db.add(session)

    base_points = []
    for i, emb in enumerate(base_emb):
        dp = DataPoint(id=f"base-{i}", dataset_id=DATASET_ID,
                       text=base_texts[i], embedding=emb.tolist())
        db.add(dp)
        base_points.append(dp)
    db.flush()

    clusters, assignments, sil = initial_clustering(
        base_points, k=args.k, session_id=SESSION_ID, turn_number=0
    )
    for c in clusters:
        db.add(c)
    for a in assignments:
        db.add(a)
    db.commit()
    print(f"\nConverged: {len(base_points)} points → k={args.k} "
          f"(k-means, seed={KMEANS_RANDOM_STATE}, fit silhouette={sil:.4f}).")

    # Freeze the convergence centroids ONCE.
    snap0 = _snapshot_at(db, 0)
    emb0 = {f"base-{i}": e for i, e in enumerate(base_emb)}
    centroid_ids, centroids = centroids_from_snapshot(emb0, snap0)

    # ── t0 eval (BEFORE any ingestion) ───────────────────────────────────────
    # A1 (silhouette) is deterministic and always computed. B2 (coherence) calls
    # the LLM judge; a provider failure must not sink the whole eval — degrade to
    # "B2 unavailable" and still report the A1 generalization signal.
    sil0 = _hard_silhouette(db, snap0)
    b2_error: str | None = None
    try:
        coh0 = _coherence_by_cluster(db, session)
    except Exception as e:  # noqa: BLE001 — any provider error degrades gracefully
        coh0, b2_error = {}, f"{type(e).__name__}: {e}"
        print(f"⚠ B2 judge unavailable at t0 — reporting A1 only.\n  {b2_error}")

    # ── Ingest the new arrivals (1+ batches) ─────────────────────────────────
    batches = np.array_split(np.arange(len(new_emb)), max(1, args.batches))
    drift: list[tuple[int, float, int]] = []  # (turn, mean_silhouette, n_points)
    last_turn = 0
    for b, idx in enumerate(batches):
        if len(idx) == 0:
            continue
        batch_points = [
            DataPoint(id=f"new-{int(i)}", dataset_id=DATASET_ID,
                      text=new_texts[int(i)],
                      embedding=new_emb[int(i)].tolist())
            for i in idx
        ]
        last_turn, _ = ingest_points(SESSION_ID, batch_points, centroid_ids, centroids, db)
        db.commit()
        snap = _snapshot_at(db, last_turn)
        sil_b = _hard_silhouette(db, snap)
        mean_b = float(np.mean(list(sil_b.values()))) if sil_b else 0.0
        drift.append((last_turn, mean_b, len(snap)))
        print(f"  ingested batch {b + 1}/{len(batches)}: +{len(batch_points)} points "
              f"→ turn {last_turn}, {len(snap)} total, mean silhouette {mean_b:.4f}")

    # ── t1 eval (final state) ────────────────────────────────────────────────
    snap1 = _snapshot_at(db, last_turn)
    sil1 = _hard_silhouette(db, snap1)
    new_ids = {pid for pid in snap1 if pid not in snap0}
    coh1 = {}
    if b2_error is None:
        try:
            coh1 = _coherence_by_cluster(db, session)
        except Exception as e:  # noqa: BLE001
            b2_error = f"{type(e).__name__}: {e}"
            print(f"⚠ B2 judge unavailable at t1.\n  {b2_error}")

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ONLINE GENERALIZATION — A1 / B2 around an ingestion event (no labels)")
    print("=" * 70)
    print(f"converged: {len(snap0)} points, k={len(centroid_ids)}   "
          f"ingested: {len(new_ids)} new points over {len(drift)} batch(es)\n")
    _report_a1(sil0, sil1, new_ids)
    print()
    if b2_error:
        print("B2 — cluster coherence: UNAVAILABLE (LLM judge failed)")
        print(f"    {b2_error}")
        print("    A1 above is deterministic and complete; re-run with a working")
        print("    LLM provider to get the B2 paired Δ.")
    else:
        _report_b2(coh0, coh1)
    if len(drift) > 1:
        print("\nDrift curve (mean silhouette as new data accrues):")
        print(f"    t0  : {float(np.mean(list(sil0.values()))):.4f}  ({len(snap0)} pts)")
        for turn, mean_b, n in drift:
            print(f"    t{turn}  : {mean_b:.4f}  ({n} pts)")
    print("=" * 70)
    print("Generalization holds when A1 stays put (paired Δ CI spans 0 or is "
          "positive) and B2 does not drop. A decline means new data is breaking "
          "the converged structure.")
    print("=" * 70)

    db.close()


if __name__ == "__main__":
    main()
