"""Online generalization eval — does a CONVERGED clustering stay coherent as
new data arrives?

The project brief's "generalization" question, framed *operationally* and
**without ground-truth labels** (clustering is unsupervised; the oracle is the
objective). The test is not "does the clustering recover a hidden category" —
it is "once the oracle is happy, do new data points that arrive into the running
system keep the clustering coherent?". We answer with three deterministic /
LLM-assisted signals around an ingestion event:

  1. Reach a converged session — REALISTICALLY. We spin up a throwaway in-memory
     session, cluster the base split with the real engine (GMM, k-means
     fallback), then drive a full LLM-as-oracle conversation (default persona
     ``curious_explorer``) through the same engine the live API runs — in-process,
     no server, demo_database.db untouched. The converged state is therefore
     oracle-shaped, with real turns and a real A1–B4 eval, not a bare clustering
     snapshot. The frozen geometry used for generalization is the snapshot at the
     LAST conversation turn. (Pass ``--no-conversation`` to converge on the bare
     initial clustering instead, for an A/B comparison.) The throwaway session is
     closed and discarded at the end.
  2. t0 — pre-ingestion eval: A1 (silhouette) and B2 (cluster coherence) on the
     converged state, plus A4 calibration (per-cluster d²_95 / μ / σ over base
     in-cluster squared distances; temperature-free, frozen for the run).
  3. Ingest the frozen split as the stream of *new arrivals*: embed →
     ``assign_gmm_posterior`` against the frozen GMM parameters → write a
     fresh full snapshot at ``turn + 1`` (``ingest_points``). All GMM params
     stay frozen; the
     pre-existing points are carried forward verbatim, never re-evaluated.
  4. t1 — post-ingestion eval: A1 and B2 again, plus an A1 sub-aggregate over
     just the newly-ingested batch ("do the new points sit cleanly relative to
     the centroids?"). B2's bottom-2 stress sample naturally picks up bad new
     members. A4 scores every new point against its assigned cluster's base
     reference: OOD when d² > d²_95(c); the headline is the pooled OOD rate
     (baseline ≈ 5% under the null).
  5. Report **paired Δ + bootstrap 95% CI** on A1 (over the common, pre-existing
     points) and B2 (over the per-cluster scores) between t0 and t1, plus A4's
     pooled OOD rate, mean z + CI, and per-cluster breakdown. With ``--batches
     N`` the ingest/eval loop repeats to trace a drift curve on A1 and A4.

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
from src.engine.generalization import (
    OOD_PERCENTILE,
    assign_gmm_posterior,
    assignment_ood,
    calibrate_distance_reference,
    gmm_params_from_snapshot,
    ingest_points,
)
from src.engine.initial_clustering import KMEANS_RANDOM_STATE, initial_clustering
from src.eval.llm_oracle import LLMOracle, OracleResponseError
from src.eval.oracle_view import build_oracle_view
from src.eval.persona import load_persona
from src.harness import DRY_RUN
from src.models import Base, ChatSession, DataPoint, Dataset, SoftAssignment
from src.schemas import InputOracle

DATASET_ID = "ds-gen-stability"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SESSION_ID = "gen-stability-eval"
BOOTSTRAP_ITERS = 10000

# Default LLM-oracle persona that drives the throwaway session's conversation.
# curious_explorer exercises real refinement (split + rename, then satisfied),
# is dataset-agnostic in its goal, and never asks for a semantic re-embed —
# keeping the converged geometry in the original embedding space so the held-out
# ingestion stays in the same space. Override with --persona.
DEFAULT_PERSONA = "personas/curious_explorer.json"
DEFAULT_MAX_TURNS = 12
# Transient engine 5xx (e.g. a 502 when the provider returns non-JSON for
# f_output) are retried this many times before the turn is abandoned. The retry
# re-rolls the LLM call, which usually succeeds on the second attempt.
_TURN_RETRIES = 2
# Shown to the oracle as the system's opening message (mirrors run_persona_eval).
_INITIAL_SYSTEM_PROMPT = (
    "Initial clustering is ready. Tell me how you'd like it changed, or "
    "let me know if it already looks right."
)
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
    """{cluster_id: coherence} via the B2 judge on the session's latest snapshot.

    The coherence judge is flaky: a transient provider hiccup makes it return
    0.0 for every cluster, which a converged clustering essentially never earns
    genuinely and would corrupt the B2 paired Δ (an all-zero t0 against a real
    t1 fabricates a huge spurious gain). When every score comes back 0.0 we
    retry — the same transient-failure guard used elsewhere in this script.
    """
    state = build_session_state(db, session)
    samples = _coherence_samples(db, state)
    if not samples:
        return {}
    scores: dict[str, float] = {}
    for attempt in range(_TURN_RETRIES + 1):
        results = f_eval_coherence(state, samples)
        scores = {r["cluster_id"]: float(r["coherence"]) for r in results}
        if scores and any(v != 0.0 for v in scores.values()):
            return scores
        if attempt < _TURN_RETRIES:
            print(f"⚠ B2 coherence judge returned all-zeros "
                  f"(likely a transient failure) — retrying {attempt + 1}/{_TURN_RETRIES}")
    return scores


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


def _report_a4(batches: list[tuple[list[str], dict]]) -> None:
    """A4 — distance-based OOD on new arrivals (temperature-free).

    ``batches`` is a list of ``(assigned_clusters, scores)`` pairs, where
    ``scores`` is the output of :func:`assignment_ood`. We pool per-point arrays
    across batches and report the pooled OOD rate (baseline ≈ 5% under the null),
    pooled mean z + bootstrap 95% CI, and a per-cluster breakdown.
    """
    if not batches:
        print("A4 — OOD on new arrivals: skipped (no batches)")
        return
    all_z = np.concatenate([s["z"] for _, s in batches])
    all_is_ood = np.concatenate([s["is_ood"] for _, s in batches])
    all_assigned: list[str] = []
    for assigned, _ in batches:
        all_assigned.extend(assigned)
    baseline = 1.0 - OOD_PERCENTILE / 100.0
    pooled_rate = float(all_is_ood.mean())
    # Bootstrap CI on mean z. If any z is ±inf (singleton calibration), the CI
    # collapses to inf — surface that honestly rather than silently dropping.
    finite = np.isfinite(all_z)
    if finite.all():
        mean_z, lo, hi = _bootstrap_mean(all_z)
        ci_str = f"{mean_z:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
    else:
        ci_str = (
            f"{float(all_z.mean()):+.4f}  "
            f"(CI unavailable — {int((~finite).sum())} non-finite z from "
            f"singleton-calibrated clusters)"
        )
    print(
        "A4 — distance-based OOD on new arrivals "
        f"(temperature-free, threshold = base d²_{int(OOD_PERCENTILE)})"
    )
    print(f"    pooled OOD rate: {pooled_rate * 100:.1f}%   "
          f"(baseline ~{baseline * 100:.1f}%)")
    print(f"    pooled mean z:   {ci_str}")

    print("    per-cluster:")
    for cid in sorted(set(all_assigned)):
        mask = np.array([c == cid for c in all_assigned])
        n = int(mask.sum())
        rate = float(all_is_ood[mask].mean()) * 100.0
        z_slice = all_z[mask]
        if np.isfinite(z_slice).all():
            z_str = f"mean_z={float(z_slice.mean()):+.3f}"
        else:
            z_str = "mean_z=inf (uncalibrated σ=0)"
        print(f"        {cid}  n={n:<4d}  ood={rate:5.1f}%  {z_str}")


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
# Throwaway-session conversation (in-process oracle, no live server)
# ---------------------------------------------------------------------------
# The converged state must be *realistic* — shaped by an oracle, with turns and
# an eval — not a bare initial_clustering snapshot. We drive an LLM-as-oracle
# conversation entirely in-process: the same engine the live API runs, called as
# plain functions against the in-memory DB. No HTTP server, no demo_database.db.
def _last_backend(log_path) -> str:
    """Read the actual clustering backend (gmm | kmeans) from the engine log.

    initial_clustering picks GMM and silently falls back to k-means on
    convergence failure; the only source of truth for which ran is the log line
    it just wrote. Returns 'unknown' if the log can't be read.
    """
    try:
        from pathlib import Path
        lines = Path(log_path).read_text().splitlines()
        if lines:
            import json as _json
            return _json.loads(lines[-1]).get("backend", "unknown")
    except Exception:
        pass
    return "unknown"


def _view_in_process(db, session):
    """Build the oracle's cluster-panel view from the in-memory DB.

    Mirrors run_persona_eval's _fetch_view but reads the DB directly instead of
    GET /sessions/{sid}/state + /clusters/{cid}/points. Shows up to 3 real
    member texts per active cluster so the oracle reasons over actual content.
    """
    state = build_session_state(db, session).model_dump()
    cluster_ids = [c["id"] for c in state.get("clusters") or []]
    cluster_points: dict[str, list[dict]] = {}
    if cluster_ids:
        latest_turn = (
            db.query(func.max(SoftAssignment.turn_number))
            .filter(SoftAssignment.cluster_id.in_(cluster_ids))
            .scalar()
        )
        if latest_turn is not None:
            rows = (
                db.query(SoftAssignment)
                .filter(
                    SoftAssignment.cluster_id.in_(cluster_ids),
                    SoftAssignment.turn_number == latest_turn,
                )
                .all()
            )
            best: dict[str, tuple[str, float]] = {}
            for a in rows:
                cur = best.get(a.data_point_id)
                if cur is None or a.probability > cur[1]:
                    best[a.data_point_id] = (a.cluster_id, a.probability)
            members: dict[str, list[tuple[str, float]]] = {cid: [] for cid in cluster_ids}
            for pid, (cid, prob) in best.items():
                members[cid].append((pid, prob))
            needed: set[str] = set()
            for cid in members:
                members[cid].sort(key=lambda it: it[1], reverse=True)
                for pid, _ in members[cid][:3]:
                    needed.add(pid)
            text_by_id = {
                dp.id: (dp.text or dp.id)
                for dp in db.query(DataPoint).filter(DataPoint.id.in_(needed)).all()
            }
            for cid in cluster_ids:
                cluster_points[cid] = [
                    {"id": pid, "data": {"text": text_by_id.get(pid, pid)}}
                    for pid, _ in members[cid][:3]
                ]
    return build_oracle_view(state, cluster_points, n_examples=3)


def _run_oracle_conversation(db, session, persona, max_turns: int) -> dict:
    """Drive the session to convergence with an in-process LLM oracle.

    Returns a record dict: {n_turns, terminated_by, final_turn, oracle_cost_usd,
    oracle_model, errors}. The session is mutated in place (turns persisted to
    the in-memory DB). Never raises — provider/engine errors degrade to a
    termination reason so the generalization assessment can still run.
    """
    # Import the engine entry point lazily. Importing backend.main first resolves
    # the backend.main <-> routers circular import; it also touches the demo DB's
    # schema (idempotent migration) but never its data or our throwaway session.
    import backend.main  # noqa: F401  — side effect: makes routers importable
    from backend.routers.turns import create_turn
    from fastapi import HTTPException

    oracle = LLMOracle(persona=persona, session_id=session.id, max_turns=max_turns)
    rec = {
        "n_turns": 0,
        "terminated_by": "max_turns",
        "final_turn": 0,
        "oracle_cost_usd": 0.0,
        "oracle_model": oracle.model,
        "errors": [],
    }
    system_display = _INITIAL_SYSTEM_PROMPT

    for turn_idx in range(1, max_turns + 1):
        view = _view_in_process(db, session)
        try:
            oracle.observe_system(system_display)
            oracle_turn = oracle.next_turn(view, system_display, turn_idx)
        except OracleResponseError as exc:
            rec["errors"].append(f"oracle_parse[{turn_idx}] {exc}")
            rec["terminated_by"] = "oracle_parse_error"
            break

        # Execute the oracle's turn through the engine. Transient 5xx (e.g. a 502
        # when the provider returns non-JSON for f_output — a flaky-provider
        # artifact, not an invalid request) are retried; the retry re-rolls the
        # LLM call and usually succeeds. 4xx mean the engine rejected the request
        # as invalid (e.g. a cluster dissolved by an earlier split) → feed the
        # error back so the oracle corrects course next turn, like a human would.
        result = None
        outcome = "ok"  # "ok" | "feedback" | "stop"
        for attempt in range(_TURN_RETRIES + 1):
            try:
                result = create_turn(InputOracle(**oracle_turn.body), db=db)
                outcome = "ok"
                break
            except HTTPException as exc:
                db.rollback()
                if exc.status_code and exc.status_code >= 500:
                    if attempt < _TURN_RETRIES:
                        rec["errors"].append(
                            f"turn[{turn_idx}] http={exc.status_code} transient — "
                            f"retry {attempt + 1}/{_TURN_RETRIES}"
                        )
                        continue
                    rec["errors"].append(
                        f"turn[{turn_idx}] http={exc.status_code} {exc.detail} (gave up)"
                    )
                    rec["terminated_by"] = "engine_error"
                    outcome = "stop"
                    break
                rec["errors"].append(f"turn[{turn_idx}] http={exc.status_code} {exc.detail}")
                system_display = (
                    f"That request couldn't be applied ({exc.detail}). The clustering is "
                    f"unchanged — pick a cluster currently shown and try again."
                )
                outcome = "feedback"
                break
            except Exception as exc:  # noqa: BLE001 — any other engine failure
                db.rollback()
                if attempt < _TURN_RETRIES:
                    rec["errors"].append(
                        f"turn[{turn_idx}] {type(exc).__name__} transient — "
                        f"retry {attempt + 1}/{_TURN_RETRIES}"
                    )
                    continue
                rec["errors"].append(f"turn[{turn_idx}] {type(exc).__name__}: {exc} (gave up)")
                rec["terminated_by"] = "engine_error"
                outcome = "stop"
                break

        if outcome == "feedback":
            continue
        if outcome == "stop":
            break

        rec["n_turns"] = turn_idx
        rec["final_turn"] = result.turn_number

        so = result.system_output.model_dump() if hasattr(result.system_output, "model_dump") else dict(result.system_output)
        display = so.get("display") or {}
        system_display = (display.get("content") if isinstance(display, dict) else str(display)) or ""

        if so.get("action") == "stop":
            rec["terminated_by"] = "system_stop"
            break
        if oracle_turn.satisfied:
            rec["terminated_by"] = "oracle_satisfied"
            break

    _, rec["oracle_cost_usd"] = oracle.totals
    return rec


def _session_eval(session_id: str, db):
    """Run the full A1–B4 session eval in-process (the live endpoint's logic).

    Returns the EvalResponse model, or None on failure. A1's silhouette *trend*
    is read by the endpoint from the real clustering log, which won't contain our
    throwaway session — so A1 here may be empty; the meaningful silhouette signal
    is the generalization A1 reported separately below.

    The B2 coherence judge is flaky: a transient provider hiccup makes it return
    0.0 for every cluster (which also drags B1 down), indistinguishable in the
    output from a genuinely incoherent clustering. A real converged clustering
    is essentially never B2==0.0, so we retry the whole eval when that happens —
    the same transient-failure guard applied to the conversation's 5xx turns.
    """
    import backend.main  # noqa: F401
    from backend.routers.sessions import eval_session
    last = None
    for attempt in range(_TURN_RETRIES + 1):
        try:
            ev = eval_session(session_id, force=True, db=db)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠ session eval attempt {attempt + 1} failed: {type(exc).__name__}: {exc}")
            last = None
            continue
        last = ev
        b2 = ev.B2.coherence_mean
        if b2 not in (None, 0.0):
            return ev
        if attempt < _TURN_RETRIES:
            print(f"⚠ session eval B2 coherence came back {b2} "
                  f"(likely a transient judge failure) — retrying "
                  f"{attempt + 1}/{_TURN_RETRIES}")
    return last


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
    parser.add_argument("--persona", default=DEFAULT_PERSONA,
                        help=f"LLM-oracle persona JSON driving the throwaway "
                             f"session's conversation (default {DEFAULT_PERSONA})")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                        help=f"hard cap on oracle turns (default {DEFAULT_MAX_TURNS})")
    parser.add_argument("--no-conversation", action="store_true",
                        help="skip the oracle conversation and converge on the bare "
                             "initial clustering (legacy behaviour, for A/B comparison)")
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

    # ── Throwaway in-memory session (no live server, demo_database.db untouched) ─
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    db.add(Dataset(id=DATASET_ID, name="generalization", description=""))
    # Start ACTIVE: the engine refuses turns on a converged/closed session, and we
    # want a real oracle conversation to drive it to convergence.
    session = ChatSession(id=SESSION_ID, dataset_id=DATASET_ID,
                          embedding_model=EMBEDDING_MODEL, status="active")
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
    backend = _last_backend(logger._clustering_log_path)
    print(f"\nInitial clustering: {len(base_points)} points → k={args.k} "
          f"(backend={backend}, seed={KMEANS_RANDOM_STATE}, fit silhouette={sil:.4f}).")

    # ── Drive the throwaway session to convergence with an LLM oracle ─────────
    converged_turn = 0
    if args.no_conversation:
        print("\n--no-conversation: converging on the bare initial clustering "
              "(legacy behaviour).")
        session.status = "converged"
        db.commit()
    else:
        persona = load_persona(args.persona)
        print(f"\nDriving a throwaway conversation with persona "
              f"'{persona.name}' (max {args.max_turns} turns)…")
        if DRY_RUN:
            print("  ⚠ HARNESS_DRY_RUN=true — the oracle's replies are MOCKED; the "
                  "conversation is not meaningful. Run without dry-run for a real session.")
        convo = _run_oracle_conversation(db, session, persona, args.max_turns)
        # The converged snapshot is the LATEST turn that actually wrote soft
        # assignments — NOT necessarily the last conversation turn. A turn with no
        # operations (e.g. the oracle just acknowledging) advances the turn number
        # but writes no new snapshot, so its turn has no rows to freeze from.
        converged_turn = (
            db.query(func.max(SoftAssignment.turn_number)).scalar() or 0
        )
        # Mark converged so the session eval + reporting read it as a finished run.
        session.status = "converged"
        db.commit()
        print(f"  conversation done: {convo['n_turns']} turns, "
              f"terminated_by={convo['terminated_by']}, "
              f"oracle_cost=${convo['oracle_cost_usd']:.4f}, "
              f"errors={len(convo['errors'])}  (converged snapshot at turn {converged_turn})")
        for e in convo["errors"][:5]:
            print(f"    · {e}")

    # ── Session eval (A1–B4) on the converged state ──────────────────────────
    ev = _session_eval(SESSION_ID, db)
    if ev is not None:
        print("\nSESSION EVAL (the converged throwaway session)")
        print(f"    k_final: {ev.k_final}")
        print(f"    A2 turns to convergence: {ev.A2.turns}  "
              f"(weighted {ev.A2.weighted_turns}, termination={ev.A2.termination})")
        print(f"    A3 mean cognitive load:  {ev.A3.mean_cognitive_load}")
        print(f"    B1 overall: {ev.B1.overall_score:.3f}   "
              f"B2 coherence mean: {ev.B2.coherence_mean}   "
              f"B3 compliance: {ev.B3.compliance_score:.3f}   "
              f"B4 contradiction: {ev.B4.contradiction_score:.3f}")

    print("\n" + "─" * 70)
    print("GENERALIZATION ASSESSMENT — does this converged state hold as new data arrives?")
    print("─" * 70)

    # Freeze the convergence GMM parameters ONCE, from the CONVERGED snapshot
    # (the latest snapshot turn the conversation produced, not the bare initial
    # clustering at turn 0). Embeddings are read from the whole dataset (a single
    # in-memory dataset) to avoid a 1000+-element SQL IN clause.
    snap0 = _snapshot_at(db, converged_turn)
    emb0 = {
        dp.id: dp.embedding
        for dp in db.query(DataPoint).filter(DataPoint.dataset_id == DATASET_ID).all()
        if dp.embedding is not None and dp.id in snap0
    }
    centroid_ids, centroids, diag_vars, log_weights = gmm_params_from_snapshot(emb0, snap0)

    # A4 calibration: per-cluster Mahalanobis-d² reference built ONCE from the
    # converged state. calibrate_distance_reference recovers diag_var internally
    # from the base hard-label assignments — consistent with gmm_params_from_snapshot.
    calib_pids = [pid for pid in snap0 if pid in emb0]
    calib_emb = np.asarray([emb0[pid] for pid in calib_pids], dtype=np.float64)
    calib_labels = [max(snap0[pid], key=snap0[pid].get) for pid in calib_pids]
    ood_calibration = calibrate_distance_reference(
        calib_emb, calib_labels, centroid_ids, centroids
    )

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
    drift: list[tuple[int, float, int, float]] = []  # (turn, mean_sil, n, ood_rate)
    a4_batches: list[tuple[list[str], dict]] = []
    last_turn = converged_turn
    for b, idx in enumerate(batches):
        if len(idx) == 0:
            continue
        batch_points = [
            DataPoint(id=f"new-{int(i)}", dataset_id=DATASET_ID,
                      text=new_texts[int(i)],
                      embedding=new_emb[int(i)].tolist())
            for i in idx
        ]
        last_turn, _ = ingest_points(
            SESSION_ID, batch_points, centroid_ids, centroids, diag_vars, log_weights, db
        )
        db.commit()
        snap = _snapshot_at(db, last_turn)
        sil_b = _hard_silhouette(db, snap)
        mean_b = float(np.mean(list(sil_b.values()))) if sil_b else 0.0

        # A4 on the batch: score new embeddings against the frozen calibration.
        batch_emb = np.asarray([new_emb[int(i)] for i in idx], dtype=np.float64)
        batch_assigned = assign_gmm_posterior(
            batch_emb, centroid_ids, centroids, diag_vars, log_weights
        )
        batch_scores = assignment_ood(
            batch_emb, batch_assigned, centroid_ids, centroids, ood_calibration
        )
        a4_batches.append((batch_assigned, batch_scores))

        drift.append((last_turn, mean_b, len(snap), batch_scores["ood_rate"]))
        print(f"  ingested batch {b + 1}/{len(batches)}: +{len(batch_points)} points "
              f"→ turn {last_turn}, {len(snap)} total, mean silhouette {mean_b:.4f}, "
              f"OOD {batch_scores['ood_rate'] * 100:.1f}%")

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
    _report_a4(a4_batches)
    print()
    if b2_error:
        print("B2 — cluster coherence: UNAVAILABLE (LLM judge failed)")
        print(f"    {b2_error}")
        print("    A1 + A4 above are deterministic and complete; re-run with a working")
        print("    LLM provider to get the B2 paired Δ.")
    else:
        _report_b2(coh0, coh1)
    if len(drift) > 1:
        print("\nDrift curve (per batch — mean silhouette + A4 OOD rate):")
        print(f"    t0  : sil={float(np.mean(list(sil0.values()))):.4f}  "
              f"({len(snap0)} pts)")
        for turn, mean_b, n, ood in drift:
            print(f"    t{turn}  : sil={mean_b:.4f}  ood={ood * 100:5.1f}%  ({n} pts)")
    print("=" * 70)
    print("Generalization holds when A1 stays put (paired Δ CI spans 0 or is "
          "positive), A4's OOD rate stays near its baseline, and B2 does not "
          "drop. A decline on any of these means new data is breaking the "
          "converged structure.")
    print("=" * 70)

    # ── Close the throwaway session ──────────────────────────────────────────
    # The in-memory DB is discarded on close — the session, its turns, clusters
    # and the ingested points vanish with it. Nothing was ever written to
    # demo_database.db.
    session.status = "closed"
    db.commit()
    print(f"\nThrowaway session '{SESSION_ID}' closed and discarded "
          f"(in-memory DB — demo_database.db was never touched).")
    db.close()


if __name__ == "__main__":
    main()
