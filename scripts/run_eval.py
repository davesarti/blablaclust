"""Evaluation runner — v1.

Drives scripted-oracle scenarios through the live API, then collects every metric
the quality spec (`docs/quality_specs.md`) calls implementable today: A1, A2, B1,
B2, B3, and B4. Out-of-band end-of-session — does **not** modify the live turn
path.

Usage:
    PYTHONPATH=. python scripts/run_eval.py --scenarios scenarios/*.json --out reports/

If --out is omitted, a timestamped directory under reports/ is created.

Each scenario produces one JSONL line in results.jsonl and contributes to a
human-readable summary.md emitted in the same directory.

The runner needs the live API up (PYTHONPATH=. python scripts/serve_ui.py).
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"

# Feedback-type weights for A2's weighted turn count.
# A "global" reframe is heavier than a "point" nudge.  Frozen for v1; revisit
# only if a real scenario surfaces a weight that doesn't match intuition.
FEEDBACK_TYPE_WEIGHTS = {
    "global": 2.0,
    "cluster": 1.0,
    "point": 0.5,
    "instructional": 0.0,
}

# How many points per cluster to validate via B4 at end-of-session.
B4_POINTS_PER_CLUSTER = 3


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _call(method: str, path: str, body=None, timeout: int = 180):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        body_str = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(body_str)
        except Exception:
            return e.code, body_str


# ---------------------------------------------------------------------------
# Per-scenario run
# ---------------------------------------------------------------------------
def run_scenario(scenario: dict) -> dict:
    """Drive one scenario end-to-end. Returns a result dict (one JSONL row)."""
    name = scenario.get("name", "<unnamed>")
    dataset = scenario["dataset"]
    k_initial = int(scenario.get("k_initial", 5))
    oracle_turns = scenario["oracle_turns"]

    record: dict = {
        "scenario": name,
        "description": scenario.get("description", ""),
        "session_id": None,
        "k_initial": k_initial,
        "k_final": None,
        "errors": [],
    }
    t_start = time.time()

    # --- create session + initial clustering -------------------------------
    s, sess = _call("POST", "/sessions",
                    {"dataset_name": dataset, "name": f"eval/{name}"})
    if s != 200:
        record["errors"].append(f"create_session http={s} body={sess}")
        return record
    sid = sess["id"]
    record["session_id"] = sid

    try:
        s, _ = _call("POST", f"/clusters/{sid}",
                     {"k": k_initial, "generate_names": True})
        if s != 200:
            record["errors"].append(f"initial_clustering http={s}")
            return record

        # --- drive scripted oracle turns ----------------------------------
        cog_loads: list[int] = []
        contradictions = 0
        per_turn_ops: list[list[str]] = []
        term_reason = None
        last_action = None
        last_feedback_types: list[str] = []

        for ot in oracle_turns:
            s, resp = _call("POST", "/turns", {
                "session_id": sid,
                "raw_text": ot["raw_text"],
                "feedback_type": ot.get("feedback_type", "global"),
                "target_cluster_ids": ot.get("target_cluster_ids", []) or [],
                "target_point_ids": ot.get("target_point_ids", []) or [],
            })
            last_feedback_types.append(ot.get("feedback_type", "global"))
            if s >= 400:
                record["errors"].append(f"turn[{len(per_turn_ops)+1}] http={s} body={resp}")
                # Don't abort — partial run still has useful signal.
                per_turn_ops.append([])
                continue

            so = resp.get("system_output", {})
            last_action = so.get("action")
            cog_loads.append(int(so.get("cognitive_load_score", 0)))
            if so.get("contradiction_detected"):
                contradictions += 1
            ops = so.get("state_snapshot", {}).get("operations", []) or []
            per_turn_ops.append([o.get("type") for o in ops])
            term_reason = (so.get("state_snapshot") or {}).get("reason")
            # If the Planner emitted stop, stop driving the scripted oracle.
            if last_action == "stop":
                break

        # --- end-of-session metrics ---------------------------------------
        # A2 termination bucket — engine emits cognitive_overload /
        # max_turns_reached; nothing emits "converged" today (no healthy-stop
        # rule in f_next_best_step). Treat run-to-end-without-overload as
        # "converged" from the runner's perspective.
        if term_reason in ("cognitive_overload", "max_turns_reached"):
            termination = term_reason
        else:
            termination = "converged"

        weighted_turns = sum(
            FEEDBACK_TYPE_WEIGHTS.get(ft, 1.0) for ft in last_feedback_types
        )
        record["A2"] = {
            "turns": len(per_turn_ops),
            "weighted_turns": round(weighted_turns, 2),
            "termination": termination,
            "last_action": last_action,
            "ops_per_turn": per_turn_ops,
        }
        record["B2"] = {
            "cognitive_load_by_turn": cog_loads,
            "mean_cognitive_load": (
                round(statistics.mean(cog_loads), 2) if cog_loads else None
            ),
        }
        record["B3"] = {"contradictions_detected": contradictions}

        # A1 — silhouette trend from the clustering-runs log.
        record["A1"] = _read_silhouette_trend(sid)

        # Final state + cluster snapshot for B1/B4.
        s, state = _call("GET", f"/sessions/{sid}/state")
        if s != 200:
            record["errors"].append(f"GET state http={s}")
            return record
        record["k_final"] = len(state["clusters"])

        # B1 — call the judge.
        record["B1"] = _call_judge(state)

        # B4 — sample points per active cluster, ask the judge each.
        record["B4"] = _validate_sampled_points(state)

    finally:
        # Always tear down the eval session so the DB doesn't accumulate.
        d_status, _ = _call("DELETE", f"/sessions/{sid}/delete")
        record["cleanup_http"] = d_status
        record["wall_time_s"] = round(time.time() - t_start, 1)

    return record


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _read_silhouette_trend(session_id: str) -> dict:
    """Pull silhouette values for this session from logs/clustering_runs.jsonl.

    The clustering-runs log records one line per initial_clustering call with a
    `session_id` field and a `silhouette` field — across a session those points
    form A1's trend.
    """
    path = "logs/clustering_runs.jsonl"
    if not os.path.exists(path):
        return {"silhouette_initial": None, "silhouette_final": None, "trend": []}
    trend: list[float | None] = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("session_id") != session_id:
                continue
            trend.append(rec.get("silhouette"))
    initial = trend[0] if trend else None
    final = trend[-1] if trend else None
    return {
        "silhouette_initial": initial,
        "silhouette_final": final,
        "trend": trend,
    }


def _call_judge(state: dict) -> dict:
    """Invoke f_eval (B1) on the final state. Returns {coherence_score, notes}
    or {error: ...} on failure — failure here must not poison the batch."""
    try:
        # Reconstruct ChatSessionState pydantic from the GET /state response.
        from src.schemas import ChatSessionState
        from src.engine.f_eval import f_eval

        st = ChatSessionState.model_validate(state)
        total_points = sum(c.size for c in st.clusters)
        return f_eval(st, total_points=total_points)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _validate_sampled_points(state: dict) -> dict:
    """Sample N points per active cluster and run f_validate_point on each.

    v1 sampling: top-N highest-uncertainty points per cluster (drawn from
    /clusters/{id}/points, ranked by `probability` ascending — lowest probability
    on the assigned cluster proxies highest uncertainty). This makes B4 a stress
    test of the boundaries, where calibration validation actually pays off.
    """
    try:
        from src.schemas import ChatSessionState, Cluster
        from src.engine.f_validate_point import f_validate_point
        st = ChatSessionState.model_validate(state)
    except Exception as exc:
        return {"error": f"setup: {type(exc).__name__}: {exc}", "endorsements": []}

    endorsements: list[dict] = []
    for cluster in st.clusters:
        s, pts = _call("GET", f"/clusters/{cluster.id}/points")
        if s != 200 or not isinstance(pts, dict):
            continue
        candidates = sorted(
            pts.get("points") or [],
            key=lambda p: p.get("probability", 1.0),
        )[:B4_POINTS_PER_CLUSTER]

        for p in candidates:
            text = ((p.get("data") or {}).get("text", "")
                    or (p.get("data") or {}).get("title", ""))
            if not text:
                continue
            try:
                result = f_validate_point(
                    state=st,
                    cluster=cluster,
                    point_id=p["id"],
                    point_text=text,
                )
                endorsements.append({
                    "cluster_id": cluster.id,
                    "point_id": p["id"],
                    "probability": p.get("probability"),
                    **result,
                })
            except Exception as exc:
                endorsements.append({
                    "cluster_id": cluster.id,
                    "point_id": p["id"],
                    "error": f"{type(exc).__name__}: {exc}",
                })

    valid = [e for e in endorsements if "endorsed" in e]
    return {
        "n_sampled": len(endorsements),
        "n_valid": len(valid),
        "endorsement_rate": (
            round(sum(1 for e in valid if e["endorsed"]) / len(valid), 3)
            if valid else None
        ),
        "mean_confidence": (
            round(statistics.mean(e["confidence"] for e in valid), 3)
            if valid else None
        ),
        "endorsements": endorsements,
    }


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------
def write_summary(records: list[dict], out_dir: str):
    """Write a human-readable summary.md alongside the results.jsonl."""
    lines = ["# Evaluation summary\n"]
    lines.append(f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append(f"\n**Scenarios run:** {len(records)}\n")

    lines.append("""
> **Metric cheat-sheet**
> | Metric | What it measures | Good looks like |
> |--------|-----------------|-----------------|
> | **A1 silhouette** | Geometric separation of clusters (−1 to 1). Higher = tighter, more distinct groups. | Stable or rising across turns. |
> | **A2 turns / weighted turns** | How much dialogue it took to converge. Weighted turns penalise heavy global feedback more than fine-grained point edits. | Fewer turns, ends in `converged`. |
> | **B1 coherence** | LLM judge rates 0–1 how focused and internally consistent the final clusters are. | ≥ 0.7 |
> | **B2 cognitive load** | Per-turn complexity score (1–5) the engine assigns to the oracle's feedback. | Stays low (1–2). |
> | **B3 contradictions** | How many times the oracle's new feedback conflicted with something said earlier. | 0 in a healthy session. |
> | **B4 endorsement rate** | Fraction of sampled boundary points the judge says actually belong in their cluster. Sampled at highest uncertainty, so this is a stress test. | > 0.6 despite the hard sampling. |
""")

    # Aggregate
    terms = [r.get("A2", {}).get("termination") for r in records]
    coherences = [r.get("B1", {}).get("coherence_score")
                  for r in records
                  if isinstance(r.get("B1", {}).get("coherence_score"), (int, float))]
    endorsement_rates = [r.get("B4", {}).get("endorsement_rate")
                         for r in records
                         if isinstance(r.get("B4", {}).get("endorsement_rate"), (int, float))]
    silhouette_finals = [r.get("A1", {}).get("silhouette_final")
                         for r in records
                         if isinstance(r.get("A1", {}).get("silhouette_final"), (int, float))]

    def _agg(label, values):
        if not values:
            return f"- **{label}**: (no data)\n"
        return (
            f"- **{label}**: mean={statistics.mean(values):.3f} "
            f"median={statistics.median(values):.3f} n={len(values)}\n"
        )

    lines.append("\n## Aggregates across all scenarios\n")
    lines.append(_agg("B1 coherence score  _(LLM judge, 0–1)_", coherences))
    lines.append(_agg("B4 endorsement rate  _(boundary-point stress test, 0–1)_", endorsement_rates))
    lines.append(_agg("A1 silhouette (final)  _(cluster separation, higher = better)_", silhouette_finals))
    term_counts = ", ".join(f"`{t}` × {terms.count(t)}" for t in sorted(set(terms)))
    lines.append(f"- **A2 termination breakdown**: {term_counts}\n")

    lines.append("\n---\n")
    lines.append("\n## Per-scenario detail\n")
    for r in records:
        lines.append(f"\n### {r['scenario']}\n")
        if r.get("description"):
            lines.append(f"_{r['description']}_\n\n")
        a1 = r.get("A1") or {}
        a2 = r.get("A2") or {}
        b1 = r.get("B1") or {}
        b2 = r.get("B2") or {}
        b3 = r.get("B3") or {}
        b4 = r.get("B4") or {}

        # A1
        sil_i = a1.get("silhouette_initial")
        sil_f = a1.get("silhouette_final")
        sil_arrow = f"{sil_i:.3f} → {sil_f:.3f}" if sil_i is not None and sil_f is not None else "(no data)"
        sil_comment = ""
        if sil_i is not None and sil_f is not None:
            delta = sil_f - sil_i
            sil_comment = f" ({'improved' if delta > 0.005 else 'dropped' if delta < -0.005 else 'stable'})"
        lines.append(f"**A1 — Cluster quality (silhouette):** {sil_arrow}{sil_comment}\n")
        lines.append(f"> Measures whether the final clusters are geometrically tight and well-separated. "
                     f"A drop can happen when adding a cluster splits a previously cohesive group.\n\n")

        # A2
        turns = a2.get("turns", "?")
        wt = a2.get("weighted_turns", "?")
        term = a2.get("termination", "?")
        ops_summary = "; ".join(
            f"turn {i+1}: {', '.join(ops) or 'no ops'}"
            for i, ops in enumerate(a2.get("ops_per_turn") or [])
        )
        lines.append(f"**A2 — Dialogue efficiency:** {turns} turns (weighted {wt}), ended as `{term}`\n")
        lines.append(f"> Operations per turn: {ops_summary}\n\n")

        # B1
        coh = b1.get("coherence_score")
        coh_str = f"{coh:.2f}" if isinstance(coh, float) else str(coh)
        notes = b1.get("notes") or b1.get("error") or ""
        lines.append(f"**B1 — Coherence (LLM judge):** {coh_str}/1.0\n")
        if notes:
            lines.append(f"> {notes}\n\n")

        # B2
        cog_mean = b2.get("mean_cognitive_load")
        cog_turns = b2.get("cognitive_load_by_turn") or []
        cog_str = f"{cog_mean}/5" if cog_mean is not None else "(no data)"
        lines.append(f"**B2 — Cognitive load:** mean {cog_str} per turn: {cog_turns}\n")
        lines.append(f"> Score 1–5 the engine assigns each oracle turn. "
                     f"High load (≥4) can trigger early termination.\n\n")

        # B3
        contra = b3.get("contradictions_detected", 0)
        lines.append(f"**B3 — Contradictions detected:** {contra}\n")
        lines.append(f"> Counts turns where the oracle's request conflicted with earlier feedback.\n\n")

        # B4
        er = b4.get("endorsement_rate")
        mc = b4.get("mean_confidence")
        nv = b4.get("n_valid")
        er_str = f"{er:.1%}" if er is not None else "(no data)"
        mc_str = f"{mc:.2f}" if mc is not None else "—"
        lines.append(f"**B4 — Point-level validation:** {er_str} endorsed, mean confidence {mc_str} (n={nv})\n")
        lines.append(f"> An independent judge checks whether the most uncertain boundary points "
                     f"actually belong in their assigned cluster. Low rates here signal soft-assignment "
                     f"boundary issues.\n\n")

        # k + timing
        lines.append(f"**Clusters:** {r.get('k_initial')} → {r.get('k_final')}  |  "
                     f"**Wall time:** {r.get('wall_time_s')}s\n")

        if r.get("errors"):
            lines.append(f"\n> **Errors:** {r['errors']}\n")

        lines.append("\n")

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", required=True,
                    help="Scenario JSON files (globs allowed).")
    ap.add_argument("--out", default=None,
                    help="Output directory. Defaults to reports/<timestamp>/.")
    args = ap.parse_args()

    # Resolve scenario files (allow globs).
    scenario_paths: list[str] = []
    for pat in args.scenarios:
        scenario_paths.extend(sorted(glob.glob(pat)))
    if not scenario_paths:
        print("ERROR: no scenarios matched.", file=sys.stderr)
        return 1

    out_dir = args.out or os.path.join(
        "reports", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")

    records: list[dict] = []
    with open(results_path, "w") as f:
        for path in scenario_paths:
            print(f"=== running scenario: {path} ===")
            with open(path) as fh:
                scenario = json.load(fh)
            rec = run_scenario(scenario)
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            term = (rec.get("A2") or {}).get("termination")
            coh = (rec.get("B1") or {}).get("coherence_score")
            err = rec.get("errors") or []
            print(f"  done: termination={term}  coherence={coh}  errors={len(err)}")

    write_summary(records, out_dir)
    print(f"\nResults  → {results_path}")
    print(f"Summary  → {os.path.join(out_dir, 'summary.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())