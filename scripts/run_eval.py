"""Evaluation runner — v1.

Drives scripted-oracle scenarios through the live API, then collects every metric
the quality spec (`docs/quality_specs.md`) calls implementable today: A1, A2, A3,
B1, B2, B3, B4. Out-of-band end-of-session — does **not** modify the live turn
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
import sys
import time
import urllib.error
import urllib.request

from src.eval_report import write_summary

BASE = "http://localhost:8000"



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
        per_turn_ops: list[list[str]] = []
        last_action = None

        for ot in oracle_turns:
            s, resp = _call("POST", "/turns", {
                "session_id": sid,
                "raw_text": ot["raw_text"],
                "feedback_type": ot.get("feedback_type", "global"),
                "target_cluster_ids": ot.get("target_cluster_ids", []) or [],
                "target_point_ids": ot.get("target_point_ids", []) or [],
            })
            if s >= 400:
                record["errors"].append(f"turn[{len(per_turn_ops)+1}] http={s} body={resp}")
                per_turn_ops.append([])
                continue

            so = resp.get("system_output", {})
            last_action = so.get("action")
            ops = so.get("state_snapshot", {}).get("operations", []) or []
            per_turn_ops.append([o.get("type") for o in ops])
            if last_action == "stop":
                break

        # --- end-of-session metrics via eval endpoint ---------------------
        s, ev = _call("POST", f"/sessions/{sid}/eval")
        if s != 200:
            record["errors"].append(f"eval http={s} body={ev}")
            return record

        record["k_final"] = ev["k_final"]
        record["A1"] = ev["A1"]
        record["A2"] = {**ev["A2"], "last_action": last_action, "ops_per_turn": per_turn_ops}
        record["A3"] = ev["A3"]
        record["B1"] = ev["B1"]
        record["B2"] = ev["B2"]
        record["B3"] = ev["B3"]
        record["B4"] = ev["B4"]

    finally:
        record["wall_time_s"] = round(time.time() - t_start, 1)

    return record


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------
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
            overall = (rec.get("B1") or {}).get("overall_score")
            contr = (rec.get("B4") or {}).get("contradiction_score")
            err = rec.get("errors") or []
            print(
                f"  done: termination={term}  overall={overall}  "
                f"contradiction={contr}  errors={len(err)}"
            )

    write_summary(records, out_dir)
    print(f"\nResults  → {results_path}")
    print(f"Summary  → {os.path.join(out_dir, 'summary.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())