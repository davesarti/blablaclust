"""LLM-as-oracle evaluation runner.

Peer of scripts/run_scenario_eval.py — drives persona-defined LLM oracles through the
live API instead of scripted scenario turns. Each persona produces one
JSONL row in results.jsonl and contributes to a human-readable summary.md
emitted in the same directory.

The runner needs the live API up (PYTHONPATH=. python scripts/serve_ui.py).

Usage:
    PYTHONPATH=. python scripts/run_persona_eval.py \
        --personas personas/*.json --out reports/ --max-turns 12
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
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.llm_oracle import LLMOracle, OracleResponseError  # noqa: E402
from src.eval.oracle_view import build_oracle_view  # noqa: E402
from src.eval.persona import Persona, load_persona  # noqa: E402

BASE = "http://localhost:8000"

_DEFAULT_MAX_TURNS = 12
_INITIAL_SYSTEM_PROMPT = (
    "Initial clustering is ready. Tell me how you'd like it changed, or "
    "let me know if it already looks right."
)


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
# Per-persona run
# ---------------------------------------------------------------------------
def run_persona(persona: Persona, max_turns: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "persona": persona.name,
        "description": persona.description or "",
        "session_id": None,
        "oracle_kind": "persona",
        "model": None,
        "k_initial": persona.k_initial,
        "k_final": None,
        "n_turns": 0,
        "terminated_by": None,
        "oracle_total_tokens": {"input": 0, "output": 0},
        "oracle_total_cost_usd": 0.0,
        "errors": [],
    }
    t_start = time.time()

    # --- resolve dataset name → id, then create session + clustering -------
    s, ds_list = _call("GET", "/datasets")
    if s != 200:
        record["errors"].append(f"list_datasets http={s} body={ds_list}")
        record["wall_time_s"] = round(time.time() - t_start, 1)
        return record
    ds_match = next((d for d in ds_list if d.get("dataset_name") == persona.dataset), None)
    if ds_match is None:
        record["errors"].append(f"dataset '{persona.dataset}' not found on server")
        record["wall_time_s"] = round(time.time() - t_start, 1)
        return record
    s, sess = _call("POST", "/sessions", {
        "dataset_id": ds_match["dataset_id"],
        "name": f"persona/{persona.name}",
        "oracle_kind": "persona",
        "persona_snapshot": persona.model_dump(),
    })
    if s != 200:
        record["errors"].append(f"create_session http={s} body={sess}")
        record["wall_time_s"] = round(time.time() - t_start, 1)
        return record
    sid = sess["id"]
    record["session_id"] = sid

    try:
        s, _ = _call("POST", f"/clusters/{sid}",
                     {"k": persona.k_initial, "generate_names": True})
        if s != 200:
            record["errors"].append(f"initial_clustering http={s}")
            return record

        oracle = LLMOracle(persona=persona, session_id=sid, max_turns=max_turns)
        record["model"] = oracle.model
        system_display = _INITIAL_SYSTEM_PROMPT

        for turn_idx in range(1, max_turns + 1):
            # 1. Fetch the view the oracle will see this turn.
            view = _fetch_view(sid)

            # 2. Oracle composes its next message.
            try:
                oracle.observe_system(system_display)
                oracle_turn = oracle.next_turn(view, system_display, turn_idx)
            except OracleResponseError as exc:
                record["errors"].append(f"oracle_parse[{turn_idx}] {exc}")
                record["terminated_by"] = "oracle_parse_error"
                break

            # 3. Post the turn to the live API.
            s, resp = _call("POST", "/turns", oracle_turn.body)
            if s >= 500:
                # Server error — nothing the oracle can do about it. Stop.
                record["errors"].append(f"turn[{turn_idx}] http={s} body={resp}")
                record["terminated_by"] = "api_error"
                break
            if s >= 400:
                # The engine rejected the oracle's request as invalid (e.g. it
                # named a cluster that an earlier split/merge had dissolved). A
                # human would just be told "that no longer exists" and pick
                # another — so feed the error back and let the oracle correct
                # course next turn instead of ending the whole session here.
                detail = resp.get("detail") if isinstance(resp, dict) else resp
                record["errors"].append(f"turn[{turn_idx}] http={s} body={resp}")
                system_display = (
                    f"That request couldn't be applied ({detail}). The clustering is "
                    f"unchanged — pick a cluster currently shown in the panel and try again."
                )
                continue
            record["n_turns"] = turn_idx

            # 4. Capture the system reply for the next prompt.
            so = (resp or {}).get("system_output") or {}
            display = so.get("display") or {}
            system_display = (
                display.get("content")
                if isinstance(display, dict)
                else str(display)
            ) or ""
            last_action = so.get("action")

            if last_action == "stop":
                record["terminated_by"] = "system_stop"
                break
            if oracle_turn.satisfied:
                record["terminated_by"] = "oracle_satisfied"
                break
        else:
            record["terminated_by"] = "max_turns"

        usage, cost = oracle.totals
        record["oracle_total_tokens"] = {
            "input":  usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
        }
        record["oracle_total_cost_usd"] = round(cost, 6)

        # --- end-of-session metrics via eval endpoint --------------------
        s, ev = _call("POST", f"/sessions/{sid}/eval")
        if s != 200:
            record["errors"].append(f"eval http={s} body={ev}")
        else:
            record["k_final"] = ev.get("k_final")
            for key in ("A1", "A2", "A3", "B1", "B2", "B3", "B4"):
                if key in ev:
                    record[key] = ev[key]

    finally:
        record["wall_time_s"] = round(time.time() - t_start, 1)

    return record


def _fetch_view(session_id: str):
    """GET /sessions/{sid}/state + /clusters/{cid}/points per active cluster."""
    s, state = _call("GET", f"/sessions/{session_id}/state")
    if s != 200 or not isinstance(state, dict):
        # Empty view — oracle will see "(no active clusters)" and can still reply.
        return build_oracle_view({"clusters": []}, {}, n_examples=3)

    cluster_points: Dict[str, List[Dict[str, Any]]] = {}
    for cluster in state.get("clusters") or []:
        cid = cluster.get("id")
        if not cid:
            continue
        s, points = _call("GET", f"/clusters/{cid}/points?limit=3")
        if s != 200 or not isinstance(points, dict):
            continue
        cluster_points[cid] = points.get("points") or []
    return build_oracle_view(state, cluster_points, n_examples=3)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _write_summary(records: List[Dict[str, Any]], out_dir: str) -> None:
    """A persona-aware summary.md. Mirrors run_scenario_eval's spirit, narrower scope."""
    lines: List[str] = []
    lines.append("# Persona evaluation summary\n")
    lines.append(f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append("")
    lines.append("| Persona | Termination | Turns | k_initial→k_final | Oracle tokens (in/out) | Cost USD | Errors |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in records:
        tokens = r.get("oracle_total_tokens") or {}
        lines.append(
            f"| {r.get('persona','?')} "
            f"| {r.get('terminated_by','-')} "
            f"| {r.get('n_turns','-')} "
            f"| {r.get('k_initial','-')}→{r.get('k_final','-')} "
            f"| {tokens.get('input',0)} / {tokens.get('output',0)} "
            f"| {r.get('oracle_total_cost_usd',0.0):.4f} "
            f"| {len(r.get('errors') or [])} |"
        )
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", required=True,
                    help="Persona JSON files (globs allowed).")
    ap.add_argument("--out", default=None,
                    help="Output directory. Defaults to reports/<timestamp>-persona/.")
    ap.add_argument("--max-turns", type=int, default=_DEFAULT_MAX_TURNS,
                    help=f"Hard cap on oracle turns per persona (default {_DEFAULT_MAX_TURNS}).")
    ap.add_argument("--dataset", default=None,
                    help="Override the dataset name for all personas (e.g. imdb_train).")
    args = ap.parse_args()

    persona_paths: List[str] = []
    for pat in args.personas:
        persona_paths.extend(sorted(glob.glob(pat)))
    if not persona_paths:
        print("ERROR: no personas matched.", file=sys.stderr)
        return 1

    out_dir = args.out or os.path.join(
        "reports",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-persona",
    )
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")

    records: List[Dict[str, Any]] = []
    with open(results_path, "w") as f:
        for path in persona_paths:
            print(f"=== running persona: {path} ===")
            persona = load_persona(path)
            if args.dataset:
                persona.dataset = args.dataset
            rec = run_persona(persona, max_turns=args.max_turns)
            records.append(rec)
            f.write(json.dumps(rec, indent=2) + "\n")
            f.flush()
            print(
                f"  done: termination={rec.get('terminated_by')}  "
                f"turns={rec.get('n_turns')}  "
                f"cost=${rec.get('oracle_total_cost_usd',0):.4f}  "
                f"errors={len(rec.get('errors') or [])}"
            )

    _write_summary(records, out_dir)
    print(f"\nResults  → {results_path}")
    print(f"Summary  → {os.path.join(out_dir, 'summary.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
