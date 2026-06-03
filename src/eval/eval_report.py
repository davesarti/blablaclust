"""Shared eval report writer used by both the API endpoint and run_scenario_eval.py."""

import datetime
import json
import os
import statistics

import numpy as np

# Percentile-bootstrap resamples for the 95% CIs on the cross-scenario means.
# Matches scripts/run_generalization_stability_eval.py so CI methodology is
# consistent across the project's eval reports.
_BOOTSTRAP_ITERS = 10000


def write_report(record: dict, out_dir: str) -> None:
    """Write one eval record to results.jsonl + summary.md in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
        f.write(json.dumps(record, indent=2) + "\n")
    write_summary([record], out_dir)


def _num(values):
    return [v for v in values if isinstance(v, (int, float))]


def _bootstrap_ci(values, iters: int = _BOOTSTRAP_ITERS, seed: int = 0):
    """Percentile bootstrap 95% CI for the mean of *values*. Returns (lo, hi).

    Returns (nan, nan) when there are fewer than 2 values — a CI over a single
    scenario is meaningless, so callers print an explicit "n=1" note instead.
    """
    arr = np.asarray([v for v in values if isinstance(v, (int, float))], dtype=float)
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(iters, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def _agg(label, values):
    if not values:
        return f"- **{label}**: (no data)\n"
    mean = statistics.mean(values)
    median = statistics.median(values)
    n = len(values)
    if n >= 2:
        lo, hi = _bootstrap_ci(values)
        tail = f" — **95% CI [{lo:.3f}, {hi:.3f}]** (median={median:.3f}, n={n})"
    else:
        tail = f" (median={median:.3f}, n=1 — CI needs ≥2 scenarios)"
    return f"- **{label}**: mean={mean:.3f}{tail}\n"


def write_summary(records: list[dict], out_dir: str) -> None:
    """Write a human-readable summary.md alongside results.jsonl."""
    lines = ["# Evaluation summary\n"]
    lines.append(f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append(f"\n**Scenarios run:** {len(records)}\n")

    lines.append("""
> **Metric cheat-sheet**
> | Metric | What it measures | Good looks like |
> |--------|-----------------|-----------------|
> | **A1 silhouette** | Geometric separation of clusters (−1 to 1). Higher = tighter, more distinct groups. | Stable or rising across turns. |
> | **A2 turns / weighted turns** | How much dialogue it took to converge. Weighted turns penalise heavy global feedback more than fine-grained point edits. | Fewer turns, ends in `converged`. |
> | **A3 cognitive load** | Per-turn deterministic LLM-side load (1–5) from turns / pre-trim tokens / cluster count. Session halts at 5; the `driver` says which signal saturated. | Stays low (1–2). |
> | **B1 overall** | Synthesis judge over B2, B3, B4 — gives one fair verdict on the session. | ≥ 0.7 |
> | **B2 coherence** | Per-cluster internal focus (LLM judge). Reports mean and the weakest cluster. | Mean ≥ 0.7, no single cluster below 0.5. |
> | **B3 compliance** | Fidelity of system operations to oracle requests (LLM judge). | ≥ 0.7 |
> | **B4 contradiction** | How hard the oracle was to understand: contradictions, drift, vague targets. Higher = harder. Feeds B1 as forgiveness context. | Low (≤ 0.3). High values excuse low B2/B3 in B1. |
""")

    overall_scores  = _num(r.get("B1", {}).get("overall_score") for r in records)
    coherence_means = _num(r.get("B2", {}).get("coherence_mean") for r in records)
    coherence_mins  = _num(r.get("B2", {}).get("coherence_min") for r in records)
    compliance      = _num(r.get("B3", {}).get("compliance_score") for r in records)
    contradiction   = _num(r.get("B4", {}).get("contradiction_score") for r in records)
    sil_initials    = _num(r.get("A1", {}).get("silhouette_initial") for r in records)
    sil_finals      = _num(r.get("A1", {}).get("silhouette_final") for r in records)
    cog_means       = _num(r.get("A3", {}).get("mean_cognitive_load") for r in records)
    turns_vals      = _num(r.get("A2", {}).get("turns") for r in records)
    weighted_vals   = _num(r.get("A2", {}).get("weighted_turns") for r in records)
    terms           = [r.get("A2", {}).get("termination") for r in records]

    lines.append("\n## Aggregates across all scenarios\n")
    lines.append(
        "_Means are over scenarios; **95% CI is a percentile bootstrap** "
        f"({_BOOTSTRAP_ITERS:,} resamples) over the per-scenario values. "
        "A CI needs ≥2 scenarios to be meaningful._\n\n"
    )
    lines.append("**Convergence**\n")
    lines.append(_agg("A2 turns to convergence", turns_vals))
    lines.append(_agg("A2 weighted turns  _(global feedback weighted heavier)_", weighted_vals))
    lines.append("\n**Quality (conversational arm)**\n")
    lines.append(_agg("B1 overall  _(synthesis judge, 0–1)_", overall_scores))
    lines.append(_agg("B2 coherence mean  _(0–1)_", coherence_means))
    lines.append(_agg("B2 coherence min  _(weakest cluster, 0–1)_", coherence_mins))
    lines.append(_agg("B3 compliance  _(0–1)_", compliance))
    lines.append(_agg("B4 contradiction  _(0–1, higher = oracle harder to understand)_", contradiction))
    lines.append(_agg("A3 mean cognitive load  _(1–5)_", cog_means))
    lines.append("\n**A1 silhouette (geometry)**\n")
    lines.append(
        "_`initial` is the turn-0 whole-dataset clustering — a clean **no-dialogue "
        "baseline**. `last-logged` is the final entry in the clustering log, which "
        "**mixes whole-dataset fits** (turn 0, semantic re-embed) **with split "
        "sub-cluster fits** (a split re-runs k-means on one cluster's subset and logs "
        "that subset's silhouette); merge / move / rename never re-log. So `last-logged` "
        "is **not** directly comparable to `initial`, and a final−initial Δ is not a "
        "clean effect-of-dialogue measure. For the quality comparison use **B2 "
        "coherence**: the no-dialogue baseline from `run_baseline_eval.py` vs the "
        "conversational B2 above (both measured on the final state, same judge)._\n"
    )
    lines.append(_agg("A1 silhouette (turn-0 / no-dialogue baseline)", sil_initials))
    lines.append(_agg("A1 silhouette (last logged — see caveat)", sil_finals))

    # Stop-driver distribution: only scenarios that terminated via
    # cognitive_overload contribute. We read the last element of each
    # cognitive_load_driver_by_turn; if a record terminated converged we
    # exclude it because no cap was tripped.
    drivers: list[str] = []
    for r in records:
        if (r.get("A2") or {}).get("termination") != "cognitive_overload":
            continue
        driver_series = (r.get("A3") or {}).get("cognitive_load_driver_by_turn") or []
        if driver_series:
            drivers.append(driver_series[-1])
    if drivers:
        driver_counts = ", ".join(
            f"`{d}` × {drivers.count(d)}" for d in sorted(set(drivers))
        )
        lines.append(f"- **A3 stop-driver distribution**: {driver_counts}\n")
    else:
        lines.append("- **A3 stop-driver distribution**: (no overload terminations)\n")

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
        a3 = r.get("A3") or {}
        b1 = r.get("B1") or {}
        b2 = r.get("B2") or {}
        b3 = r.get("B3") or {}
        b4 = r.get("B4") or {}

        # --- A1 -----------------------------------------------------------
        sil_i = a1.get("silhouette_initial")
        sil_f = a1.get("silhouette_final")
        sil_arrow = f"{sil_i:.3f} → {sil_f:.3f}" if sil_i is not None and sil_f is not None else "(no data)"
        sil_comment = ""
        if sil_i is not None and sil_f is not None:
            delta = sil_f - sil_i
            sil_comment = f" ({'improved' if delta > 0.005 else 'dropped' if delta < -0.005 else 'stable'})"
        lines.append(f"**A1 — Cluster quality (silhouette):** {sil_arrow}{sil_comment}\n")
        lines.append("> Measures whether the final clusters are geometrically tight and well-separated. "
                     "A drop can happen when adding a cluster splits a previously cohesive group.\n\n")

        # --- A2 -----------------------------------------------------------
        turns = a2.get("turns", "?")
        wt = a2.get("weighted_turns", "?")
        term = a2.get("termination", "?")
        ops_summary = "; ".join(
            f"turn {i+1}: {', '.join(ops) or 'no ops'}"
            for i, ops in enumerate(a2.get("ops_per_turn") or [])
        )
        lines.append(f"**A2 — Dialogue efficiency:** {turns} turns (weighted {wt}), ended as `{term}`\n")
        if ops_summary:
            lines.append(f"> Operations per turn: {ops_summary}\n\n")

        # --- A3 -----------------------------------------------------------
        cog_mean = a3.get("mean_cognitive_load")
        cog_turns = a3.get("cognitive_load_by_turn") or []
        cog_str = f"{cog_mean}/5" if cog_mean is not None else "(no data)"
        lines.append(f"**A3 — Cognitive load:** mean {cog_str} per turn: {cog_turns}\n")
        drivers = a3.get("cognitive_load_driver_by_turn") or []
        if drivers:
            lines.append(f"> Driver per turn: {drivers}\n")
        lines.append("> Deterministic 1–5 score from turns / pre-trim tokens / "
                     "cluster count. High load (5) triggers early termination; "
                     "the driver names which signal saturated.\n\n")

        # --- B1 -----------------------------------------------------------
        overall = b1.get("overall_score")
        overall_str = f"{overall:.2f}" if isinstance(overall, (int, float)) else str(overall)
        notes = b1.get("notes") or ""
        lines.append(f"**B1 — Overall quality (synthesis judge):** {overall_str}/1.0\n")
        if notes:
            lines.append(f"> {notes}\n\n")

        # --- B2 -----------------------------------------------------------
        c_mean = b2.get("coherence_mean")
        c_min = b2.get("coherence_min")
        per_cluster = b2.get("per_cluster") or []
        if isinstance(c_mean, (int, float)):
            weakest_str = ""
            if per_cluster:
                w = min(per_cluster, key=lambda r: r.get("coherence", 1.0))
                weakest_name = w.get("cluster_name") or "unnamed"
                weakest_str = (f" — weakest: `{weakest_name}` @ {w.get('coherence', 0):.2f} "
                               f"({w.get('reasoning', '')})")
            lines.append(f"**B2 — Coherence:** mean {c_mean:.2f}, min {c_min:.2f}{weakest_str}\n\n")
        else:
            lines.append("**B2 — Coherence:** (no data)\n\n")

        # --- B3 -----------------------------------------------------------
        comp = b3.get("compliance_score")
        comp_notes = b3.get("notes") or ""
        if isinstance(comp, (int, float)):
            lines.append(f"**B3 — Compliance:** {comp:.2f}/1.0\n")
            if comp_notes:
                lines.append(f"> {comp_notes}\n\n")
        else:
            lines.append("**B3 — Compliance:** (no data)\n\n")

        # --- B4 -----------------------------------------------------------
        contr = b4.get("contradiction_score")
        contr_notes = b4.get("notes") or ""
        examples = b4.get("examples") or []
        if isinstance(contr, (int, float)):
            lines.append(
                f"**B4 — Oracle contradiction:** {contr:.2f}/1.0 "
                f"_(higher = oracle harder to understand)_\n"
            )
            if contr_notes:
                lines.append(f"> {contr_notes}\n")
            for ex in examples:
                lines.append(f"> - {ex}\n")
            lines.append("\n")
        else:
            lines.append("**B4 — Oracle contradiction:** (no data)\n\n")

        lines.append(f"**Clusters:** {r.get('k_initial', '?')} → {r.get('k_final')}  |  "
                     f"**Wall time:** {r.get('wall_time_s')}s\n")

        if r.get("errors"):
            lines.append(f"\n> **Errors:** {r['errors']}\n")

        lines.append("\n")

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.writelines(lines)
