"""Generate charts for docs/generalization-stability-report.md.

Reads the six per-session logs in logs/generalization/, extracts the metrics
+ CIs the script already bootstrapped, computes Wilson binomial CIs for A4
OOD rates (which the script reports point-estimate-only), and writes:

  docs/generalization-a1-forest.png   — paired Δ on existing-point silhouette
  docs/generalization-b2-forest.png   — paired Δ on cluster-coherence
  docs/generalization-a4-ood.png      — pooled OOD rate vs 5% baseline
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Data extracted verbatim from logs/generalization/user{1..6}_*.log.
# The fields here are exactly what the script printed — no recomputation,
# no aggregation. Wilson CIs for A4 OOD rates are derived from (n, rate)
# below, since the script only reports the point estimate.
SESSIONS = [
    {
        "label": "user1/newsgroups",
        "k_live": 8,
        "n_new": 300,
        "a1_t0": 0.0432, "a1_t1": 0.0426,
        "a1_paired_delta": +0.0001, "a1_paired_lo": -0.0000, "a1_paired_hi": +0.0002,
        "a1_new_mean": 0.0398, "a1_new_lo": 0.0353, "a1_new_hi": 0.0442,
        "a4_ood_rate": 0.087,
        "a4_mean_z": +0.2476, "a4_mean_z_lo": +0.1348, "a4_mean_z_hi": +0.3643,
        "b2_t0_mean": 0.656, "b2_t0_min": 0.350,
        "b2_t1_mean": 0.663, "b2_t1_min": 0.450,
        "b2_paired_delta": +0.0062, "b2_paired_lo": -0.0750, "b2_paired_hi": +0.0813,
        "a2_turns": 5, "a3_load": 2.0,
        "b1_overall": 0.650, "b3_compliance": 1.000, "b4_contradiction": 0.700,
    },
    {
        "label": "user2/amazon",
        "k_live": 6,
        "n_new": 300,
        "a1_t0": 0.0321, "a1_t1": 0.0322,
        "a1_paired_delta": -0.0005, "a1_paired_lo": -0.0007, "a1_paired_hi": -0.0003,
        "a1_new_mean": 0.0348, "a1_new_lo": 0.0306, "a1_new_hi": 0.0391,
        "a4_ood_rate": 0.053,
        "a4_mean_z": +0.0412, "a4_mean_z_lo": -0.0703, "a4_mean_z_hi": +0.1515,
        "b2_t0_mean": 0.750, "b2_t0_min": 0.550,
        "b2_t1_mean": 0.575, "b2_t1_min": 0.400,
        "b2_paired_delta": -0.1750, "b2_paired_lo": -0.2583, "b2_paired_hi": -0.0833,
        "a2_turns": 4, "a3_load": 1.5,
        "b1_overall": 0.550, "b3_compliance": 0.500, "b4_contradiction": 0.600,
    },
    {
        "label": "user3/imdb",
        "k_live": 7,
        "n_new": 300,
        "a1_t0": -0.0053, "a1_t1": -0.0047,
        "a1_paired_delta": +0.0004, "a1_paired_lo": +0.0002, "a1_paired_hi": +0.0005,
        "a1_new_mean": -0.0038, "a1_new_lo": -0.0077, "a1_new_hi": +0.0001,
        "a4_ood_rate": 0.080,
        "a4_mean_z": +0.2031, "a4_mean_z_lo": +0.0859, "a4_mean_z_hi": +0.3269,
        "b2_t0_mean": 0.679, "b2_t0_min": 0.500,
        "b2_t1_mean": 0.679, "b2_t1_min": 0.550,
        "b2_paired_delta": +0.0000, "b2_paired_lo": -0.0800, "b2_paired_hi": +0.0586,
        "a2_turns": 3, "a3_load": 2.0,
        "b1_overall": 0.450, "b3_compliance": 0.330, "b4_contradiction": 0.700,
    },
    {
        "label": "user4/newsgroups",
        "k_live": 5,
        "n_new": 300,
        "a1_t0": 0.0176, "a1_t1": 0.0199,
        "a1_paired_delta": +0.0003, "a1_paired_lo": -0.0000, "a1_paired_hi": +0.0006,
        "a1_new_mean": 0.0279, "a1_new_lo": 0.0233, "a1_new_hi": 0.0328,
        "a4_ood_rate": 0.050,
        "a4_mean_z": +0.0479, "a4_mean_z_lo": -0.0660, "a4_mean_z_hi": +0.1614,
        "b2_t0_mean": 0.680, "b2_t0_min": 0.450,
        "b2_t1_mean": 0.660, "b2_t1_min": 0.450,
        "b2_paired_delta": -0.0200, "b2_paired_lo": -0.0900, "b2_paired_hi": +0.0300,
        "a2_turns": 4, "a3_load": 1.0,
        "b1_overall": 0.650, "b3_compliance": 1.000, "b4_contradiction": 0.400,
    },
    {
        "label": "user5/amazon",
        "k_live": 5,
        "n_new": 300,
        "a1_t0": 0.0145, "a1_t1": 0.0159,
        "a1_paired_delta": -0.0001, "a1_paired_lo": -0.0005, "a1_paired_hi": +0.0003,
        "a1_new_mean": 0.0221, "a1_new_lo": 0.0190, "a1_new_hi": 0.0252,
        "a4_ood_rate": 0.017,
        "a4_mean_z": -0.1175, "a4_mean_z_lo": -0.2211, "a4_mean_z_hi": -0.0137,
        "b2_t0_mean": 0.530, "b2_t0_min": 0.350,
        "b2_t1_mean": 0.560, "b2_t1_min": 0.400,
        "b2_paired_delta": +0.0300, "b2_paired_lo": -0.0800, "b2_paired_hi": +0.1600,
        "a2_turns": 3, "a3_load": 1.33,
        "b1_overall": 0.650, "b3_compliance": 0.800, "b4_contradiction": 0.400,
    },
    {
        "label": "user6/imdb",
        "k_live": 3,
        "n_new": 300,
        "a1_t0": -0.0023, "a1_t1": -0.0007,
        "a1_paired_delta": -0.0007, "a1_paired_lo": -0.0012, "a1_paired_hi": -0.0003,
        "a1_new_mean": 0.0088, "a1_new_lo": 0.0063, "a1_new_hi": 0.0113,
        "a4_ood_rate": 0.043,
        "a4_mean_z": +0.0538, "a4_mean_z_lo": -0.0476, "a4_mean_z_hi": +0.1595,
        "b2_t0_mean": 0.757, "b2_t0_min": 0.700,
        "b2_t1_mean": 0.700, "b2_t1_min": 0.650,
        "b2_paired_delta": -0.0567, "b2_paired_lo": -0.1200, "b2_paired_hi": +0.0000,
        "a2_turns": 5, "a3_load": 1.2,
        "b1_overall": 0.550, "b3_compliance": 0.500, "b4_contradiction": 0.600,
    },
]
OOD_BASELINE = 0.05  # base d²_95 → 5% by construction


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. (k = successes / OOD-flagged)."""
    if n == 0:
        return (0.0, 0.0)
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def forest_paired_delta(metric_key: str, title: str, xlabel: str, outpath: Path) -> None:
    """Horizontal forest plot of paired Δ + CI, sorted by Δ ascending."""
    rows = sorted(SESSIONS, key=lambda s: s[f"{metric_key}_paired_delta"])
    labels = [s["label"] for s in rows]
    deltas = np.array([s[f"{metric_key}_paired_delta"] for s in rows])
    los = np.array([s[f"{metric_key}_paired_lo"] for s in rows])
    his = np.array([s[f"{metric_key}_paired_hi"] for s in rows])
    err = np.vstack([deltas - los, his - deltas])

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(rows))
    colors = ["#d62728" if hi < 0 else ("#2ca02c" if lo > 0 else "#7f7f7f")
              for lo, hi in zip(los, his)]
    ax.errorbar(deltas, y, xerr=err, fmt="o", ecolor="#444", elinewidth=1.4,
                capsize=4, markersize=7, markerfacecolor="white", zorder=3)
    for yi, c in zip(y, colors):
        ax.scatter([deltas[yi]], [yi], s=80, color=c, zorder=4, edgecolor="black",
                   linewidth=0.8)
    ax.axvline(0, color="#888", linewidth=1, linestyle="--", zorder=1)
    ax.set_yticks(y, labels=labels, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"wrote {outpath}")


def a4_ood_bar(outpath: Path) -> None:
    """Per-session pooled OOD rate with Wilson CI; horizontal reference at 5% baseline."""
    rows = sorted(SESSIONS, key=lambda s: s["a4_ood_rate"])
    labels = [s["label"] for s in rows]
    rates = np.array([s["a4_ood_rate"] for s in rows])
    cis = [wilson_ci(int(round(s["a4_ood_rate"] * s["n_new"])), s["n_new"]) for s in rows]
    los = np.array([c[0] for c in cis])
    his = np.array([c[1] for c in cis])
    err = np.vstack([rates - los, his - rates])

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(rows))
    colors = ["#d62728" if lo > OOD_BASELINE else ("#2ca02c" if hi < OOD_BASELINE else "#7f7f7f")
              for lo, hi in zip(los, his)]
    ax.bar(x, rates * 100, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.errorbar(x, rates * 100, yerr=err * 100, fmt="none", ecolor="#222",
                elinewidth=1.2, capsize=4, zorder=3)
    ax.axhline(OOD_BASELINE * 100, color="#444", linewidth=1, linestyle="--",
               label=f"null baseline ({int(OOD_BASELINE * 100)}%)")
    ax.set_xticks(x, labels=labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Pooled OOD rate (%)")
    ax.set_title(
        "A4 — pooled OOD rate on new arrivals (Wilson 95% CI)\n"
        "red: above baseline · grey: CI spans baseline · green: below baseline",
        fontsize=11,
    )
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"wrote {outpath}")


def b2_t0_vs_t1(outpath: Path) -> None:
    """Mean coherence at t0 vs t1 (dumbbell). Per-session arrow shows the shift."""
    rows = sorted(SESSIONS, key=lambda s: s["b2_t0_mean"])
    labels = [s["label"] for s in rows]
    t0 = np.array([s["b2_t0_mean"] for s in rows])
    t1 = np.array([s["b2_t1_mean"] for s in rows])

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(rows))
    for yi, a, b in zip(y, t0, t1):
        color = "#d62728" if b < a else ("#2ca02c" if b > a else "#7f7f7f")
        ax.annotate("", xy=(b, yi), xytext=(a, yi),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
    ax.scatter(t0, y, s=80, color="#1f77b4", label="t0 (pre-ingestion)", zorder=3,
               edgecolor="black", linewidth=0.5)
    ax.scatter(t1, y, s=80, color="#ff7f0e", label="t1 (post-ingestion)", zorder=3,
               edgecolor="black", linewidth=0.5)
    ax.set_yticks(y, labels=labels, fontsize=10)
    ax.set_xlabel("Mean cluster coherence (B2 judge)")
    ax.set_xlim(0.4, 0.85)
    ax.set_title("B2 — mean cluster coherence shift around the ingestion event",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"wrote {outpath}")


def print_wilson_table() -> None:
    print("\nA4 OOD pooled rate — Wilson 95% CIs:")
    for s in SESSIONS:
        n = s["n_new"]
        k = int(round(s["a4_ood_rate"] * n))
        lo, hi = wilson_ci(k, n)
        print(f"  {s['label']:22s} n={n} k={k:3d}  rate={s['a4_ood_rate'] * 100:5.1f}%  "
              f"CI [{lo * 100:4.1f}%, {hi * 100:4.1f}%]")


def main() -> None:
    docs = Path(__file__).resolve().parent.parent / "docs"
    docs.mkdir(exist_ok=True)
    forest_paired_delta(
        "a1",
        "A1 — paired Δ silhouette on pre-existing points (bootstrap 95% CI)",
        "Δ silhouette (t1 − t0)",
        docs / "generalization-a1-forest.png",
    )
    forest_paired_delta(
        "b2",
        "B2 — paired Δ cluster coherence (bootstrap 95% CI over clusters)",
        "Δ coherence (t1 − t0)",
        docs / "generalization-b2-forest.png",
    )
    a4_ood_bar(docs / "generalization-a4-ood.png")
    b2_t0_vs_t1(docs / "generalization-b2-shift.png")
    print_wilson_table()


if __name__ == "__main__":
    main()
