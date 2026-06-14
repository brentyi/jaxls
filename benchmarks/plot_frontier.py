"""PR-friendly summary figure: cost vs LM iterations, and cost vs wall-clock.

Reads the matched-iteration data saved by `matched_iters.py` and renders a
two-panel comparison of full CG, Schur + CG, and Schur + dense on
Ladybug-49. Usage: python benchmarks/plot_frontier.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

RESULTS_DIR = Path(__file__).parent / "results"

STYLE = {
    "full CG": dict(color="#d62728", marker="o"),
    "Schur + CG": dict(color="#2ca02c", marker="D"),
    "Schur + dense": dict(color="#1f77b4", marker="^"),
}


def main() -> None:
    data = json.loads((RESULTS_DIR / "matched_ladybug49.json").read_text())
    ks = data["ks"]

    fig, (ax_k, ax_t) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

    for name in STYLE:
        series = data["solvers"][name]
        common = dict(
            label=name,
            lw=1.8,
            ms=5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            **STYLE[name],
        )
        ax_k.plot(ks, series["costs"], **common)
        ax_t.plot(series["times"], series["costs"], **common)

    best = min(min(s["costs"]) for s in data["solvers"].values())
    for ax in (ax_k, ax_t):
        ax.axhline(best, color="0.6", lw=0.8, ls=(0, (4, 4)), zorder=0)
        ax.set_ylim(best - 30, 27_900)
        ax.grid(True, which="both", color="0.92", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax_k.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    ax_k.set_xlabel("Levenberg–Marquardt iterations")
    ax_k.set_ylabel("cost (accepted solution)")
    ax_k.set_xticks(ks)
    ax_k.legend(frameon=False, loc="upper right")

    ax_t.set_xscale("log")
    ax_t.set_xlabel("wall-clock (s)")

    # Headline annotation: time to reach full CG's best-ever cost.
    full_cg = data["solvers"]["full CG"]
    schur_dense = data["solvers"]["Schur + dense"]
    t_cg, c_cg = full_cg["times"][-1], full_cg["costs"][-1]
    idx = next(i for i, c in enumerate(schur_dense["costs"]) if c <= c_cg)
    t_sd = schur_dense["times"][idx]
    ax_t.annotate(
        f"{t_cg / t_sd:.0f}× less wall-clock\nto full CG's best cost",
        xy=(t_sd, schur_dense["costs"][idx]),
        xytext=(t_sd * 2.6, c_cg + 380),
        fontsize=9,
        color="#1f77b4",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0),
    )

    fig.suptitle(
        "Bundle adjustment, Ladybug-49 (BAL): variable elimination vs full-system CG",
        fontsize=12,
    )
    ax_k.set_title("matched LM iterations", fontsize=10, color="0.35")
    ax_t.set_title("same data, against wall-clock", fontsize=10, color="0.35")
    fig.text(
        0.5,
        0.005,
        "float64, CPU (Apple Silicon) · exactly k LM iterations, early termination "
        "disabled · warmup excluded, min of 3 runs",
        ha="center",
        fontsize=8,
        color="0.45",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = RESULTS_DIR / "frontier_ladybug49.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
