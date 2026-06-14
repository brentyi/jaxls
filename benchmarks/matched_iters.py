"""Matched-iteration benchmark: cost vs wall-clock at exactly k LM iterations.

"Time to convergence" is confounded by termination: a solver that happens to
stop earlier looks faster for reasons unrelated to the linear algebra. This
harness instead runs exactly k outer Levenberg-Marquardt iterations with
early termination disabled, for a sweep of k, and records (best cost so far,
wall-clock) at each k. Solvers are compared at matched k.

Timing methodology (each rule guards against a real measurement bug):
- One full warmup solve per (solver, k), with `jax.block_until_ready` on the
  result, so neither compilation nor asynchronously dispatched device work
  leaks into the timed window.
- Each timed measurement is the minimum of 3 repeats.
- The full-CG baseline doubles as a control line: it shares no code with the
  elimination path, so its run-to-run variation is the noise floor.

Usage:
    python benchmarks/matched_iters.py            # run everything
    python benchmarks/matched_iters.py --replot   # regenerate plots from JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt  # noqa: E402

import jaxls  # noqa: E402

from bal import download_bal, load_bal, make_toy_ba, run_k_iterations  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"

_STYLE = {
    "full CG": ("#d62728", "o", "-"),
    "full dense": ("#7f7f7f", "s", ":"),
    "Schur + dense": ("#1f77b4", "^", "-"),
    "Schur + CG": ("#2ca02c", "D", "-"),
}


# The measurement kernel lives in bal.py (shared with device_sweep.py) so
# the two scripts cannot drift apart methodologically.
run_solver = run_k_iterations


def run_study(
    name: str,
    problem: jaxls.AnalyzedLeastSquaresProblem,
    problem_full: jaxls.AnalyzedLeastSquaresProblem,
    initial_vals: jaxls.VarValues,
    solvers: dict[str, dict],
    ks: list[int],
) -> dict:
    results: dict = {"ks": ks, "solvers": {}}
    for solver_name, kwargs in solvers.items():
        kwargs = dict(kwargs)
        target = problem_full if kwargs.pop("elimination", True) is False else problem
        costs, times = [], []
        for k in ks:
            best_cost, elapsed = run_solver(target, initial_vals, k, **kwargs)
            costs.append(best_cost)
            times.append(elapsed)
            print(
                f"  {solver_name:<14} k={k:>2}: {elapsed:>7.2f}s "
                f"best_cost={best_cost:g}"
            )
        results["solvers"][solver_name] = {"costs": costs, "times": times}
    out = RESULTS_DIR / f"matched_{name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return results


def plot_study(name: str, title: str) -> None:
    data = json.loads((RESULTS_DIR / f"matched_{name}.json").read_text())
    ks = data["ks"]
    fig, (ax_cost, ax_time) = plt.subplots(1, 2, figsize=(11, 4))
    for solver_name, series in data["solvers"].items():
        color, marker, ls = _STYLE[solver_name]
        ax_cost.plot(
            ks, series["costs"], marker=marker, ls=ls, color=color, label=solver_name
        )
        ax_time.plot(
            ks, series["times"], marker=marker, ls=ls, color=color, label=solver_name
        )
    ax_cost.set_yscale("log")
    ax_cost.set_xlabel("outer LM iterations k")
    ax_cost.set_ylabel("best cost (log scale)")
    ax_time.set_xlabel("outer LM iterations k")
    ax_time.set_ylabel("wall-clock (s)")
    ax_cost.legend()
    fig.suptitle(title)
    fig.tight_layout()
    out = RESULTS_DIR / f"matched_{name}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)


def plot_tail(name: str, title: str, y_window: float = 0.03) -> None:
    """Linear-y zoom on the convergence tail. The interesting cost
    differences are ~0.2%; a log axis spanning 1e6 to 3e4 hides them."""
    data = json.loads((RESULTS_DIR / f"matched_{name}.json").read_text())
    ks = data["ks"]
    best = min(min(s["costs"]) for s in data["solvers"].values())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for solver_name, series in data["solvers"].items():
        color, marker, ls = _STYLE[solver_name]
        ax.plot(
            ks, series["costs"], marker=marker, ls=ls, color=color, label=solver_name
        )
        for k, cost, t in zip(ks, series["costs"], series["times"]):
            if cost < best * (1 + y_window):
                ax.annotate(
                    f"{t:.2f}s",
                    (k, cost),
                    textcoords="offset points",
                    xytext=(4, 5),
                    fontsize=7,
                    color=color,
                )
    ax.set_ylim(best * 0.9995, best * (1 + y_window))
    ax.set_xlabel("outer LM iterations k")
    ax.set_ylabel("best cost (linear scale, zoomed)")
    ax.axhline(best, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.legend()
    ax.set_title(f"{title} — convergence tail (wall-clock annotated)")
    fig.tight_layout()
    out = RESULTS_DIR / f"matched_{name}_tail.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--skip-toy", action="store_true")
    parser.add_argument("--skip-ladybug", action="store_true")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    if not args.replot:
        if not args.skip_toy:
            print("## Toy (30 cams, 700 points) — full dense feasible")
            problem, init = make_toy_ba()
            problem_full, _ = make_toy_ba(schur_elimination=False)
            run_study(
                "toy",
                problem,
                problem_full,
                init,
                solvers={
                    "full CG": dict(
                        linear_solver="conjugate_gradient", elimination=False
                    ),
                    "Schur + dense": dict(linear_solver="dense_cholesky"),
                    "Schur + CG": dict(linear_solver="conjugate_gradient"),
                    "full dense": dict(
                        linear_solver="dense_cholesky", elimination=False
                    ),
                },
                ks=[1, 2, 3, 4, 6, 8, 12, 16],
            )

        if not args.skip_ladybug:
            print("## Ladybug-49 (real BAL) — full dense infeasible")
            path = download_bal(
                "ladybug/problem-49-7776-pre.txt.bz2", Path("/tmp/ladybug-49.txt")
            )
            problem, init = load_bal(path)
            problem_full, _ = load_bal(path, schur_elimination=False)
            run_study(
                "ladybug49",
                problem,
                problem_full,
                init,
                solvers={
                    "full CG": dict(
                        linear_solver="conjugate_gradient", elimination=False
                    ),
                    "Schur + dense": dict(linear_solver="dense_cholesky"),
                    "Schur + CG": dict(linear_solver="conjugate_gradient"),
                },
                ks=[1, 2, 4, 6, 9, 13, 18, 24, 30],
            )

    if not args.skip_toy:
        plot_study("toy", "Toy BA, matched outer iterations (float64, CPU)")
    if not args.skip_ladybug:
        plot_study("ladybug49", "Ladybug-49, matched outer iterations (float64, CPU)")
        plot_tail("ladybug49", "Ladybug-49")


if __name__ == "__main__":
    main()
