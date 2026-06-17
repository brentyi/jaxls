"""Cross-device benchmark: linear-solver methods on CPU vs GPU.

Runs a matched-iteration study on both CPU and GPU, across a sweep of problem
sizes, comparing every available linear solver:

  - full CG       : conjugate gradient on the full system (no elimination)
  - full dense    : dense Cholesky on the full system (small problems only)
  - cholmod         : sparse Cholesky on the full system (CPU only)
  - Schur + dense   : variable elimination, dense reduced solve
  - Schur + CG      : variable elimination, matrix-free reduced CG
  - Schur + cholmod : variable elimination, sparse-direct reduced solve (CPU only)

Methodology: run exactly k Levenberg-Marquardt iterations with early
termination off, for a sweep of k, and record
(accepted cost, wall-clock) at each k. One warmup solve per configuration
with `jax.block_until_ready` absorbs compilation and async dispatch; each
timed point is the min of `repeats`.

Device selection is by `jax.devices(platform)`; the array inputs are placed
on the target device with `jax.device_put` and the solve is run there. The
CPU device is always available; the GPU rows are skipped if no GPU backend
is present.

Usage:
    uv run --extra dev --extra docs python benchmarks/device_sweep.py                 # run everything
    uv run --extra dev --extra docs python benchmarks/device_sweep.py --replot        # plots from JSON only
    uv run --extra dev --extra docs python benchmarks/device_sweep.py --problems toy ladybug49
    uv run --extra dev --extra docs python benchmarks/device_sweep.py --devices cpu   # CPU only
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)


import jaxls  # noqa: E402

from bal import download_bal, load_bal, make_toy_ba, run_k_iterations  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Problem registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    title: str
    ks: tuple[int, ...]
    full_dense_ok: bool
    """Whether a dense full-system solve is feasible (small problems only)."""
    load: Callable[[bool], tuple[jaxls.AnalyzedLeastSquaresProblem, jaxls.VarValues]]
    """load(schur_elimination) -> (problem, initial_vals)."""


def _bal_loader(
    name: str, dest: str
) -> Callable[[bool], tuple[jaxls.AnalyzedLeastSquaresProblem, jaxls.VarValues]]:
    def load(
        schur_elimination: bool,
    ) -> tuple[jaxls.AnalyzedLeastSquaresProblem, jaxls.VarValues]:
        path = download_bal(name, Path(dest))
        return load_bal(path, schur_elimination=schur_elimination)

    return load


PROBLEMS: dict[str, ProblemSpec] = {
    "toy": ProblemSpec(
        name="toy",
        title="Toy BA (30 cams, 700 points)",
        ks=(1, 2, 3, 4, 6, 8, 12, 16),
        full_dense_ok=True,
        load=lambda se: make_toy_ba(schur_elimination=se),
    ),
    "ladybug49": ProblemSpec(
        name="ladybug49",
        title="Ladybug-49 (49 cams, 7,776 pts, 31,843 obs)",
        ks=(1, 2, 4, 6, 9, 13, 18, 24, 30),
        full_dense_ok=False,
        load=_bal_loader("ladybug/problem-49-7776-pre.txt.bz2", "/tmp/ladybug-49.txt"),
    ),
    # Larger: 138 cameras -> 1,242-dim reduced system. Stresses the dense
    # O(n_keep^3) factorization and the GPU's strength at big matmuls.
    "trafalgar138": ProblemSpec(
        name="trafalgar138",
        title="Trafalgar-138 (138 cams, 44,033 pts)",
        ks=(1, 2, 4, 6, 9, 13, 18, 24, 30),
        full_dense_ok=False,
        load=_bal_loader(
            "trafalgar/problem-138-44033-pre.txt.bz2", "/tmp/trafalgar-138.txt"
        ),
    ),
}


# Which methods to run, and how. `elimination=False` selects the
# full-system problem; `devices` restricts a method to a subset.
@dataclass(frozen=True)
class Method:
    name: str
    linear_solver: str
    elimination: bool
    devices: tuple[str, ...] | None = None  # None -> all devices
    full_dense_only: bool = False  # only on problems where full dense is feasible


METHODS: tuple[Method, ...] = (
    Method("full CG", "conjugate_gradient", elimination=False),
    Method("cholmod", "cholmod", elimination=False, devices=("cpu",)),
    Method("Schur + dense", "dense_cholesky", elimination=True),
    Method("Schur + CG", "conjugate_gradient", elimination=True),
    # Variable elimination with a sparse-direct CHOLMOD factorization of the
    # reduced system (Ceres/g2o-style). CPU only: CHOLMOD runs as a host
    # callback.
    Method("Schur + cholmod", "cholmod", elimination=True, devices=("cpu",)),
    Method(
        "full dense",
        "dense_cholesky",
        elimination=False,
        full_dense_only=True,
    ),
)


# ---------------------------------------------------------------------------
# Timing — the measurement kernel itself is bal.run_k_iterations, shared with
# the benchmark suite so they cannot drift apart methodologically.
# ---------------------------------------------------------------------------


def device_for(platform: str) -> "jax.Device | None":  # type: ignore[name-defined]
    try:
        return jax.devices(platform)[0]
    except RuntimeError:
        return None


def run_study(spec: ProblemSpec, devices: list[str], repeats: int) -> dict:
    """Run the full method x device x k grid for one problem."""
    print(f"\n## {spec.title}")
    problem_elim, init = spec.load(True)
    problem_full, _ = spec.load(False)

    results: dict = {"title": spec.title, "ks": list(spec.ks), "runs": {}}
    for platform in devices:
        dev = device_for(platform)
        if dev is None:
            print(f"  [{platform}] no device, skipping")
            continue
        print(f"  --- device: {platform} ({dev}) ---")
        # Place each problem variant on the device once; every (method, k)
        # measurement reuses the placed arrays.
        elim_d = jax.device_put(problem_elim, dev)
        full_d = jax.device_put(problem_full, dev)
        init_d = jax.device_put(init, dev)
        for method in METHODS:
            if method.devices is not None and platform not in method.devices:
                continue
            if method.full_dense_only and not spec.full_dense_ok:
                continue
            target = elim_d if method.elimination else full_d
            key = f"{platform}:{method.name}"
            costs, times = [], []
            try:
                for k in spec.ks:
                    cost, t = run_k_iterations(
                        target,
                        init_d,
                        k,
                        linear_solver=method.linear_solver,
                        repeats=repeats,
                    )
                    costs.append(cost)
                    times.append(t)
                    print(f"    {method.name:<14} k={k:>2}: {t:>8.3f}s  cost={cost:g}")
            except Exception as e:  # noqa: BLE001
                print(f"    {method.name:<14} FAILED: {type(e).__name__}: {e}")
                continue
            results["runs"][key] = {
                "method": method.name,
                "device": platform,
                "ks": list(spec.ks),
                "costs": costs,
                "times": times,
            }
            gc.collect()

    out = RESULTS_DIR / f"device_{spec.name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  wrote {out}")
    return results


# Damping used for the cholmod-vs-Schur+cholmod comparison. Bundle-adjustment
# curvature scales want lambda ~1e1-1e3; 1e2 matches the benchmark suite
# (suite/workloads.py ba_lambda_initial) and is applied identically to both
# methods so the comparison is apples-to-apples.
_CHOLMOD_BA_LAMBDA = 1e2


def run_cholmod_study(spec: ProblemSpec, repeats: int) -> dict:
    """CPU-only matched-iteration study of full-system CHOLMOD vs Schur +
    CHOLMOD (sparse-direct on the reduced system), both with the same tuned
    damping. Writes device_<name>_cholmod.json."""
    print(f"\n## {spec.title} — CHOLMOD comparison")
    problem_elim, init = spec.load(True)
    problem_full, _ = spec.load(False)
    dev = device_for("cpu")
    assert dev is not None  # CPU is always available.
    elim_d = jax.device_put(problem_elim, dev)
    full_d = jax.device_put(problem_full, dev)
    init_d = jax.device_put(init, dev)

    results: dict = {"title": spec.title, "ks": list(spec.ks), "runs": {}}
    for label, target, elim in (
        ("cholmod", full_d, False),
        ("Schur + cholmod", elim_d, True),
    ):
        del elim  # `target` already selects the right problem variant.
        costs, times = [], []
        for k in spec.ks:
            cost, t = run_k_iterations(
                target,
                init_d,
                k,
                linear_solver="cholmod",
                repeats=repeats,
                lambda_initial=_CHOLMOD_BA_LAMBDA,
            )
            costs.append(cost)
            times.append(t)
            print(f"    {label:<16} k={k:>2}: {t:>8.3f}s  cost={cost:g}")
        results["runs"][f"cpu:{label}"] = {
            "method": label,
            "device": "cpu",
            "ks": list(spec.ks),
            "costs": costs,
            "times": times,
        }
        gc.collect()

    out = RESULTS_DIR / f"device_{spec.name}_cholmod.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  wrote {out}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Method styling. Full-system methods get warm/neutral colors and a thin line;
# Schur (variable-elimination) methods get a cool family and a thicker line, so
# elimination-vs-full reads at a glance without spending the linestyle (which
# encodes tuned-PR "-" vs main-baseline "--").
_COLOR = {
    "full CG": "#d62728",  # red
    "full dense": "#7f7f7f",  # gray
    "cholmod": "#e377c2",  # pink
    "Schur + dense": "#1f77b4",  # blue
    "Schur + CG": "#2ca02c",  # green
    "Schur + cholmod": "#17becf",  # teal
}
_SCHUR_LINEWIDTH = 2.8
_FULL_LINEWIDTH = 1.4

# Draw order: full-system methods first, then Schur, so the legend groups them.
_METHOD_ORDER = (
    "full CG",
    "full dense",
    "cholmod",
    "Schur + dense",
    "Schur + CG",
    "Schur + cholmod",
)


def _is_schur(method: str) -> bool:
    return method.startswith("Schur")


def _load_runs(name: str) -> "tuple[dict, dict, str] | None":
    """Gather plot data for one problem. Returns (final_runs, main_runs, title),
    each runs dict keyed by "device:method".

    Sources, in priority order: the tuned PR JSON, then the plain sweep JSON
    (so methods added after the tuned pair was generated still appear; a
    device:method already present from the tuned data is not overwritten). The
    tuned main-baseline JSON is returned separately for the dashed comparison."""
    final: dict = {}
    main: dict = {}
    title = None
    tuned_final = RESULTS_DIR / f"device_{name}_tuned_final.json"
    if tuned_final.exists():
        d = json.loads(tuned_final.read_text())
        title = d["title"]
        final.update(d["runs"])
        tuned_main = RESULTS_DIR / f"device_{name}_tuned_main.json"
        if tuned_main.exists():
            main.update(json.loads(tuned_main.read_text())["runs"])
    plain = RESULTS_DIR / f"device_{name}.json"
    if plain.exists():
        d = json.loads(plain.read_text())
        title = title or d["title"]
        for k, r in d["runs"].items():
            final.setdefault(k, r)  # don't overwrite a tuned curve
    if not final:
        return None
    return final, main, title or name


def _time_to(run: dict, target: float) -> float | None:
    """Wall-clock at which a run first reaches `target` cost (or below)."""
    for t, c in zip(run["times"], run["costs"]):
        if c <= target:
            return t
    return None


def plot_study(name: str) -> None:
    """One figure per device, two stacked panels telling the elimination story:

    - top: suboptimality gap (cost - best-observed optimum) vs wall-clock,
      log-log. A method's curve plunging to the floor marks when it reaches the
      solution; its horizontal position is the time-to-solution. Schur
      (variable-elimination) methods are cool-colored and thick, full-system
      methods warm/neutral and thin, so elimination-vs-full reads at a glance.
    - bottom: speedup-to-solution of each method over the fastest full-system
      method (bars). This is the headline number.

    The tuned main-system baseline (if present) is drawn dashed in the top
    panel for context; it is not included in the speedup bars."""
    loaded = _load_runs(name)
    if loaded is None:
        return
    final, main, title = loaded

    for device in ("cpu", "gpu"):
        runs = {r["method"]: r for r in final.values() if r["device"] == device}
        if not runs:
            continue
        main_d = {r["method"]: r for r in main.values() if r["device"] == device}
        out = RESULTS_DIR / f"device_{name}_{device}.png"
        _render_study_figure(runs, main_d, f"{title} — {device.upper()}", out)


def plot_cholmod_comparison(name: str) -> None:
    """Focused CPU comparison: full-system CHOLMOD vs Schur + CHOLMOD, from
    device_<name>_cholmod.json. Same two-panel layout as plot_study; the
    speedup bar is Schur + CHOLMOD over full-system CHOLMOD."""
    path = RESULTS_DIR / f"device_{name}_cholmod.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    runs = {r["method"]: r for r in data["runs"].values() if r["device"] == "cpu"}
    if not runs:
        return
    out = RESULTS_DIR / f"device_{name}_cholmod.png"
    _render_study_figure(runs, {}, f"{data['title']} — CHOLMOD (CPU)", out)


def _render_study_figure(runs: dict, main: dict, title: str, out: "Path") -> None:
    """Render one two-panel study figure (top: suboptimality gap vs wall-clock;
    bottom: speedup-to-solution bars over the fastest full-system method) for a
    set of method runs (keyed by method name). `main` holds optional dashed
    main-system baselines."""
    import matplotlib.pyplot as plt
    import numpy as np

    methods = [m for m in _METHOD_ORDER if m in runs]
    if not methods:
        return
    copt = min(min(r["costs"]) for r in runs.values())
    # "Everyone reaches this" threshold, for an apples-to-apples speedup.
    target = max(r["costs"][-1] for r in runs.values()) * 1.03

    fig = plt.figure(figsize=(7.0, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0])
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # --- top: suboptimality gap vs wall-clock ---
    for m in methods:
        color, is_schur = _COLOR[m], _is_schur(m)
        r = runs[m]
        gap = [max(c - copt, 1.0) for c in r["costs"]]
        ax.plot(
            r["times"],
            gap,
            marker="o",
            ms=3.5,
            lw=_SCHUR_LINEWIDTH if is_schur else _FULL_LINEWIDTH,
            color=color,
            label=m,
            zorder=3 if is_schur else 2,
            solid_capstyle="round",
        )
    # Main-system baselines, dashed and thin, for context.
    for m, r in main.items():
        if m not in _COLOR:
            continue
        gap = [max(c - copt, 1.0) for c in r["costs"]]
        ax.plot(
            r["times"],
            gap,
            marker="o",
            ms=2.5,
            lw=1.3,
            ls="--",
            color=_COLOR[m],
            label=f"main: {m}",
            zorder=1,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock (s)")
    ax.set_ylabel("cost above optimum")
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.07)
    ax.legend(fontsize=8, framealpha=0.92, loc="lower left")

    # --- bottom: speedup-to-solution over fastest full-system method ---
    tt = {m: _time_to(runs[m], target) for m in methods}
    full_times = [t for m in methods if (t := tt[m]) is not None and not _is_schur(m)]
    bar_methods = [m for m in methods if tt[m] is not None]
    if full_times and bar_methods:
        base = min(full_times)
        speed = [base / t for m in bar_methods if (t := tt[m]) is not None]
        xs = np.arange(len(bar_methods))
        ax2.bar(
            xs,
            speed,
            width=0.72,
            color=[_COLOR[m] for m in bar_methods],
            edgecolor=["#222" if _is_schur(m) else "none" for m in bar_methods],
            linewidth=1.3,
            zorder=3,
        )
        ax2.axhline(1.0, color="k", lw=0.9, ls="--", alpha=0.55, zorder=1)
        ax2.set_yscale("log")
        ax2.set_ylim(top=max(speed) * 2.3)
        ax2.set_xticks(xs)
        ax2.set_xticklabels(
            [
                m.replace("Schur + ", "Schur\n").replace("full ", "full\n")
                for m in bar_methods
            ],
            fontsize=8,
        )
        ax2.set_ylabel("speedup to solution\n(vs fastest full-system)")
        ax2.grid(True, axis="y", alpha=0.22, zorder=0)
        for x, s in zip(xs, speed):
            ax2.text(
                x,
                s * 1.08,
                (f"{s:.0f}×" if s >= 10 else f"{s:.1f}×" if s >= 1 else f"{s:.2g}×"),
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)


def plot_ba_comparison() -> None:
    """One figure, three subplots (toy / Ladybug-49 / Trafalgar-138), GPU.

    Each subplot is cost vs wall-clock with a marker per LM step, comparing
    main's full-system CG baseline against the PR's Schur+dense and
    Schur+CG. Reads the tuned PR/main JSON pair per problem."""
    import matplotlib.pyplot as plt

    problems = [
        ("toy", "Toy (30 cams)"),
        ("ladybug49", "Ladybug-49"),
        ("trafalgar138", "Trafalgar-138"),
    ]
    # (label, source-file suffix, method, color, marker)
    series = [
        ("full CG (main)", "main", "full CG", "#d62728", "o"),
        ("Schur+dense (PR)", "final", "Schur + dense", "#1f77b4", "^"),
        ("Schur+CG (PR)", "final", "Schur + CG", "#2ca02c", "s"),
    ]

    def _running_best(run: dict, c0: float) -> tuple[list[float], list[float]]:
        # "Best solution found by time t." Matched-k runs independent solves
        # per k, so an unconverged inexact-CG baseline can be non-monotone in
        # k; cummin gives the honest, comparable curve. Step 0 (initial cost
        # at t=0) is the common anchor every method descends from.
        times = [0.0] + list(run["times"])
        best = [c0] + list(run["costs"])
        for i in range(1, len(best)):
            best[i] = min(best[i], best[i - 1])
        return times, best

    def _time_to_cost(run: dict, c0: float, target: float) -> float | None:
        """First wall-clock time the running-best cost reaches `target`."""
        for t, b in zip(*_running_best(run, c0)):
            if b <= target:
                return t
        return None

    initial = json.loads((RESULTS_DIR / "ba_initial_cost.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    for ax, (prob, title) in zip(axes, problems):
        c0 = initial.get(prob)
        cache: dict[str, dict] = {}

        def _load(suffix: str) -> dict | None:
            if suffix not in cache:
                path = RESULTS_DIR / f"device_{prob}_tuned_{suffix}.json"
                cache[suffix] = json.loads(path.read_text()) if path.exists() else {}
            return cache[suffix] or None

        # Collect the runs present for this problem.
        runs = []  # (label, run, color, marker)
        for label, suffix, method, color, marker in series:
            data = _load(suffix)
            run = data["runs"].get(f"gpu:{method}") if data else None
            if run is not None:
                runs.append((label, run, color, marker))

        # Threshold: the *worst* converged cost across the three methods —
        # i.e. the best cost every method actually reaches. Labeling each
        # curve's crossing time there gives an apples-to-apples "how long to
        # the same solution?" that every method can answer (a stricter
        # threshold would leave the slowest-converging method unlabeled). A
        # small tolerance absorbs matched-k jitter so a method that lands
        # ~0.1% above its floor still counts as having reached it.
        threshold = max(min(run["costs"]) for _, run, _, _ in runs) if runs else None
        target = threshold * 1.003 if threshold is not None else None

        crossings = []  # (label, time, color) to annotate after curves drawn
        for label, run, color, marker in runs:
            times, best = _running_best(run, c0)
            ax.plot(times, best, marker=marker, ms=5, color=color, label=label)
            t_hit = _time_to_cost(run, c0, target) if target is not None else None
            if t_hit is not None and t_hit > 0:
                crossings.append((label, t_hit, color))
            if run.get("budget_stopped_at_k") is not None:
                ax.plot(times[-1], best[-1], marker="x", ms=11, mew=2, color=color)

        if threshold is not None:
            ax.axhline(threshold, color="0.4", lw=1.4, ls="--", zorder=0)
            ax.annotate(
                "shared cost target",
                (0, threshold),
                textcoords="offset points",
                xytext=(4, 4),
                ha="left",
                va="bottom",
                fontsize=8,
                color="0.4",
            )
        # Mark where each method crosses the threshold and label the absolute
        # time at that point — staggered vertically so the labels don't
        # collide, with a leader dot on the threshold line.
        crossings.sort(key=lambda c: c[1])
        for i, (label, t_hit, color) in enumerate(crossings):
            ax.plot([t_hit], [threshold], marker="o", ms=7, color=color, zorder=5)
            txt = f"{t_hit * 1e3:.0f} ms" if t_hit < 1.0 else f"{t_hit:.1f} s"
            ax.annotate(
                txt,
                (t_hit, threshold),
                textcoords="offset points",
                xytext=(0, 14 + 16 * i),
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
            )
        ax.set_yscale("log")
        # symlog: linear through t=0 (so step 0 shows), log beyond — the
        # fast Schur methods (~0.1 s) and the slow baseline (~10-30 s)
        # otherwise can't share a readable x-axis.
        ax.set_xscale("symlog", linthresh=1e-2)
        ax.set_xlim(left=0)
        ax.set_xlabel("wall-clock (s, symlog)", fontsize=11)
        ax.set_ylabel("accepted cost", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.tick_params(labelsize=10)
        ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
    fig.suptitle(
        "Bundle adjustment on GPU: cost vs wall-clock — Schur elimination "
        "vs full-system CG\n(marker per LM step from step 0; dashed line = "
        "shared cost target reached by all three methods, labels = time each "
        "reaches it; × = budget cut-off)",
        fontsize=13,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "ba_comparison_gpu.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true")
    parser.add_argument(
        "--problems", nargs="+", default=list(PROBLEMS), choices=list(PROBLEMS)
    )
    parser.add_argument(
        "--devices", nargs="+", default=["cpu", "gpu"], choices=["cpu", "gpu"]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--cholmod",
        action="store_true",
        help="CPU CHOLMOD vs Schur+CHOLMOD comparison instead of the full sweep.",
    )
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("JAX devices:", jax.devices())
    for plat in ("cpu", "gpu"):
        d = device_for(plat)
        print(f"  {plat}: {d}")

    for prob_name in args.problems:
        spec = PROBLEMS[prob_name]
        if args.cholmod:
            if not args.replot:
                run_cholmod_study(spec, args.repeats)
            plot_cholmod_comparison(spec.name)
        else:
            if not args.replot:
                run_study(spec, args.devices, args.repeats)
            plot_study(spec.name)


if __name__ == "__main__":
    main()
