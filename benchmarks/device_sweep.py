"""Cross-device benchmark: linear-solver methods on CPU vs GPU.

Extends the matched-iteration methodology of `matched_iters.py` to run the
same study on both CPU and GPU, across a sweep of problem sizes, comparing
every available linear solver:

  - full CG       : conjugate gradient on the full system (no elimination)
  - full dense    : dense Cholesky on the full system (small problems only)
  - cholmod       : sparse Cholesky on the full system (CPU only)
  - Schur + dense : variable elimination, dense reduced solve
  - Schur + CG    : variable elimination, matrix-free reduced CG

Methodology (same as matched_iters.py): run exactly k Levenberg-Marquardt
iterations with early termination off, for a sweep of k, and record
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
    Method(
        "full dense",
        "dense_cholesky",
        elimination=False,
        full_dense_only=True,
    ),
)


# ---------------------------------------------------------------------------
# Timing — the measurement kernel itself is bal.run_k_iterations, shared with
# matched_iters.py so the two scripts cannot drift apart methodologically.
# ---------------------------------------------------------------------------


def device_for(platform: str) -> jax.Device | None:
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COLOR = {
    "full CG": "#d62728",
    "full dense": "#7f7f7f",
    "cholmod": "#9467bd",
    "Schur + dense": "#1f77b4",
    "Schur + CG": "#2ca02c",
}


def plot_study(name: str) -> None:
    """One figure per device: cost vs LM steps (left) and cost vs
    wall-clock (right). Prefers the tuned PR/main pair of JSONs (main's
    methods drawn dashed); falls back to the plain sweep JSON."""
    import matplotlib.pyplot as plt

    sources = []  # (style label prefix, linestyle, data)
    tuned_final = RESULTS_DIR / f"device_{name}_tuned_final.json"
    if tuned_final.exists():
        sources.append(("", "-", json.loads(tuned_final.read_text())))
        tuned_main = RESULTS_DIR / f"device_{name}_tuned_main.json"
        if tuned_main.exists():
            sources.append(("main: ", "--", json.loads(tuned_main.read_text())))
    else:
        sources.append(
            ("", "-", json.loads((RESULTS_DIR / f"device_{name}.json").read_text()))
        )
    title = sources[0][2]["title"]

    for device in ("cpu", "gpu"):
        fig, (ax_steps, ax_time) = plt.subplots(1, 2, figsize=(12, 4.5))
        plotted = False
        for prefix, ls, data in sources:
            for run in data["runs"].values():
                if run["device"] != device:
                    continue
                plotted = True
                color = _COLOR[run["method"]]
                label = prefix + run["method"]
                run_ks = run.get("ks", data["ks"])
                ax_steps.plot(
                    run_ks,
                    run["costs"],
                    marker="o",
                    ms=3,
                    ls=ls,
                    color=color,
                    label=label,
                )
                ax_time.plot(
                    run["times"],
                    run["costs"],
                    marker="o",
                    ms=3,
                    ls=ls,
                    color=color,
                    label=label,
                )
        if not plotted:
            plt.close(fig)
            continue
        for ax, xlabel, xlog in (
            (ax_steps, "outer LM steps", False),
            (ax_time, "wall-clock (s)", True),
        ):
            ax.set_yscale("log")
            if xlog:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("accepted cost (log)")
        ax_steps.legend(fontsize=8)
        fig.suptitle(f"{title} — {device.upper()}")
        fig.tight_layout()
        out = RESULTS_DIR / f"device_{name}_{device}.png"
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

        # Baseline target: the converged cost the main full-CG run reaches.
        # Speedups are "time for each method to reach that same cost" — the
        # apples-to-apples question a reader actually cares about. A small
        # tolerance absorbs matched-k jitter so a method that lands ~0.1%
        # above the baseline floor still counts as having reached it.
        base_data = _load("main")
        base_run = base_data["runs"].get("gpu:full CG") if base_data else None
        target = min(base_run["costs"]) * 1.003 if base_run else None
        t_base = _time_to_cost(base_run, c0, target) if base_run else None

        for label, suffix, method, color, marker in series:
            data = _load(suffix)
            run = data["runs"].get(f"gpu:{method}") if data else None
            if run is None:
                continue
            times, best = _running_best(run, c0)
            speed = ""
            t_hit = _time_to_cost(run, c0, target) if target is not None else None
            if t_hit is not None and t_base is not None and t_hit > 0:
                factor = t_base / t_hit
                speed = f"  ({factor:.0f}× faster)" if factor >= 1.5 else ""
            ax.plot(
                times, best, marker=marker, ms=5, color=color, label=label + speed
            )
            if run.get("budget_stopped_at_k") is not None:
                ax.plot(times[-1], best[-1], marker="x", ms=11, mew=2, color=color)
        if target is not None:
            ax.axhline(
                target / 1.003,
                color="0.6",
                lw=1,
                ls=":",
                zorder=0,
                label="baseline converged cost",
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
        "vs full-system CG\n(marker per LM step from step 0; speedup = time "
        "to reach the baseline's converged cost; × = budget cut-off)",
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
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("JAX devices:", jax.devices())
    for plat in ("cpu", "gpu"):
        d = device_for(plat)
        print(f"  {plat}: {d}")

    for prob_name in args.problems:
        spec = PROBLEMS[prob_name]
        if not args.replot:
            run_study(spec, args.devices, args.repeats)
        plot_study(spec.name)


if __name__ == "__main__":
    main()
