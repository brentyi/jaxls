"""Per-iteration cost-vs-time traces for the example notebooks.

Uses the in-solve timestamps recorded via `jaxls.record_iteration_times()`
(host callback per outer LM step), so a single solve yields a full
convergence trace — no matched-k re-solving. Every top-level solve in a
notebook is instrumented and the one with the largest relative cost drop is
kept (the headline optimization, not a warmup/forward/demo solve); the
captured (cost_history, elapsed_s, iterations) is dumped as JSON, then
plotted as a grid of cost-vs-time subplots (one per notebook), running-best
cost with a marker per LM step from step 0.

    uv run --extra dev --extra docs python benchmarks/trace_examples.py            # run + plot
    uv run --extra dev --extra docs python benchmarks/trace_examples.py --plot     # plot from saved JSON
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "docs/source/examples"
RESULTS = Path(__file__).parent / "results"
TRACE_JSON = RESULTS / "example_traces.json"

# Instrumentation: capture EVERY top-level solve, then (at finalize) emit
# the one with the largest cost reduction — the notebook's headline
# optimization, not an incidental warmup/forward/demo solve.
#
# Per-step timestamps come from `jaxls.record_iteration_times()` (host
# float64 list, off the jitted path). The `main` config predates that API
# and instead exposes `SolveSummary.time_history`; we feature-detect and use
# whichever is present so both sides yield real per-step times. A warmup
# solve absorbs compilation before the timed one.
INSTRUMENT = """
import jax, jaxls, jaxls._problem, json as _json
import numpy as _onp
_traces = []
_orig = jaxls._problem.AnalyzedLeastSquaresProblem.solve
_HAS_RECORDER = hasattr(jaxls, "record_iteration_times")

def _traced(self, *a, **k):
    import jax.core
    def _tr(x, s=None):
        s = set() if s is None else s
        if id(x) in s: return False
        s.add(id(x))
        if isinstance(x, jax.core.Tracer): return True
        return any(_tr(v, s) for v in getattr(x, "__dict__", {}).values())
    if _tr(self) or _tr(a) or _tr(k):
        return _orig(self, *a, **k)
    k = dict(k); want = k.get("return_summary", False); k["return_summary"] = True
    if _HAS_RECORDER:
        with jaxls.record_iteration_times() as _times:
            jax.block_until_ready(_orig(self, *a, **k))    # warmup / compile
            _times.clear()
            sol, summ = _orig(self, *a, **k)
            jax.block_until_ready((sol, summ))
        elapsed = [t - _times[0] for t in _times]
    else:
        # Legacy main: timestamps always recorded in SolveSummary.time_history.
        jax.block_until_ready(_orig(self, *a, **k))       # warmup / compile
        sol, summ = _orig(self, *a, **k)
        jax.block_until_ready((sol, summ))
        th = _onp.asarray(summ.time_history)
        elapsed = (th - th[0]).tolist()
    n = int(summ.iterations)
    # cost_history holds init at [0] + one slot per step but is only
    # max_iterations long, so it caps at init + (max_iterations-1) steps; the
    # timestamp recorder has no such cap. Align both to the shorter length.
    ch = _onp.asarray(summ.cost_history)[: n + 1].tolist()
    m = min(len(ch), len(elapsed))
    _traces.append({
        "iterations": n,
        "cost_history": ch[:m],
        "elapsed_s": elapsed[:m],
    })
    return (sol, summ) if want else sol

jaxls._problem.AnalyzedLeastSquaresProblem.solve = _traced
"""

FINALIZE = """
import json as _json, math as _math
def _drop(r):
    c = r["cost_history"]
    if len(c) < 2 or c[0] <= 0 or not _math.isfinite(c[0]):
        return -1.0
    return (c[0] - min(c)) / c[0]   # relative reduction
_best = max(_traces, key=_drop, default=None) if _traces else None
if _best is not None and _drop(_best) > 0:
    print("__TRACE__" + _json.dumps(_best) + "__END__", flush=True)
"""


# jaxls under test per label; PYTHONPATH for the notebook subprocess.
CONFIGS = {"main": "/tmp/jaxls-main/src", "PR": "src"}


def _list_notebooks() -> list[Path]:
    """Source notebooks under the examples tree. Excludes checkpoints and the
    `tmp*.ipynb` instrumented copies this script writes (a crash/kill can skip
    their cleanup, and we must not re-trace those stragglers next run)."""
    return sorted(
        p
        for p in EXAMPLES_DIR.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in str(p) and not p.name.startswith("tmp")
    )


def run_one(nb_path: Path, src: str, timeout: float) -> dict | None:
    import nbformat

    nb = nbformat.read(open(nb_path), as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell(source=INSTRUMENT))
    nb.cells.append(nbformat.v4.new_code_cell(source=FINALIZE))
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ipynb", delete=False, dir=nb_path.parent
    ) as f:
        nbformat.write(nb, f)
        tmp = f.name
    try:
        p = subprocess.run(
            [sys.executable, "-m", "jupyter", "execute", tmp, "--inplace"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(EXAMPLES_DIR.parent.parent.parent),
            # PREALLOCATE=false: when run under the benchmark suite, the parent
            # process already holds most of the GPU; without this the notebook
            # subprocess OOMs.
            env={
                **__import__("os").environ,
                "PYTHONPATH": src,
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            },
        )
        ex = nbformat.read(open(tmp), as_version=4)
        for cell in ex.cells:
            for out in cell.get("outputs", []):
                txt = out.get("text", "")
                if "__TRACE__" in txt:
                    s = txt.index("__TRACE__") + len("__TRACE__")
                    e = txt.index("__END__", s)
                    return json.loads(txt[s:e])
        if p.returncode != 0:
            print(f"  {nb_path.stem}: exit {p.returncode}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  {nb_path.stem}: timeout")
        return None
    finally:
        Path(tmp).unlink(missing_ok=True)


def trace_single(src: str = "src", timeout: float = 300.0) -> dict:
    """Trace every notebook under one jaxls (PYTHONPATH=`src`). Returns
    {name: trace_rec}. Used by the benchmark suite for single-config metrics
    (the A/B `run_all` traces two configs)."""
    out: dict = {}
    nbs = _list_notebooks()
    for i, nb in enumerate(nbs):
        print(f"[{i + 1}/{len(nbs)}] {nb.stem}", flush=True)
        rec = run_one(nb, src, timeout)
        if rec and rec["iterations"] > 0:
            out[nb.stem] = rec
    return out


def run_all(timeout: float) -> dict:
    traces: dict = {}
    nbs = _list_notebooks()
    for label, src in CONFIGS.items():
        print(f"\n=== config: {label} ({src}) ===", flush=True)
        for i, nb in enumerate(nbs):
            name = nb.stem
            print(f"[{label} {i + 1}/{len(nbs)}] {name}", flush=True)
            rec = run_one(nb, src, timeout)
            if rec and rec["iterations"] > 0:
                traces.setdefault(name, {})[label] = rec
                print(
                    f"  {rec['iterations']} steps, "
                    f"cost {rec['cost_history'][0]:.4g} -> {rec['cost_history'][-1]:.4g}, "
                    f"{rec['elapsed_s'][-1]:.3f}s"
                )
    TRACE_JSON.write_text(json.dumps(traces, indent=2))
    print(f"wrote {TRACE_JSON}")
    return traces


def _running_best(costs: list[float]) -> list[float]:
    # cost_history records each step's proposed cost, so a rejected step can
    # bump it up; cummin shows the best solution found so far.
    best = list(costs)
    for i in range(1, len(best)):
        best[i] = min(best[i], best[i - 1])
    return best


def _speedup_label(traces_for_name: dict) -> str:
    """PR-vs-main speedup: ratio of wall-clock to reach main's final cost.
    Empty string unless both configs are present and the ratio is meaningful."""
    main, pr = traces_for_name.get("main"), traces_for_name.get("PR")
    if main is None or pr is None:
        return ""
    target = min(main["cost_history"]) * 1.003
    t_main = main["elapsed_s"][-1]

    def time_to(rec: dict) -> float | None:
        for t, b in zip(rec["elapsed_s"], _running_best(rec["cost_history"])):
            if b <= target:
                return t
        return None

    t_pr = time_to(pr)
    if t_pr is None or t_pr <= 0 or t_main <= 0:
        return ""
    factor = t_main / t_pr
    return f"  ({factor:.1f}× faster)" if factor >= 1.2 else ""


def plot(traces: dict) -> None:
    import math

    import matplotlib.pyplot as plt

    style = {"main": ("#d62728", "o", "--"), "PR": ("#1f77b4", "^", "-")}
    names = sorted(traces)
    ncol = 4
    nrow = math.ceil(len(names) / ncol)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.5 * ncol, 3.4 * nrow), squeeze=False
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, names):
        ax.set_visible(True)
        for label, (color, marker, ls) in style.items():
            rec = traces[name].get(label)
            if rec is None:
                continue
            best = _running_best(rec["cost_history"])
            ax.plot(
                rec["elapsed_s"],
                best,
                marker=marker,
                ms=5,
                ls=ls,
                color=color,
                label=f"{label} ({rec['iterations']} steps)",
            )
        ax.set_yscale("log")
        ax.set_xscale("symlog", linthresh=1e-2)
        ax.set_xlim(left=0)
        ax.set_title(name + _speedup_label(traces[name]), fontsize=11)
        ax.set_xlabel("wall-clock (s)", fontsize=10)
        ax.set_ylabel("cost", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(
        "jaxls examples: cost vs wall-clock per LM step — main (dashed) vs PR "
        "(solid)\nreal in-solve timestamps on both; speedup = time to reach "
        "main's final cost",
        fontsize=14,
    )
    fig.tight_layout()
    out = RESULTS / "example_traces.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()
    traces = json.loads(TRACE_JSON.read_text()) if args.plot else run_all(args.timeout)
    plot(traces)


if __name__ == "__main__":
    main()
