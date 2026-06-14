"""Per-iteration cost-vs-time traces for the example notebooks.

Uses the in-solve timestamps now recorded in `SolveSummary.time_history`
(host callback per outer LM step), so a single solve yields a full
convergence trace — no matched-k re-solving. Every top-level solve in a
notebook is instrumented and the one with the largest relative cost drop is
kept (the headline optimization, not a warmup/forward/demo solve); the
captured (cost_history, time_history, iterations) is dumped as JSON, then
plotted as a grid of cost-vs-time subplots (one per notebook), running-best
cost with a marker per LM step from step 0.

    python benchmarks/trace_examples.py            # run + plot
    python benchmarks/trace_examples.py --plot     # plot from saved JSON
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
# Both configs carry SolveSummary.time_history (main is the
# instrumentation-timestamps branch off main), so per-step times are real on
# both sides. A warmup solve absorbs compilation before the timed one.
INSTRUMENT = """
import jax, jaxls, jaxls._problem, json as _json
import numpy as _onp
_traces = []
_orig = jaxls._problem.AnalyzedLeastSquaresProblem.solve

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
    # Turn on per-iteration timestamps (opt-in on TerminationConfig).
    import dataclasses as _dc
    _term = k.get("termination", None) or jaxls.TerminationConfig()
    k["termination"] = _dc.replace(_term, record_time_history=True)
    jax.block_until_ready(_orig(self, *a, **k))           # warmup / compile
    sol, summ = _orig(self, *a, **k)
    jax.block_until_ready((sol, summ))
    n = int(summ.iterations)
    ch = _onp.asarray(summ.cost_history)[: n + 1]
    th = _onp.asarray(summ.time_history)[: n + 1]
    _traces.append({
        "iterations": n,
        "cost_history": ch.tolist(),
        "elapsed_s": (th - th[0]).tolist(),
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
    nbs = sorted(
        p for p in EXAMPLES_DIR.rglob("*.ipynb") if ".ipynb_checkpoints" not in str(p)
    )
    for i, nb in enumerate(nbs):
        print(f"[{i + 1}/{len(nbs)}] {nb.stem}", flush=True)
        rec = run_one(nb, src, timeout)
        if rec and rec["iterations"] > 0:
            out[nb.stem] = rec
    return out


def run_all(timeout: float) -> dict:
    traces: dict = {}
    nbs = sorted(
        p for p in EXAMPLES_DIR.rglob("*.ipynb") if ".ipynb_checkpoints" not in str(p)
    )
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


def plot(traces: dict) -> None:
    import math

    import matplotlib.pyplot as plt

    style = {"main": ("#d62728", "o", "--"), "PR": ("#1f77b4", "^", "-")}
    names = sorted(traces)
    ncol = 4
    nrow = math.ceil(len(names) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, names):
        ax.set_visible(True)
        for label, (color, marker, ls) in style.items():
            rec = traces[name].get(label)
            if rec is None:
                continue
            # Running-best: cost_history records each step's proposed cost, so
            # a rejected step can bump it up; cummin shows "best so far".
            best = list(rec["cost_history"])
            for i in range(1, len(best)):
                best[i] = min(best[i], best[i - 1])
            ax.plot(
                rec["elapsed_s"],
                best,
                marker=marker,
                ms=4,
                ls=ls,
                color=color,
                label=label,
            )
        ax.set_yscale("log")
        ax.set_xscale("symlog", linthresh=1e-2)
        ax.set_xlim(left=0)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("wall-clock (s)", fontsize=8)
        ax.set_ylabel("cost", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    fig.suptitle(
        "jaxls examples: cost vs wall-clock per LM step — main (dashed) vs PR "
        "(solid); real in-solve timestamps on both."
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
