"""Workloads: each runs a piece of jaxls and reduces it to `Metric`s.

These are deliberately thin wrappers over the existing harness modules
(`bal`, `device_sweep`, `float32_check`) so there is one implementation of
the measurement kernels, not two. Every workload is a function
`(cfg) -> WorkloadResult` registered in `ALL`.

Timing note: metrics use `bal.run_k_iterations` (warmup + min-of-repeats),
so they are execution times, not compile times. Per-solve budgets keep the
slow full-system baselines from dominating wall-clock; a budget-stopped row
still yields its last cost/time, flagged in the metric notes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import jax

# benchmarks/ on path for the shared harness modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bal  # noqa: E402
import device_sweep as ds  # noqa: E402

from .metrics import Metric, WorkloadResult  # noqa: E402


@dataclass(frozen=True)
class SuiteConfig:
    quick: bool = False
    """Quick tier: GPU only, drop the slow CPU full-system baselines, fewer
    k. For the hill-climbing inner loop."""
    devices: tuple[str, ...] = ("gpu", "cpu")
    repeats: int = 3
    ba_lambda_initial: float = 1e2
    """Uniform damping for the BA matrix (BA curvature scale; see
    results.md). Applied to every method so the comparison is solver-only."""
    solve_budget_s: float = 25.0
    """Skip the rest of a method's k-ladder once one solve exceeds this."""


def _device_list(cfg: SuiteConfig) -> list[str]:
    want = ["gpu"] if cfg.quick else list(cfg.devices)
    return [p for p in want if ds.device_for(p) is not None]


def _subprocess_env() -> dict:
    """Env for subprocess workloads. The parent suite process holds GPU
    memory (and leaves it fragmented after the BA warmups), so a child
    targeting the same GPU OOMs — even with preallocation off — when the
    child is memory-hungry (pyroki's batch-2000 vmap IK). Strategy:

      1. If another GPU is free, pin the child to it (CUDA_VISIBLE_DEVICES).
      2. Otherwise share the parent's GPU but disable preallocation and CUDA
         command buffers (graph capture needs a large contiguous block).

    Falls back gracefully to CPU-only env when no GPU / nvidia-smi.
    """
    import os
    import subprocess

    env = {
        **os.environ,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_FLAGS": (
            os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
        ).strip(),
    }
    # Pick a GPU the parent isn't using, if visible to nvidia-smi.
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        free = [
            line.split(",")[0].strip()
            for line in out.strip().splitlines()
            if int(line.split(",")[1]) < 500  # MiB used -> effectively idle
        ]
        # Don't reuse a GPU the parent already pinned itself to.
        parent = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        free = [g for g in free if g != parent]
        if free:
            env["CUDA_VISIBLE_DEVICES"] = free[0]
    except Exception:  # noqa: BLE001
        pass
    return env


def _run_marker_subprocess(argv: list[str], marker: str, timeout: float):
    """Run argv as a subprocess (with _subprocess_env so it doesn't OOM as a
    co-tenant of the CUDA-warmed parent) and parse JSON printed between
    `marker` and `__END__`. Returns (parsed, None) on success or
    (None, skip_reason) — surfacing the subprocess's real stderr instead of a
    cryptic missing-marker error."""
    import json
    import subprocess

    try:
        p = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, env=_subprocess_env()
        )
        out = p.stdout
        if marker not in out:
            tail = (p.stderr or "").strip().splitlines()[-3:]
            return None, f"subprocess produced no result; stderr: {' | '.join(tail)}"
        seg = out[out.index(marker) + len(marker) : out.index("__END__")]
        return json.loads(seg), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# Bundle adjustment matrix
# --------------------------------------------------------------------------

_BA_METHODS = (
    # (label, linear_solver, elimination)
    ("full_cg", "conjugate_gradient", False),
    ("schur_dense", "dense_cholesky", True),
    ("schur_cg", "conjugate_gradient", True),
)


def bundle_adjustment(cfg: SuiteConfig) -> WorkloadResult:
    res = WorkloadResult(name="bundle_adjustment")
    problems = (
        ["toy", "ladybug49"] if cfg.quick else ["toy", "ladybug49", "trafalgar138"]
    )
    for prob in problems:
        spec = ds.PROBLEMS[prob]
        ks = spec.ks[: 5 if cfg.quick else len(spec.ks)]
        elim, init = spec.load(True)
        full, _ = spec.load(False)
        for platform in _device_list(cfg):
            dev = ds.device_for(platform)
            elim_d = jax.device_put(elim, dev)
            full_d = jax.device_put(full, dev)
            init_d = jax.device_put(init, dev)
            for label, solver, elimination in _BA_METHODS:
                target = elim_d if elimination else full_d
                last_cost = last_t = float("nan")
                stopped = None
                for k in ks:
                    cost, t = bal.run_k_iterations(
                        target,
                        init_d,
                        k,
                        linear_solver=solver,
                        repeats=1 if cfg.quick else cfg.repeats,
                        lambda_initial=cfg.ba_lambda_initial,
                        warmup_budget_s=cfg.solve_budget_s,
                    )
                    last_cost, last_t = cost, t
                    if t > cfg.solve_budget_s:
                        stopped = k
                        break
                base = f"{prob}.{platform}.{label}"
                note = f"budget-stopped at k={stopped}" if stopped else ""
                res.metrics.append(
                    Metric(
                        f"{base}.cost",
                        last_cost,
                        "cost",
                        "lower_better",
                        rel_tol=0.02,
                        notes=note,
                    )
                )
                res.metrics.append(
                    Metric(
                        f"{base}.time",
                        last_t,
                        "s",
                        "lower_better",
                        rel_tol=0.30,
                        notes=note,
                    )
                )
    return res


# --------------------------------------------------------------------------
# float32 robustness (Ladybug, direct Schur path)
# --------------------------------------------------------------------------


def float32_robustness(cfg: SuiteConfig) -> WorkloadResult:
    res = WorkloadResult(name="float32_robustness")
    # float32_check flips the x64 flag at import time, so run it in a
    # subprocess to avoid contaminating the rest of the suite.
    # Run float32_check.run(x64=False) in a subprocess: it flips the global
    # x64 flag at import, so it must not share this process. Reuses the
    # importable module rather than re-spelling the solve recipe.
    root = str(Path(__file__).resolve().parent.parent)
    code = (
        f"import sys; sys.path.insert(0, {root!r})\n"
        "import numpy as onp, json, float32_check\n"
        "ch = float32_check.run(x64=False)\n"
        "print('__F32__' + json.dumps({'nans': int(onp.isnan(ch).sum()),\n"
        "    'best': float(onp.nanmin(ch))}) + '__END__')\n"
    )
    d, skip = _run_marker_subprocess(
        [sys.executable, "-c", code], "__F32__", timeout=300
    )
    if skip is not None:
        res.skipped = skip
        return res
    assert d is not None  # skip is None => parsed result present
    res.metrics.append(
        Metric(
            "float32.ladybug49.nan_count",
            float(d["nans"]),
            "",
            "lower_better",
            rel_tol=0.0,
            abs_tol=0.0,
        )
    )
    res.metrics.append(
        Metric(
            "float32.ladybug49.best_cost",
            float(d["best"]),
            "cost",
            "lower_better",
            rel_tol=0.05,
            notes="float32 direct Schur path",
        )
    )
    return res


# --------------------------------------------------------------------------
# pyroki IK-Beam downstream (optional; needs pyroki + the A/B script)
# --------------------------------------------------------------------------


def pyroki_ik(cfg: SuiteConfig) -> WorkloadResult:
    res = WorkloadResult(name="pyroki_ik")
    try:
        import pyroki  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        res.skipped = "pyroki not installed (pip install the pyroki repo)"
        return res
    script = Path(__file__).resolve().parent / "pyroki_ik.py"
    if not script.exists():
        res.skipped = "benchmarks/suite/pyroki_ik.py missing"
        return res
    batches = ["100", "1000"] if cfg.quick else ["100", "1000", "2000"]
    rows, skip = _run_marker_subprocess(
        [sys.executable, str(script), "--batch-sizes", *batches],
        "__PYROKI__",
        timeout=600,
    )
    if skip is not None:
        res.skipped = skip
        return res
    assert rows is not None  # skip is None => parsed result present
    for r in rows:
        b = r["batch"]
        res.metrics.append(
            Metric(
                f"pyroki.b{b}.success_pct",
                r["succ_pct"],
                "%",
                "higher_better",
                rel_tol=0.02,
            )
        )
        res.metrics.append(
            Metric(
                f"pyroki.b{b}.p99_pos_err_mm",
                r["pos_err_mm_p99"],
                "mm",
                "lower_better",
                rel_tol=0.25,
            )
        )
        res.metrics.append(
            Metric(
                f"pyroki.b{b}.time_ms", r["time_ms"], "ms", "lower_better", rel_tol=0.30
            )
        )
    return res


# --------------------------------------------------------------------------
# Example notebooks (convergence to a relative cost floor)
# --------------------------------------------------------------------------


def example_notebooks(cfg: SuiteConfig) -> WorkloadResult:
    """Reduce each notebook's headline solve to (final cost, steps, time)
    via the trace harness. Skipped in --quick (notebook execution is slow)."""
    res = WorkloadResult(name="example_notebooks")
    if cfg.quick:
        res.skipped = "skipped in --quick (notebook execution is slow)"
        return res
    import trace_examples as tx

    traces = tx.trace_single(src="src", timeout=300.0)
    for name, trace in traces.items():
        if trace.get("iterations", 0) <= 0:
            continue
        res.metrics.append(
            Metric(
                f"example.{name}.final_cost",
                float(min(trace["cost_history"])),
                "cost",
                "lower_better",
                rel_tol=0.05,
            )
        )
        res.metrics.append(
            Metric(
                f"example.{name}.steps",
                float(trace["iterations"]),
                "steps",
                "lower_better",
                rel_tol=0.20,
            )
        )
    return res


ALL = {
    "bundle_adjustment": bundle_adjustment,
    "float32_robustness": float32_robustness,
    "pyroki_ik": pyroki_ik,
    "example_notebooks": example_notebooks,
}

# Default set for a no-argument run / the regression gate: the fast,
# in-process-reliable workloads. Two workloads are opt-in (run via
# `--only`), kept out of the gate so it stays deterministic:
#   - example_notebooks: 19 jupyter subprocesses, ~minutes.
#   - pyroki_ik: needs pyroki installed, and its memory-hungry batched IK is
#     unreliable as a *co-tenant* subprocess of a CUDA-warmed parent (cuSolver
#     init / command-buffer OOM); it runs fine standalone (`--only pyroki_ik`)
#     and is the canonical downstream-regression check.
DEFAULT = ("bundle_adjustment", "float32_robustness")
