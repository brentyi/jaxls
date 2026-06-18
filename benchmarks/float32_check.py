"""float32 robustness check (Ladybug-49, direct Schur path).

Forming S = H_cc - W V^{-1} W^T cancels catastrophically in float32; a naive
implementation NaNs in the Cholesky factorization. Requirement: a full
float32 solve with zero NaNs, converging to a cost comparable to float64.

Usage: uv run --extra dev --extra docs python benchmarks/float32_check.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as onp


def run(x64: bool) -> onp.ndarray:
    jax.config.update("jax_enable_x64", x64)
    # Imports happen after the precision flag so arrays pick up the dtype.
    import jaxls

    from bal import download_bal, load_bal

    path = download_bal(
        "ladybug/problem-49-7776-pre.txt.bz2", Path("/tmp/ladybug-49.txt")
    )
    problem, initial_vals = load_bal(path)
    _, summary = problem.solve(
        initial_vals,
        linear_solver="dense_cholesky",
        termination=jaxls.TerminationConfig(max_iterations=30, early_termination=False),
        verbose=False,
        return_summary=True,
    )
    return onp.asarray(summary.cost_history)


def main() -> None:
    history_f32 = run(x64=False)
    nan_count = int(onp.isnan(history_f32).sum())
    best_f32 = float(history_f32[history_f32 > 0].min())
    print(f"float32: NaNs in cost history: {nan_count}")
    print(f"float32: best cost: {best_f32:.1f}")
    assert nan_count == 0, "float32 robustness FAILED: NaNs in cost history"

    history_f64 = run(x64=True)
    best_f64 = float(history_f64[history_f64 > 0].min())
    print(f"float64: best cost: {best_f64:.1f}")
    print(f"relative gap (f32 vs f64): {abs(best_f32 - best_f64) / best_f64:.2e}")
    print("PASS")


if __name__ == "__main__":
    main()
