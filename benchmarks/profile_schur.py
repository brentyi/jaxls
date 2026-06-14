"""Micro-profile the Schur solve phases to guide optimization.

Breaks one outer LM iteration of the Schur path into its sub-phases and times
each in isolation (jit-compiled, block_until_ready, min of N), so we can see
where the per-iteration cost actually goes:

  prepare_schur          : damping-independent assembly (Gram blocks, W, H_cc)
  damped_vinv            : (V + lambda I)^{-1}, per inner iteration
  reduced_rhs            : b_c - W V^{-1} b_l
  assemble_dense_S       : form the dense reduced matrix S (dense path)
  solve_spd_scaled       : Cholesky-factor + solve S (dense path)
  back_substitute        : recover eliminated update

Run:
    python benchmarks/profile_schur.py --problem ladybug49 --device gpu
"""

from __future__ import annotations

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

from jaxls import _schur  # noqa: E402
from jaxls._solvers import _compute_jacobian_scaler  # noqa: E402

from device_sweep import PROBLEMS  # noqa: E402


def _time(fn, *args, repeats=20):
    """Time a jitted call; return (best milliseconds, output)."""
    f = jax.jit(fn)
    out = jax.block_until_ready(f(*args))
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        best = min(best, time.perf_counter() - t)
    return best * 1e3, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="ladybug49", choices=list(PROBLEMS))
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--solver", default="dense_cholesky")
    args = ap.parse_args()

    dev = jax.devices(args.device)[0]
    print(f"device: {dev}  solver: {args.solver}")

    problem, init = PROBLEMS[args.problem].load(True)
    problem = jax.device_put(problem, dev)
    init = jax.device_put(init, dev)
    plan = problem._elimination
    assert plan is not None
    print(f"reduced_dim={plan.reduced_dim}  tangent_dim={plan.tangent_dim}")

    import jax.numpy as jnp

    # Reproduce A_blocksparse + ATb the way lm_outer_step does. The column
    # scaler is imported from the solver so this profile cannot drift from
    # the production linear system.
    def build_system(vals):
        cost_info = problem._compute_cost_info(vals)
        A = problem._compute_jac_values(vals, cost_info.jac_cache)
        A = A.scale_columns(_compute_jacobian_scaler(A.compute_column_norms()))
        AT_multiply_ = jax.linear_transpose(A.multiply, jnp.zeros((A.shape[1],)))
        ATb = -AT_multiply_(cost_info.residual_vector)[0]
        return A, ATb

    A, ATb = jax.block_until_ready(jax.jit(build_system)(init))

    lambd = jnp.array(1e-3)

    t_prep, factors = _time(
        lambda A, ATb: _schur.prepare_schur(plan, A, ATb, args.solver), A, ATb
    )
    print(f"prepare_schur          {t_prep:8.3f} ms")

    t_vinv, vinv = _time(lambda f, l: _schur._damped_vinv(f, l), factors, lambd)
    print(f"damped_vinv            {t_vinv:8.3f} ms")

    t_rhs, b_red = _time(lambda f, v: _schur._reduced_rhs(f, v), factors, vinv)
    print(f"reduced_rhs            {t_rhs:8.3f} ms")

    if args.solver == "dense_cholesky":
        t_asm, S = _time(
            lambda f, l, v: _schur._assemble_dense_S(f, l, v), factors, lambd, vinv
        )
        print(f"assemble_dense_S       {t_asm:8.3f} ms")

        t_solve, dc = _time(lambda S, b: _schur._solve_spd_scaled(S, b), S, b_red)
        print(f"solve_spd_scaled       {t_solve:8.3f} ms")

        t_back, _ = _time(
            lambda f, v, dc: _schur._back_substitute(f, v, dc), factors, vinv, dc
        )
        print(f"back_substitute        {t_back:8.3f} ms")

        # Full single dense solve, for reference.
        t_full, _ = _time(lambda f, l: _schur.solve_schur_dense(f, l), factors, lambd)
        print(f"--- solve_schur_dense  {t_full:8.3f} ms (full inner step)")
        print(f"--- prepare + 1 inner  {t_prep + t_full:8.3f} ms")


if __name__ == "__main__":
    main()
