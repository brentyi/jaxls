from __future__ import annotations

from typing import Any

import jax
import jax.flatten_util
from jax import numpy as jnp


@jax.custom_vjp
def _solve_trivial(target: Any) -> Any:
    return target


def _solve_trivial_fwd(target: Any) -> Any:
    return target, target


def _solve_trivial_bwd(target: Any, cotangent: Any) -> Any:
    return (cotangent,)


_solve_trivial.defvjp(_solve_trivial_fwd, _solve_trivial_bwd)


@jax.custom_vjp
def _solve_overdetermined(A: Any, b: Any) -> Any:
    x, _, _, _ = jnp.linalg.lstsq(A, b, rcond=None)
    return x


def _solve_overdetermined_fwd(A: Any, b: Any) -> Any:
    x = _solve_overdetermined(A, b)
    return x, (A, b, x)


def _solve_overdetermined_bwd(residuals: Any, cotangent: Any) -> Any:
    A, b, x = residuals
    v = cotangent

    ATA = A.T @ A

    ATA = ATA + 1e-8 * jnp.eye(ATA.shape[0])
    u = jnp.linalg.solve(ATA, v)

    grad_b = A @ u

    residual = A @ x - b
    grad_A = -jnp.outer(residual, u) - jnp.outer(A @ u, x)

    return grad_A, grad_b


_solve_overdetermined.defvjp(_solve_overdetermined_fwd, _solve_overdetermined_bwd)


@jax.custom_vjp
def _solve_nonlinear_iterative(target: Any) -> Any:
    x = jnp.sqrt(jnp.maximum(target, 0.1))

    def gn_step(x: Any) -> Any:
        residual = x**2 - target

        delta = residual / (2 * x + 1e-8)
        return x - delta

    for _ in range(20):
        x = gn_step(x)

    return x


def _solve_nonlinear_iterative_fwd(
    target: Any,
) -> Any:
    x = _solve_nonlinear_iterative(target)
    return x, (target, x)


def _solve_nonlinear_iterative_bwd(residuals: Any, cotangent: Any) -> Any:
    _target, x = residuals
    v = cotangent

    grad_target = v / (2 * x + 1e-8)

    return (grad_target,)


_solve_nonlinear_iterative.defvjp(
    _solve_nonlinear_iterative_fwd, _solve_nonlinear_iterative_bwd
)


def solve_differentiable(
    problem: Any,
    initial_vals: Any = None,
    *,
    linear_solver: Any = "conjugate_gradient",
    trust_region: Any = None,
    termination: Any = None,
    sparse_mode: Any = "blockrow",
    verbose: Any = False,
) -> Any:
    from ._solvers import (
        ConjugateGradientConfig,
        TerminationConfig,
        TrustRegionConfig,
    )
    from ._variables import VarValues

    has_constraints = any(cost.kind != "l2_squared" for cost in problem._stacked_costs)
    if has_constraints:
        raise ValueError(
            "Differentiable solve does not support constraints. "
            "Use solve() with augmented_lagrangian for constrained problems."
        )

    if linear_solver == "cholmod":
        raise ValueError(
            "CHOLMOD linear solver is not supported for differentiable solve "
            "due to pure_callback limitations. Use 'conjugate_gradient' or "
            "'dense_cholesky' instead."
        )

    if trust_region is None:
        trust_region = TrustRegionConfig()
    if termination is None:
        termination = TerminationConfig()

    if initial_vals is None:
        initial_vals = VarValues.make(
            var_type(ids) for var_type, ids in problem._sorted_ids_from_var_type.items()
        )

    flat_init, unravel_fn = jax.flatten_util.ravel_pytree(initial_vals)
    del flat_init

    conjugate_gradient_config: Any = None
    linear_solver_str: Any
    if isinstance(linear_solver, ConjugateGradientConfig):
        conjugate_gradient_config = linear_solver
        linear_solver_str = "conjugate_gradient"
    else:
        linear_solver_str = linear_solver

    solve_with_custom_vjp = _make_differentiable_solve(
        linear_solver_str,
        trust_region,
        termination,
        conjugate_gradient_config,
        sparse_mode,
        verbose,
        unravel_fn,
        problem._tangent_ordering,
    )

    solution_flat = solve_with_custom_vjp(problem, initial_vals)

    return unravel_fn(solution_flat)


def _make_differentiable_solve(
    linear_solver: Any,
    trust_region: Any,
    termination: Any,
    conjugate_gradient_config: Any,
    sparse_mode: Any,
    verbose: Any,
    unravel_fn: Any,
    tangent_ordering: Any,
) -> Any:
    from ._preconditioning import (
        make_block_jacobi_precoditioner,
        make_point_jacobi_precoditioner,
    )
    from ._solvers import ConjugateGradientConfig, NonlinearSolver

    @jax.custom_vjp
    def solve_impl(
        problem: Any,
        initial_vals: Any,
    ) -> Any:
        solver = NonlinearSolver(
            linear_solver=linear_solver,
            trust_region=trust_region,
            termination=termination,
            conjugate_gradient_config=conjugate_gradient_config,
            sparse_mode=sparse_mode,
            verbose=verbose,
            augmented_lagrangian=None,
        )

        solution = solver.solve(problem=problem, initial_vals=initial_vals)

        flat_solution, _ = jax.flatten_util.ravel_pytree(solution)
        return flat_solution

    def fwd(
        problem: Any,
        initial_vals: Any,
    ) -> Any:
        solution_flat = solve_impl(problem, initial_vals)

        residuals = (problem, solution_flat)

        return solution_flat, residuals

    def bwd(
        residuals: Any,
        cotangent_flat: Any,
    ) -> Any:
        problem, solution_flat = residuals

        solution = unravel_fn(solution_flat)

        cost_info = problem._compute_cost_info(solution)
        A_blocksparse = problem._compute_jac_values(solution, cost_info.jac_cache)

        A_multiply = A_blocksparse.multiply
        AT_multiply_ = jax.linear_transpose(
            A_multiply, jnp.zeros((A_blocksparse.shape[1],))
        )
        AT_multiply = lambda vec: AT_multiply_(vec)[0]

        def ATA_multiply(vec: Any) -> Any:
            return AT_multiply(A_multiply(vec))

        v_tangent = _map_cotangent_to_tangent(
            cotangent_flat, solution, tangent_ordering, unravel_fn, problem._tangent_dim
        )

        if linear_solver == "conjugate_gradient":
            cg_config = (
                ConjugateGradientConfig()
                if conjugate_gradient_config is None
                else conjugate_gradient_config
            )

            if cg_config.preconditioner == "block_jacobi":
                preconditioner = make_block_jacobi_precoditioner(problem, A_blocksparse)
            elif cg_config.preconditioner == "point_jacobi":
                preconditioner = make_point_jacobi_precoditioner(A_blocksparse)
            else:
                preconditioner = lambda x: x

            def ATA_reg_multiply(vec: Any) -> Any:
                return ATA_multiply(vec) + 1e-8 * vec

            u, _ = jax.scipy.sparse.linalg.cg(
                A=ATA_reg_multiply,
                b=v_tangent,
                x0=jnp.zeros(problem._tangent_dim),
                maxiter=problem._tangent_dim,
                tol=cg_config.tolerance_min,
                M=preconditioner,
            )
        elif linear_solver == "dense_cholesky":
            A_dense = A_blocksparse.to_dense()
            ATA = A_dense.T @ A_dense
            ATA = ATA + 1e-8 * jnp.eye(problem._tangent_dim)
            cho_factor = jax.scipy.linalg.cho_factor(ATA)
            u = jax.scipy.linalg.cho_solve(cho_factor, v_tangent)
        else:
            raise ValueError(f"Unknown linear solver: {linear_solver}")

        def compute_gradient_at_solution(
            problem_inner: Any,
        ) -> Any:
            cost_info_inner = problem_inner._compute_cost_info(solution)
            A_inner = problem_inner._compute_jac_values(
                solution, cost_info_inner.jac_cache
            )
            AT_inner_ = jax.linear_transpose(
                A_inner.multiply, jnp.zeros((A_inner.shape[1],))
            )
            AT_inner = lambda vec: AT_inner_(vec)[0]
            return AT_inner(cost_info_inner.residual_vector)

        _, vjp_fn = jax.vjp(compute_gradient_at_solution, problem)
        (problem_grad,) = vjp_fn(-u)

        return (problem_grad, None)

    solve_impl.defvjp(fwd, bwd)
    return solve_impl


def _map_cotangent_to_tangent(
    cotangent_flat: Any,
    solution: Any,
    tangent_ordering: Any,
    unravel_fn: Any,
    tangent_dim: Any,
) -> Any:
    def retract_from_zero(tangent: Any) -> Any:
        retracted = solution._retract(tangent, tangent_ordering)
        flat, _ = jax.flatten_util.ravel_pytree(retracted)
        return flat

    _, vjp_fn = jax.vjp(retract_from_zero, jnp.zeros(tangent_dim))
    (tangent_cotangent,) = vjp_fn(cotangent_flat)

    return tangent_cotangent
