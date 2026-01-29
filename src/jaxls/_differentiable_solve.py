"""Differentiable solving via the adjoint/implicit differentiation method.

This module implements backward-mode differentiation through the jaxls solver
using the implicit function theorem. At the optimum x*, the first-order optimality
condition J^T @ r = 0 holds. By differentiating this implicitly, we get:

    dx*/dθ = -(J^T J)^{-1} @ d(J^T r)/dθ

For the VJP (backward pass) with cotangent v = dL/dx*:
1. Solve (J^T J) @ u = v  — one linear solve
2. Compute dL/dθ = -u^T @ d(J^T r)/dθ via JAX autodiff

This is memory-efficient compared to unrolling the optimization loop, requiring
only O(n) storage instead of O(n * iterations).

References:
- "Efficient and Modular Implicit Differentiation" (Blondel et al., 2021)
- "OptNet: Differentiable Optimization as a Layer" (Amos & Kolter, 2017)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import jax
import jax.flatten_util
from jax import numpy as jnp

if TYPE_CHECKING:
    from ._problem import AnalyzedLeastSquaresProblem
    from ._solvers import (
        ConjugateGradientConfig,
        TerminationConfig,
        TrustRegionConfig,
    )
    from ._variables import VarValues

def solve_differentiable(
    problem: "AnalyzedLeastSquaresProblem",
    initial_vals: "VarValues | None" = None,
    *,
    linear_solver: Literal["conjugate_gradient", "dense_cholesky"]
    | "ConjugateGradientConfig" = "conjugate_gradient",
    trust_region: "TrustRegionConfig | None" = None,
    termination: "TerminationConfig | None" = None,
    sparse_mode: Literal["blockrow", "coo", "csr"] = "blockrow",
    verbose: bool = False,
) -> "VarValues":
    """Solve the least squares problem with gradient support for problem parameters.

    This method uses the adjoint/implicit differentiation approach to enable
    backpropagation through the optimization. The backward pass solves one
    linear system instead of unrolling through all iterations.

    Args:
        problem: The analyzed least squares problem.
        initial_vals: Initial values for variables. If None, uses defaults.
        linear_solver: Linear solver to use. Note: "cholmod" is not supported
            for differentiable solve due to pure_callback limitations.
        trust_region: Trust region config. If None, uses Gauss-Newton.
        termination: Termination criteria config.
        sparse_mode: Sparse matrix representation mode.
        verbose: Whether to print optimization progress.

    Returns:
        Optimized variable values with gradient support.

    Raises:
        ValueError: If the problem has constraints (not yet supported).
        ValueError: If "cholmod" linear solver is requested.

    Example:
        >>> @jaxls.Cost.factory
        ... def prior_cost(vals, var, target):
        ...     return vals[var] - target
        >>>
        >>> def loss_fn(target):
        ...     costs = [prior_cost(var, target)]
        ...     problem = jaxls.LeastSquaresProblem(costs, [var]).analyze()
        ...     solution = problem.solve_differentiable()
        ...     return jnp.sum(solution[var] ** 2)
        >>>
        >>> grad = jax.grad(loss_fn)(target)  # Gradient flows through solve
    """
    from ._solvers import (
        ConjugateGradientConfig,
        TerminationConfig,
        TrustRegionConfig,
    )
    from ._variables import VarValues

    # Check for unsupported features
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

    # Set up default configs
    if trust_region is None:
        trust_region = TrustRegionConfig()
    if termination is None:
        termination = TerminationConfig()

    # Create initial values if not provided
    if initial_vals is None:
        initial_vals = VarValues.make(
            var_type(ids)
            for var_type, ids in problem._sorted_ids_from_var_type.items()
        )

    # Prepare flattening/unraveling functions for VarValues
    flat_init, unravel_fn = jax.flatten_util.ravel_pytree(initial_vals)
    del flat_init

    # In our internal API, linear_solver needs to always be a string
    conjugate_gradient_config: ConjugateGradientConfig | None = None
    linear_solver_str: Literal["conjugate_gradient", "dense_cholesky"]
    if isinstance(linear_solver, ConjugateGradientConfig):
        conjugate_gradient_config = linear_solver
        linear_solver_str = "conjugate_gradient"
    else:
        linear_solver_str = linear_solver

    # Create the custom_vjp wrapped solve function with static args captured
    # This avoids passing non-JAX types (strings, etc.) through the custom_vjp
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

    # Call the custom_vjp wrapped function (only JAX pytrees passed)
    solution_flat = solve_with_custom_vjp(problem, initial_vals)

    return unravel_fn(solution_flat)


def _make_differentiable_solve(
    linear_solver: Literal["conjugate_gradient", "dense_cholesky"],
    trust_region: "TrustRegionConfig",
    termination: "TerminationConfig",
    conjugate_gradient_config: "ConjugateGradientConfig | None",
    sparse_mode: Literal["blockrow", "coo", "csr"],
    verbose: bool,
    unravel_fn: Callable[[jax.Array], "VarValues"],
    tangent_ordering: Any,
) -> Callable[["AnalyzedLeastSquaresProblem", "VarValues"], jax.Array]:
    """Create a custom_vjp wrapped solve function with static args captured.

    This factory function captures all static/non-JAX arguments in closures,
    so the returned function only takes JAX pytrees as arguments.
    """
    from ._preconditioning import (
        make_block_jacobi_precoditioner,
        make_point_jacobi_precoditioner,
    )
    from ._solvers import ConjugateGradientConfig, NonlinearSolver

    @jax.custom_vjp
    def solve_impl(
        problem: "AnalyzedLeastSquaresProblem",
        initial_vals: "VarValues",
    ) -> jax.Array:
        """Core differentiable solve with custom VJP."""
        # Create solver and run forward pass
        solver = NonlinearSolver(
            linear_solver=linear_solver,
            trust_region=trust_region,
            termination=termination,
            conjugate_gradient_config=conjugate_gradient_config,
            sparse_mode=sparse_mode,
            verbose=verbose,
            augmented_lagrangian=None,  # Constraints not supported
        )

        solution = solver.solve(problem=problem, initial_vals=initial_vals)

        # Return flattened solution
        flat_solution, _ = jax.flatten_util.ravel_pytree(solution)
        return flat_solution

    def fwd(
        problem: "AnalyzedLeastSquaresProblem",
        initial_vals: "VarValues",
    ) -> tuple[jax.Array, tuple[Any, ...]]:
        """Forward pass: solve and cache values for backward."""
        solution_flat = solve_impl(problem, initial_vals)

        # Cache what we need for backward pass
        # We store the problem and solution, and will recompute Jacobian in backward
        residuals = (problem, solution_flat)

        return solution_flat, residuals

    def bwd(
        residuals: tuple[Any, ...],
        cotangent_flat: jax.Array,
    ) -> tuple[Any, Any]:
        """Backward pass using the adjoint method.

        Given cotangent v = dL/dx* (in flattened form), we compute:
        1. Unravel cotangent to VarValues tangent space
        2. Solve (J^T J) @ u = v using the same linear solver
        3. Compute parameter gradients via VJP of the gradient function
        """
        problem, solution_flat = residuals

        # Unravel solution to VarValues
        solution = unravel_fn(solution_flat)

        # Recompute Jacobian at solution
        cost_info = problem._compute_cost_info(solution)
        A_blocksparse = problem._compute_jac_values(solution, cost_info.jac_cache)

        # Build A @ v and A^T @ v functions
        A_multiply = A_blocksparse.multiply
        AT_multiply_ = jax.linear_transpose(
            A_multiply, jnp.zeros((A_blocksparse.shape[1],))
        )
        AT_multiply = lambda vec: AT_multiply_(vec)[0]

        def ATA_multiply(vec: jax.Array) -> jax.Array:
            return AT_multiply(A_multiply(vec))

        # Map cotangent from flat VarValues to tangent space
        v_tangent = _map_cotangent_to_tangent(
            cotangent_flat, solution, tangent_ordering, unravel_fn, problem._tangent_dim
        )

        # Solve (J^T J) @ u = v
        if linear_solver == "conjugate_gradient":
            cg_config = (
                ConjugateGradientConfig()
                if conjugate_gradient_config is None
                else conjugate_gradient_config
            )

            # Set up preconditioner
            if cg_config.preconditioner == "block_jacobi":
                preconditioner = make_block_jacobi_precoditioner(problem, A_blocksparse)
            elif cg_config.preconditioner == "point_jacobi":
                preconditioner = make_point_jacobi_precoditioner(A_blocksparse)
            else:
                preconditioner = lambda x: x

            # Add small regularization for stability
            def ATA_reg_multiply(vec: jax.Array) -> jax.Array:
                return ATA_multiply(vec) + 1e-8 * vec

            u, _ = jax.scipy.sparse.linalg.cg(
                A=ATA_reg_multiply,
                b=v_tangent,
                x0=jnp.zeros(problem._tangent_dim),
                maxiter=problem._tangent_dim,
                tol=cast(float, cg_config.tolerance_min),
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

        # Compute parameter gradients using VJP of the optimality condition
        # At optimum: g(x*, θ) = J(x*, θ)^T @ r(x*, θ) = 0
        # We need: dL/dθ = -u^T @ (∂g/∂θ)
        def compute_gradient_at_solution(
            problem_inner: "AnalyzedLeastSquaresProblem",
        ) -> jax.Array:
            """Compute J^T @ r at the solution."""
            cost_info_inner = problem_inner._compute_cost_info(solution)
            A_inner = problem_inner._compute_jac_values(
                solution, cost_info_inner.jac_cache
            )
            AT_inner_ = jax.linear_transpose(
                A_inner.multiply, jnp.zeros((A_inner.shape[1],))
            )
            AT_inner = lambda vec: AT_inner_(vec)[0]
            return AT_inner(cost_info_inner.residual_vector)

        # Use VJP to compute parameter gradients
        _, vjp_fn = jax.vjp(compute_gradient_at_solution, problem)
        (problem_grad,) = vjp_fn(-u)

        # Return gradients for problem and initial_vals (None for initial_vals)
        return (problem_grad, None)

    solve_impl.defvjp(fwd, bwd)
    return solve_impl


def _map_cotangent_to_tangent(
    cotangent_flat: jax.Array,
    solution: "VarValues",
    tangent_ordering: Any,
    unravel_fn: Callable[[jax.Array], "VarValues"],
    tangent_dim: int,
) -> jax.Array:
    """Map cotangent from flattened VarValues space to tangent space.

    For Euclidean variables, this is an identity mapping.
    For Lie group variables, we use the VJP of the retraction to compute
    the correct cotangent in tangent space.

    The key insight is: if x_new = retract(x_old, delta), then by VJP:
        d_delta = vjp(retract, x_old)(dx_new)

    We compute this using JAX's autodiff.
    """
    # Compute the VJP of retraction: cotangent in ambient -> cotangent in tangent
    # This uses the identity: d_delta = sum_i (d_x_i * d_retract/d_delta)
    def retract_from_zero(tangent: jax.Array) -> jax.Array:
        """Apply retraction from solution with given tangent, then flatten."""
        retracted = solution._retract(tangent, tangent_ordering)
        flat, _ = jax.flatten_util.ravel_pytree(retracted)
        return flat

    # Compute VJP: given cotangent in flattened VarValues space,
    # get cotangent in tangent space
    _, vjp_fn = jax.vjp(retract_from_zero, jnp.zeros(tangent_dim))
    (tangent_cotangent,) = vjp_fn(cotangent_flat)

    return tangent_cotangent
