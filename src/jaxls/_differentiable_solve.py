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

# =============================================================================
# Step 1: Trivial case - validate custom_vjp mechanics
# =============================================================================


@jax.custom_vjp
def _solve_trivial(target: jax.Array) -> jax.Array:
    """Trivial 'solve' for min_x ||x - target||^2.

    Solution: x* = target, Jacobian: J = I (identity).
    """
    return target


def _solve_trivial_fwd(target: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Forward pass for trivial solve."""
    return target, target  # (result, residuals for backward)


def _solve_trivial_bwd(target: jax.Array, cotangent: jax.Array) -> tuple[jax.Array]:
    """Backward pass for trivial solve.

    Since x* = target and J = I, we have:
    - (J^T J)^{-1} = I
    - dx*/dtarget = I

    So the VJP is just: dL/dtarget = dL/dx* @ I = cotangent
    """
    return (cotangent,)


_solve_trivial.defvjp(_solve_trivial_fwd, _solve_trivial_bwd)


# =============================================================================
# Step 2: Over-determined system - validate (J^T J)^{-1} solve pattern
# =============================================================================


@jax.custom_vjp
def _solve_overdetermined(A: jax.Array, b: jax.Array) -> jax.Array:
    """Solve min_x ||Ax - b||^2 for overdetermined system (A is tall).

    Solution: x* = (A^T A)^{-1} A^T b
    """
    # Use lstsq for numerical stability
    x, _, _, _ = jnp.linalg.lstsq(A, b, rcond=None)
    return x


def _solve_overdetermined_fwd(
    A: jax.Array, b: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """Forward pass for overdetermined solve."""
    x = _solve_overdetermined(A, b)
    return x, (A, b, x)  # Cache for backward pass


def _solve_overdetermined_bwd(
    residuals: tuple[jax.Array, jax.Array, jax.Array], cotangent: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Backward pass for overdetermined solve.

    For x* = (A^T A)^{-1} A^T b, the VJP computes:
    - dL/db given dL/dx* (cotangent)
    - dL/dA given dL/dx* (cotangent)

    Using implicit differentiation:
    1. Solve (A^T A) u = cotangent for u
    2. dL/db = A @ u
    3. dL/dA = -outer(A @ u, u) - outer(b - A @ x*, u_reshaped)
              where u_reshaped accounts for the A^T b term
    """
    A, b, x = residuals
    v = cotangent  # dL/dx*

    # Solve (A^T A) @ u = v
    # This is the key adjoint equation
    ATA = A.T @ A
    # Add small regularization for numerical stability
    ATA = ATA + 1e-8 * jnp.eye(ATA.shape[0])
    u = jnp.linalg.solve(ATA, v)

    # Gradient w.r.t. b: dL/db = A @ u
    grad_b = A @ u

    # Gradient w.r.t. A: uses the chain rule through both A^T A and A^T b
    # d(A^T A)^{-1}/dA and d(A^T b)/dA terms
    # Simplified: dL/dA = -(A @ x* - b) @ u^T - A @ u @ x*^T
    # But more directly, using the implicit function:
    residual = A @ x - b  # This should be small at optimum
    grad_A = -jnp.outer(residual, u) - jnp.outer(A @ u, x)

    return grad_A, grad_b


_solve_overdetermined.defvjp(_solve_overdetermined_fwd, _solve_overdetermined_bwd)


# =============================================================================
# Step 3: Iterative nonlinear solve with adjoint backward pass
# =============================================================================


@jax.custom_vjp
def _solve_nonlinear_iterative(target: jax.Array) -> jax.Array:
    """Solve min_x ||x^2 - target||^2 using Gauss-Newton iterations.

    This demonstrates that the adjoint method works correctly even when
    the forward pass uses iterative optimization.
    """
    # Initialize with sqrt of target (good initial guess)
    x = jnp.sqrt(jnp.maximum(target, 0.1))

    # Gauss-Newton iterations for f(x) = x^2
    # Jacobian of f: J = diag(2*x)
    # Update: x_new = x - (J^T J)^{-1} J^T (f(x) - target)
    #       = x - (4*x^2)^{-1} * 2*x * (x^2 - target)
    #       = x - (x^2 - target) / (2*x)

    def gn_step(x: jax.Array) -> jax.Array:
        residual = x**2 - target
        # J = diag(2*x), so (J^T J)^{-1} J^T r = r / (2*x) for diagonal case
        delta = residual / (2 * x + 1e-8)
        return x - delta

    # Run fixed number of iterations
    for _ in range(20):
        x = gn_step(x)

    return x


def _solve_nonlinear_iterative_fwd(
    target: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Forward pass for nonlinear iterative solve."""
    x = _solve_nonlinear_iterative(target)
    return x, (target, x)


def _solve_nonlinear_iterative_bwd(
    residuals: tuple[jax.Array, jax.Array], cotangent: jax.Array
) -> tuple[jax.Array]:
    """Backward pass for nonlinear iterative solve using adjoint method.

    For the problem min_x ||f(x) - target||^2 where f(x) = x^2:
    - At optimum x*, we have f(x*) = target, so x* = sqrt(target)
    - The Jacobian of f is J = diag(2*x)

    Using implicit differentiation through the optimality condition g(x, target) = J^T r = 0:
    - dg/dx = J^T @ J (at r=0, ignoring second-order terms)
    - dg/dtarget = J^T @ dr/dtarget = J^T @ (-I) = -J^T

    So: dx/dtarget = -(J^T J)^{-1} @ dg/dtarget = (J^T J)^{-1} @ J^T

    For the diagonal case:
    - J = diag(2*x), J^T J = diag(4*x^2)
    - (J^T J)^{-1} @ J^T = diag(1/(4*x^2)) @ diag(2*x) = diag(1/(2*x))

    VJP: dL/dtarget = cotangent^T @ dx/dtarget = cotangent / (2*x)
    """
    _target, x = residuals
    v = cotangent  # dL/dx*

    # dx*/dtarget = diag(1/(2*x)) for this problem
    # VJP: dL/dtarget = v @ diag(1/(2*x)) = v / (2*x)
    grad_target = v / (2 * x + 1e-8)

    return (grad_target,)


_solve_nonlinear_iterative.defvjp(
    _solve_nonlinear_iterative_fwd, _solve_nonlinear_iterative_bwd
)


# =============================================================================
# Step 4 & 5: Full jaxls integration
# =============================================================================


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
