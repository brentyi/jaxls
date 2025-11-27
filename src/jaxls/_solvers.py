from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Hashable,
    Literal,
    assert_never,
    cast,
    overload,
)

import jax
import jax.flatten_util
import jax_dataclasses as jdc
import scipy
import scipy.sparse
from jax import numpy as jnp

from jaxls._preconditioning import (
    make_block_jacobi_precoditioner,
    make_point_jacobi_precoditioner,
)

from ._sparse_matrices import BlockRowSparseMatrix, SparseCooMatrix, SparseCsrMatrix
from ._variables import VarTypeOrdering, VarValues
from .utils import jax_log

if TYPE_CHECKING:
    import sksparse.cholmod

    from ._core import AnalyzedLeastSquaresProblem, Cost, CustomJacobianCache
    from ._variables import Var


_cholmod_analyze_cache: dict[Hashable, sksparse.cholmod.Factor] = {}


def _cholmod_solve(
    A: SparseCsrMatrix, ATb: jax.Array, lambd: float | jax.Array
) -> jax.Array:
    """JIT-friendly linear solve using CHOLMOD."""
    return jax.pure_callback(
        _cholmod_solve_on_host,
        ATb,  # Result shape/dtype.
        A,
        ATb,
        lambd,
        vmap_method="sequential",
    )


def _cholmod_solve_on_host(
    A: SparseCsrMatrix,
    ATb: jax.Array,
    lambd: float | jax.Array,
) -> jax.Array:
    """Solve a linear system using CHOLMOD. Should be called on the host."""
    import sksparse.cholmod

    # Matrix is transposed when we convert CSR to CSC.
    A_T_scipy = scipy.sparse.csc_matrix(
        (A.values, A.coords.indices, A.coords.indptr), shape=A.coords.shape[::-1]
    )

    # Cache sparsity pattern analysis.
    cache_key = (
        A.coords.indices.tobytes(),
        A.coords.indptr.tobytes(),
        A.coords.shape,
    )
    cost = _cholmod_analyze_cache.get(cache_key, None)
    if cost is None:
        cost = sksparse.cholmod.analyze_AAt(A_T_scipy)
        _cholmod_analyze_cache[cache_key] = cost

        max_cache_size = 512
        if len(_cholmod_analyze_cache) > max_cache_size:
            _cholmod_analyze_cache.pop(next(iter(_cholmod_analyze_cache)))

    # Factorize and solve
    cost = cost.cholesky_AAt(
        A_T_scipy,
        # Some simple linear problems blow up without this 1e-5 term.
        beta=lambd + 1e-5,
    )
    return cost.solve_A(ATb)


@jdc.pytree_dataclass
class _ConjugateGradientState:
    """State used for Eisenstat-Walker criterion in ConjugateGradientLinearSolver."""

    ATb_norm_prev: float | jax.Array
    """Previous norm of ATb."""
    eta: float | jax.Array
    """Current tolerance."""


@jdc.pytree_dataclass
class ConjugateGradientConfig:
    """Iterative solver for sparse linear systems. Can run on CPU or GPU.

    For inexact steps, we use the Eisenstat-Walker criterion. For reference,
    see "Choosing the Forcing Terms in an Inexact Newton Method", Eisenstat &
    Walker, 1996."
    """

    tolerance_min: float | jax.Array = 1e-7
    tolerance_max: float | jax.Array = 1e-2

    eisenstat_walker_gamma: float | jax.Array = 0.9
    """Eisenstat-Walker criterion gamma term. Controls how quickly the tolerance
    decreases. Typical values range from 0.5 to 0.9. Higher values lead to more
    aggressive tolerance reduction."""
    eisenstat_walker_alpha: float | jax.Array = 2.0
    """ Eisenstat-Walker criterion alpha term. Determines rate at which the
    tolerance changes based on residual reduction. Typical values are 1.5 or
    2.0. Higher values make the tolerance more sensitive to residual changes."""

    preconditioner: jdc.Static[Literal["block_jacobi", "point_jacobi"] | None] = (
        "block_jacobi"
    )
    """Preconditioner to use for linear solves."""

    def _solve(
        self,
        problem: AnalyzedLeastSquaresProblem,
        A_blocksparse: BlockRowSparseMatrix,
        ATA_multiply: Callable[[jax.Array], jax.Array],
        ATb: jax.Array,
        prev_linear_state: _ConjugateGradientState,
    ) -> tuple[jax.Array, _ConjugateGradientState]:
        assert len(ATb.shape) == 1, "ATb should be 1D!"

        # Preconditioning setup.
        if self.preconditioner == "block_jacobi":
            preconditioner = make_block_jacobi_precoditioner(problem, A_blocksparse)
        elif self.preconditioner == "point_jacobi":
            preconditioner = make_point_jacobi_precoditioner(A_blocksparse)
        elif self.preconditioner is None:
            preconditioner = lambda x: x
        else:
            assert_never(self.preconditioner)

        # Calculate tolerance using Eisenstat-Walker criterion.
        ATb_norm = jnp.linalg.norm(ATb)
        current_eta = jnp.minimum(
            self.eisenstat_walker_gamma
            * (ATb_norm / (prev_linear_state.ATb_norm_prev + 1e-7))
            ** self.eisenstat_walker_alpha,
            self.tolerance_max,
        )
        current_eta = jnp.maximum(
            self.tolerance_min, jnp.minimum(current_eta, prev_linear_state.eta)
        )

        # Solve with conjugate gradient.
        initial_x = jnp.zeros(ATb.shape)
        solution_values, _ = jax.scipy.sparse.linalg.cg(
            A=ATA_multiply,
            b=ATb,
            x0=initial_x,
            # https://en.wikipedia.org/wiki/Conjugate_gradient_method#Convergence_properties
            maxiter=len(initial_x),
            tol=cast(float, current_eta),
            M=preconditioner,
        )
        return solution_values, _ConjugateGradientState(
            ATb_norm_prev=ATb_norm, eta=current_eta
        )


# Nonlinear solvers.


@jdc.pytree_dataclass
class SolveSummary:
    iterations: jax.Array
    termination_criteria: jax.Array
    termination_deltas: jax.Array
    cost_history: jax.Array
    lambda_history: jax.Array


@jdc.pytree_dataclass
class _NonlinearSolverState:
    vals: VarValues
    cost: jax.Array
    residual_vector: jax.Array
    summary: SolveSummary
    lambd: float | jax.Array

    jac_cache: tuple[CustomJacobianCache, ...]
    """Cache used to save intermediate values from `compute_residual_vector`
    for use in custom Jacobians."""

    cg_state: _ConjugateGradientState | None
    """Conjugate gradient state. Not used for other solvers."""

    jacobian_scaler: jax.Array | None


@jdc.pytree_dataclass
class NonlinearSolver:
    """Helper class for solving using Gauss-Newton or Levenberg-Marquardt."""

    linear_solver: jdc.Static[
        Literal["conjugate_gradient", "cholmod", "dense_cholesky"]
    ]
    trust_region: TrustRegionConfig | None
    termination: TerminationConfig
    conjugate_gradient_config: ConjugateGradientConfig | None
    sparse_mode: jdc.Static[Literal["blockrow", "coo", "csr"]]
    verbose: jdc.Static[bool]

    @overload
    def solve(
        self,
        problem: AnalyzedLeastSquaresProblem,
        initial_vals: VarValues,
        return_summary: Literal[False] = False,
    ) -> VarValues: ...

    @overload
    def solve(
        self,
        problem: AnalyzedLeastSquaresProblem,
        initial_vals: VarValues,
        return_summary: Literal[True],
    ) -> tuple[VarValues, SolveSummary]: ...

    @overload
    def solve(
        self,
        problem: AnalyzedLeastSquaresProblem,
        initial_vals: VarValues,
        return_summary: bool,
    ) -> VarValues | tuple[VarValues, SolveSummary]: ...

    @jdc.jit
    def solve(
        self,
        problem: AnalyzedLeastSquaresProblem,
        initial_vals: VarValues,
        return_summary: jdc.Static[bool] = False,
    ) -> VarValues | tuple[VarValues, SolveSummary]:
        vals = initial_vals
        residual_vector, jac_cache = problem.compute_residual_vector(
            vals, include_jac_cache=True
        )

        initial_cost = jnp.sum(residual_vector**2)
        cost_history = jnp.zeros(self.termination.max_iterations)
        cost_history = cost_history.at[0].set(initial_cost)
        lambda_history = jnp.zeros(self.termination.max_iterations)
        if self.trust_region is not None:
            lambda_history = lambda_history.at[0].set(self.trust_region.lambda_initial)

        state = _NonlinearSolverState(
            vals=vals,
            cost=initial_cost,
            residual_vector=residual_vector,
            summary=SolveSummary(
                iterations=jnp.array(0),
                cost_history=cost_history,
                lambda_history=lambda_history,
                termination_criteria=jnp.array([False, False, False, False]),
                termination_deltas=jnp.zeros(3),
            ),
            lambd=self.trust_region.lambda_initial
            if self.trust_region is not None
            else 0.0,
            cg_state=None
            if self.linear_solver != "conjugate_gradient"
            else _ConjugateGradientState(
                ATb_norm_prev=0.0,
                eta=(
                    ConjugateGradientConfig()
                    if self.conjugate_gradient_config is None
                    else self.conjugate_gradient_config
                ).tolerance_max,
            ),
            jac_cache=jac_cache,
            jacobian_scaler=None,
        )

        # Optimization.
        state = self.step(problem, state, first=True)
        if self.termination.early_termination:
            state = jax.lax.while_loop(
                cond_fun=lambda state: jnp.logical_not(
                    jnp.any(state.summary.termination_criteria)
                ),
                body_fun=lambda state: self.step(problem, state, first=False),
                init_val=state,
            )
        else:
            state = jax.lax.fori_loop(
                1,  # Start from 1 since we already did one step!
                self.termination.max_iterations,
                body_fun=lambda step, state: self.step(problem, state, first=False),
                init_val=state,
            )
        if self.verbose:
            jax_log(
                "Terminated @ iteration #{i}: cost={cost:.4f} criteria={criteria}, term_deltas={cost_delta:.1e},{grad_mag:.1e},{param_delta:.1e}",
                i=state.summary.iterations,
                cost=state.cost,
                criteria=state.summary.termination_criteria.astype(jnp.int32),
                cost_delta=state.summary.termination_deltas[0],
                grad_mag=state.summary.termination_deltas[1],
                param_delta=state.summary.termination_deltas[2],
            )

        if return_summary:
            return state.vals, state.summary
        else:
            return state.vals

    def step(
        self,
        problem: AnalyzedLeastSquaresProblem,
        state: _NonlinearSolverState,
        first: bool,
    ) -> _NonlinearSolverState:
        # Log optimizer state for debugging.
        if self.verbose:
            self._log_state(problem, state)

        # Get nonzero values of Jacobian.
        A_blocksparse = problem._compute_jac_values(state.vals, state.jac_cache)

        # Compute Jacobian scaler.
        if first:
            with jdc.copy_and_mutate(state, validate=False) as state:
                state.jacobian_scaler = (
                    1.0 / (1.0 + A_blocksparse.compute_column_norms()) * 0.0 + 1.0
                )
        assert state.jacobian_scaler is not None
        A_blocksparse = A_blocksparse.scale_columns(state.jacobian_scaler)

        # Get flattened version for COO/CSR matrices.
        jac_values = jnp.concatenate(
            [
                block_row.blocks_concat.flatten()
                for block_row in A_blocksparse.block_rows
            ],
            axis=0,
        )

        # linear_transpose() will return a tuple, with one element per primal.
        if self.sparse_mode == "blockrow":
            A_multiply = A_blocksparse.multiply
            AT_multiply_ = jax.linear_transpose(
                A_multiply, jnp.zeros((A_blocksparse.shape[1],))
            )
            AT_multiply = lambda vec: AT_multiply_(vec)[0]
        elif self.sparse_mode == "coo":
            A_coo = SparseCooMatrix(
                values=jac_values, coords=problem.jac_coords_coo
            ).as_jax_bcoo()
            AT_coo = A_coo.transpose()
            A_multiply = lambda vec: A_coo @ vec
            AT_multiply = lambda vec: AT_coo @ vec
        elif self.sparse_mode == "csr":
            A_csr = SparseCsrMatrix(
                values=jac_values, coords=problem.jac_coords_csr
            ).as_jax_bcsr()
            A_multiply = lambda vec: A_csr @ vec
            AT_multiply_ = jax.linear_transpose(
                A_multiply, jnp.zeros((A_blocksparse.shape[1],))
            )
            AT_multiply = lambda vec: AT_multiply_(vec)[0]
        else:
            assert_never(self.sparse_mode)

        # Compute right-hand side of normal equation.
        ATb = -AT_multiply(state.residual_vector)

        linear_state = None
        if (
            isinstance(self.linear_solver, ConjugateGradientConfig)
            or self.linear_solver == "conjugate_gradient"
        ):
            # Use default CG config is specified as a string, otherwise use the provided config.
            cg_config = (
                ConjugateGradientConfig()
                if self.linear_solver == "conjugate_gradient"
                else self.linear_solver
            )
            assert isinstance(state.cg_state, _ConjugateGradientState)
            local_delta, linear_state = cg_config._solve(
                problem,
                A_blocksparse,
                # We could also use (lambd * ATA_diagonals * vec) for
                # scale-invariant damping. But this is hard to match with CHOLMOD.
                lambda vec: AT_multiply(A_multiply(vec)) + state.lambd * vec,
                ATb=ATb,
                prev_linear_state=state.cg_state,
            )
        elif self.linear_solver == "cholmod":
            # Use CHOLMOD for direct solve.
            A_csr = SparseCsrMatrix(jac_values, problem.jac_coords_csr)
            local_delta = _cholmod_solve(A_csr, ATb, lambd=state.lambd)
        elif self.linear_solver == "dense_cholesky":
            A_dense = A_blocksparse.to_dense()
            ATA = A_dense.T @ A_dense
            diag_idx = jnp.arange(ATA.shape[0])
            ATA = ATA.at[diag_idx, diag_idx].add(state.lambd)
            cho_factor = jax.scipy.linalg.cho_factor(ATA)
            local_delta = jax.scipy.linalg.cho_solve(cho_factor, ATb)
        else:
            assert_never(self.linear_solver)

        scaled_local_delta = local_delta * state.jacobian_scaler

        proposed_vals = state.vals._retract(
            scaled_local_delta, problem.tangent_ordering
        )
        proposed_residual_vector, proposed_jac_cache = problem.compute_residual_vector(
            proposed_vals, include_jac_cache=True
        )
        proposed_cost = jnp.sum(proposed_residual_vector**2)

        # Always accept Gauss-Newton steps.
        if self.trust_region is None:
            with jdc.copy_and_mutate(state) as state_next:
                # Update ATb_norm for Eisenstat-Walker criterion.
                if linear_state is not None:
                    state_next.cg_state = linear_state

                state_next.vals = proposed_vals
                state_next.residual_vector = proposed_residual_vector
                state_next.cost = proposed_cost
                accept_flag = None
        # For Levenberg-Marquardt, we need to evaluate the step quality.
        else:
            step_quality = (proposed_cost - state.cost) / (
                jnp.sum(
                    (A_blocksparse.multiply(scaled_local_delta) + state.residual_vector)
                    ** 2
                )
                - state.cost
            )
            accept_flag = step_quality >= self.trust_region.step_quality_min

            # What does the accepted state look like?
            with jdc.copy_and_mutate(state) as state_accept:
                # Update ATb_norm for Eisenstat-Walker criterion.
                if linear_state is not None:
                    state_accept.cg_state = linear_state

                state_accept.vals = proposed_vals
                state_accept.residual_vector = proposed_residual_vector
                state_accept.cost = proposed_cost
                state_accept.jac_cache = proposed_jac_cache
                state_accept.lambd = state.lambd / self.trust_region.lambda_factor

            # What does the rejected state look like?
            with jdc.copy_and_mutate(state) as state_reject:
                state_reject.lambd = jnp.maximum(
                    self.trust_region.lambda_min,
                    jnp.minimum(
                        state.lambd * self.trust_region.lambda_factor,
                        self.trust_region.lambda_max,
                    ),
                )

            # Update the state with the accepted or rejected values.
            state_next = jax.tree.map(
                lambda x, y: x if (x is y) else jnp.where(accept_flag, x, y),
                state_accept,
                state_reject,
            )

        # Update termination criteria + summary.
        with jdc.copy_and_mutate(state_next) as state_next:
            if self.termination.early_termination:
                (
                    state_next.summary.termination_criteria,
                    state_next.summary.termination_deltas,
                ) = self.termination._check_convergence(
                    state,
                    cost_updated=proposed_cost,
                    tangent=local_delta,
                    tangent_ordering=problem.tangent_ordering,
                    ATb=ATb,
                    accept_flag=accept_flag,
                )
            state_next.summary.iterations += 1
            state_next.summary.cost_history = state_next.summary.cost_history.at[
                state_next.summary.iterations
            ].set(state_next.cost)
            state_next.summary.lambda_history = state_next.summary.lambda_history.at[
                state_next.summary.iterations
            ].set(state_next.lambd)
        return state_next

    @staticmethod
    def _log_state(
        problem: AnalyzedLeastSquaresProblem, state: _NonlinearSolverState
    ) -> None:
        if state.cg_state is None:
            jax_log(
                " step #{i}: cost={cost:.4f} lambd={lambd:.4f} term_deltas={cost_delta:.1e},{grad_mag:.1e},{param_delta:.1e}",
                i=state.summary.iterations,
                cost=state.cost,
                lambd=state.lambd,
                cost_delta=state.summary.termination_deltas[0],
                grad_mag=state.summary.termination_deltas[1],
                param_delta=state.summary.termination_deltas[2],
                ordered=True,
            )
        else:
            jax_log(
                " step #{i}: cost={cost:.4f} lambd={lambd:.4f} term_deltas={cost_delta:.1e},{grad_mag:.1e},{param_delta:.1e} inexact_tol={inexact_tol:.1e}",
                i=state.summary.iterations,
                cost=state.cost,
                lambd=state.lambd,
                cost_delta=state.summary.termination_deltas[0],
                grad_mag=state.summary.termination_deltas[1],
                param_delta=state.summary.termination_deltas[2],
                inexact_tol=state.cg_state.eta,
                ordered=True,
            )
        residual_index = 0
        for f, count in zip(problem.stacked_costs, problem.cost_counts):
            stacked_dim = count * f.residual_flat_dim
            partial_cost = jnp.sum(
                state.residual_vector[residual_index : residual_index + stacked_dim]
                ** 2
            )
            residual_index += stacked_dim
            jax_log(
                "     - "
                + f"{f._get_name()}({count}):".ljust(15)
                + " {:.5f} (avg {:.5f})",
                partial_cost,
                partial_cost / stacked_dim,
                ordered=True,
            )


@jdc.pytree_dataclass
class TrustRegionConfig:
    # Levenberg-Marquardt parameters.
    lambda_initial: float | jax.Array = 5e-4
    """Initial damping factor. Only used for Levenberg-Marquardt."""
    lambda_factor: float | jax.Array = 2.0
    """Factor to increase or decrease damping. Only used for Levenberg-Marquardt."""
    lambda_min: float | jax.Array = 1e-5
    """Minimum damping factor. Only used for Levenberg-Marquardt."""
    lambda_max: float | jax.Array = 1e10
    """Maximum damping factor. Only used for Levenberg-Marquardt."""
    step_quality_min: float | jax.Array = 1e-3
    """Minimum step quality for Levenberg-Marquardt. Only used for Levenberg-Marquardt."""


@jdc.pytree_dataclass
class TerminationConfig:
    # Termination criteria.
    max_iterations: jdc.Static[int] = 100
    """Maximum number of optimization steps."""
    early_termination: jdc.Static[bool] = True
    """If set to `True`, terminate when any of the tolerances are met. If
    `False`, always run `max_iterations` steps."""
    cost_tolerance: float | jax.Array = 1e-5
    """We terminate if `|cost change| / cost < cost_tolerance`."""
    gradient_tolerance: float | jax.Array = 1e-4
    """We terminate if `norm_inf(x - rplus(x, linear delta)) < gradient_tolerance`."""
    gradient_tolerance_start_step: int | jax.Array = 10
    """When to start checking the gradient tolerance condition. Helps solve precision
    issues caused by inexact Newton steps."""
    parameter_tolerance: float | jax.Array = 1e-6
    """We terminate if `norm_2(linear delta) < (norm2(x) + parameter_tolerance) * parameter_tolerance`."""

    def _check_convergence(
        self,
        state_prev: _NonlinearSolverState,
        cost_updated: jax.Array,
        tangent: jax.Array,
        tangent_ordering: VarTypeOrdering,
        ATb: jax.Array,
        accept_flag: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Check for convergence!"""

        # Cost tolerance
        cost_absdelta = jnp.abs(cost_updated - state_prev.cost)
        cost_reldelta = cost_absdelta / state_prev.cost
        converged_cost = cost_reldelta < self.cost_tolerance

        # Gradient tolerance
        flat_vals = jax.flatten_util.ravel_pytree(state_prev.vals)[0]
        gradient_mag = jnp.max(
            flat_vals
            - jax.flatten_util.ravel_pytree(
                state_prev.vals._retract(ATb, tangent_ordering)
            )[0]
        )
        converged_gradient = jnp.where(
            state_prev.summary.iterations >= self.gradient_tolerance_start_step,
            gradient_mag < self.gradient_tolerance,
            False,
        )

        # Parameter tolerance
        param_delta = jnp.linalg.norm(jnp.abs(tangent)) / (
            jnp.linalg.norm(flat_vals) + self.parameter_tolerance
        )
        converged_parameters = param_delta < self.parameter_tolerance

        # Check termination flags. We'll terminate if any of the conditions are met.
        term_flags = jnp.array(
            [
                converged_cost,
                converged_gradient,
                converged_parameters,
                state_prev.summary.iterations >= (self.max_iterations - 1),
            ]
        )

        # Only consider the first three conditions if steps are accepted.
        if accept_flag is not None:
            term_flags = term_flags.at[:3].set(
                jnp.logical_and(
                    term_flags[:3],
                    # We ignore accept_flag if the cost _actually_ didn't change at all.
                    jnp.logical_or(accept_flag, cost_absdelta == 0.0),
                )
            )

        return term_flags, jnp.array([cost_reldelta, gradient_mag, param_delta])


@jdc.pytree_dataclass
class AugmentedLagrangianConfig:
    """Configuration for Augmented Lagrangian solver (ALGENCAN-style)."""

    penalty_initial: float | jax.Array | jdc.Static[Literal["auto"]] = "auto"
    """Initial penalty. If 'auto', computed from initial cost and constraint violation."""

    penalty_factor: float | jax.Array = 10.0
    """Penalty multiplier when constraint progress stagnates."""

    penalty_max: float | jax.Array = 1e8
    """Maximum penalty parameter."""

    penalty_min: float | jax.Array = 1e-6
    """Minimum penalty parameter."""

    tolerance_absolute: float | jax.Array = 1e-6
    """Absolute convergence tolerance: max(snorm, csupn) < tol."""

    tolerance_relative: float | jax.Array = 1e-4
    """Relative convergence tolerance: snorm / snorm_initial < tol."""

    max_iterations: jdc.Static[int] = 50
    """Maximum outer loop iterations."""

    inner_tolerance_factor: float | jax.Array = 0.1
    """Inner solver tolerance as fraction of current snorm."""

    violation_reduction_threshold: float | jax.Array = 0.5
    """Increase penalty if violation doesn't reduce by this fraction."""

    lambda_min: float | jax.Array = -1e8
    """Minimum Lagrange multiplier (safeguard)."""

    lambda_max: float | jax.Array = 1e8
    """Maximum Lagrange multiplier (safeguard)."""


@jdc.pytree_dataclass
class _AugmentedLagrangianState:
    """State for outer Augmented Lagrangian loop."""

    vals: VarValues
    """Current variable values."""

    lagrange_multipliers: jax.Array
    """Lagrange multipliers, flat array of shape (constraint_dim,)."""

    penalty_params: jax.Array
    """Per-constraint penalty parameters, shape (constraint_dim,)."""

    snorm: jax.Array
    """Complementarity measure (infinity-norm). Equality: |h(x)|, inequality: |min(-g(x), λ)|."""

    snorm_prev: jax.Array
    """Previous snorm for progress checking."""

    constraint_values_prev: jax.Array
    """Previous constraint values for per-constraint progress, shape (constraint_dim,)."""

    constraint_violation: jax.Array
    """Constraint violation (infinity-norm). Equality: max|h(x)|, inequality: max(0, g(x))."""

    initial_snorm: jax.Array
    """Initial snorm for relative tolerance check."""

    outer_iteration: int
    """Current outer iteration number."""

    inner_summary: SolveSummary
    """Summary from most recent inner solve."""

    # Pre-allocated history arrays for JIT compatibility.
    constraint_violation_history: jax.Array
    """Constraint violation history, shape (max_outer_iterations,)."""

    penalty_history: jax.Array
    """Max penalty history, shape (max_outer_iterations,)."""

    inner_iterations_count: jax.Array
    """Inner iteration counts, shape (max_outer_iterations,)."""


@jdc.pytree_dataclass
class AugmentedLagrangianSolver:
    """Solver for constrained NLLS problems using Augmented Lagrangian method.

    This solver handles equality constraints of the form h(x) = 0 by converting
    them into an augmented cost function and iteratively solving unconstrained
    subproblems while updating Lagrange multipliers and penalty parameters.

    Note: For simple equality constraints between variables (e.g., x₁ = x₂),
    reparameterizing the problem to eliminate redundant variables will typically
    result in better conditioning and faster convergence. Augmented Lagrangian
    is most useful for nonlinear constraints where direct variable elimination
    is not straightforward.
    """

    config: AugmentedLagrangianConfig
    """Configuration for the Augmented Lagrangian method."""

    inner_solver: NonlinearSolver
    """Solver for the inner unconstrained subproblems."""

    verbose: jdc.Static[bool]
    """Whether to print verbose logging information."""

    def _extract_original_costs(
        self, problem: AnalyzedLeastSquaresProblem
    ) -> list[Cost]:
        """Extract original Cost objects from analyzed problem.

        We need to recreate Cost objects from the analyzed problem to combine
        with augmented constraint costs and re-analyze.
        """
        from ._core import Cost

        original_costs = []
        for stacked_cost, count in zip(problem.stacked_costs, problem.cost_counts):
            # Keep the stacked cost as-is
            original_costs.append(
                Cost(
                    compute_residual=stacked_cost.compute_residual,
                    args=stacked_cost.args,
                    jac_mode=stacked_cost.jac_mode,
                    jac_batch_size=stacked_cost.jac_batch_size,
                    jac_custom_fn=stacked_cost.jac_custom_fn,
                    jac_custom_with_cache_fn=stacked_cost.jac_custom_with_cache_fn,
                    name=stacked_cost.name,
                )
            )
        return original_costs

    def _extract_variables(self, problem: AnalyzedLeastSquaresProblem) -> list[Var]:
        """Extract variable objects from analyzed problem."""

        variables = []
        for var_type, ids in problem.sorted_ids_from_var_type.items():
            for var_id in ids:
                variables.append(var_type(int(var_id)))
        return variables

    def _analyze_augmented_problem(
        self, problem: AnalyzedLeastSquaresProblem, constraint_dim: int
    ) -> AnalyzedLeastSquaresProblem:
        """Pre-analyze augmented problem structure once.

        Creates parameterized augmented costs with external parameter references,
        then analyzes the combined problem. The structure is fixed; only the
        lagrange_multipliers and penalty_param values will vary during optimization.

        Args:
            problem: Original analyzed problem with constraints.
            constraint_dim: Total dimension of all constraints.

        Returns:
            Analyzed augmented problem with external params initialized to zeros.
        """
        from ._core import LeastSquaresProblem, create_augmented_constraint_cost

        # Create augmented costs for each constraint group
        constraint_costs = []
        constraint_dims = []
        constraint_is_inequality = []  # Track which constraints are inequalities

        for i, (stacked_constraint, constraint_count) in enumerate(
            zip(problem.stacked_constraints, problem.constraint_counts)
        ):
            constraint_flat_dim = stacked_constraint.constraint_flat_dim
            total_dim = constraint_count * constraint_flat_dim
            constraint_dims.append(total_dim)
            constraint_is_inequality.append(
                stacked_constraint.constraint_type == "leq_zero"
            )

            # Create augmented cost that accepts AugmentedLagrangianParams
            cost = create_augmented_constraint_cost(stacked_constraint, i, total_dim)
            constraint_costs.append(cost)

        # Extract original costs
        original_costs = self._extract_original_costs(problem)

        # Extract variables
        variables = self._extract_variables(problem)

        # Create and analyze combined problem (ONE TIME ONLY!)
        if self.verbose:
            jax_log("Pre-analyzing augmented problem structure (one-time cost)...")

        augmented = LeastSquaresProblem(
            costs=original_costs + constraint_costs,
            variables=variables,
        ).analyze()

        # Initialize AL params with zeros/ones (will be updated in loop)
        from ._core import AugmentedLagrangianParams

        lagrange_mult_arrays = tuple(jnp.zeros(dim) for dim in constraint_dims)
        # Per-constraint penalties: one penalty value per constraint element
        penalty_param_arrays = tuple(jnp.ones(dim) for dim in constraint_dims)
        al_params = AugmentedLagrangianParams(
            lagrange_multipliers=lagrange_mult_arrays,
            penalty_params=penalty_param_arrays,
        )

        augmented = jdc.replace(augmented, augmented_lagrangian_params=al_params)

        return augmented

    @overload
    def solve(
        self,
        problem: AnalyzedLeastSquaresProblem,
        initial_vals: VarValues,
        return_summary: Literal[False] = False,
    ) -> VarValues: ...

    @overload
    def solve(
        self,
        problem: AnalyzedLeastSquaresProblem,
        initial_vals: VarValues,
        return_summary: Literal[True],
    ) -> tuple[VarValues, SolveSummary]: ...

    @overload
    def solve(
        self,
        problem: AnalyzedLeastSquaresProblem,
        initial_vals: VarValues,
        return_summary: bool,
    ) -> VarValues | tuple[VarValues, SolveSummary]: ...

    def solve(
        self,
        problem: AnalyzedLeastSquaresProblem,
        initial_vals: VarValues,
        return_summary: jdc.Static[bool] = False,
    ) -> VarValues | tuple[VarValues, SolveSummary]:
        """Solve constrained optimization problem using Augmented Lagrangian (ALGENCAN-style).

        Args:
            problem: The analyzed least squares problem with constraints.
            initial_vals: Initial values for all variables.
            return_summary: Whether to return solve summary.

        Returns:
            Solution values, and optionally a solve summary.
        """
        # Ensure we have constraints.
        assert len(problem.stacked_constraints) > 0, (
            "AugmentedLagrangianSolver requires constraints. "
            "Use NonlinearSolver for unconstrained problems."
        )

        # Compute initial constraint values.
        h_vals = problem.compute_constraint_values(initial_vals)
        constraint_dim = len(h_vals)

        # Initialize Lagrange multipliers (zeros, as recommended by literature).
        lagrange_multipliers = jnp.zeros(constraint_dim)

        # Compute initial snorm (complementarity measure) and csupn (constraint violation).
        # For initial state, lambdas are zero, so snorm = csupn for equalities,
        # and snorm = |min(-g, 0)| = |max(g, 0)| for violated inequalities.
        initial_snorm, initial_csupn = self._compute_snorm_csupn(
            problem, h_vals, lagrange_multipliers
        )

        # Initialize penalty parameter using ALGENCAN formula.
        if self.config.penalty_initial == "auto":
            # ALGENCAN formula: rho = 10 * max(1, |f|) / max(1, 0.5 * sum(c^2))
            residual_vector_result = problem.compute_residual_vector(initial_vals)
            if isinstance(residual_vector_result, tuple):
                residual_vector = residual_vector_result[0]
            else:
                residual_vector = residual_vector_result
            initial_cost = jnp.sum(residual_vector**2)

            # Compute sum of squared constraint violations
            # For inequalities, only count violated constraints (g > 0)
            sum_c_squared = jnp.array(0.0)
            offset = 0
            for i, stacked_constraint in enumerate(problem.stacked_constraints):
                constraint_count = problem.constraint_counts[i]
                constraint_flat_dim = stacked_constraint.constraint_flat_dim
                total_dim = constraint_count * constraint_flat_dim
                h_slice = h_vals[offset : offset + total_dim]

                if stacked_constraint.constraint_type == "leq_zero":
                    # Only count violated inequalities
                    sum_c_squared = sum_c_squared + jnp.sum(
                        0.5 * jnp.maximum(0.0, h_slice) ** 2
                    )
                else:
                    # Count all equality constraint values
                    sum_c_squared = sum_c_squared + jnp.sum(0.5 * h_slice**2)
                offset += total_dim

            penalty_initial = (
                10.0
                * jnp.maximum(1.0, jnp.abs(initial_cost))
                / jnp.maximum(1.0, sum_c_squared)
            )
            penalty_initial = jnp.clip(
                penalty_initial, self.config.penalty_min, self.config.penalty_max
            )
        else:
            penalty_initial = self.config.penalty_initial

        # Initialize per-constraint penalties (all same initially).
        penalty_params = jnp.full(constraint_dim, penalty_initial)

        # PRE-ANALYZE augmented problem structure ONCE (major optimization!)
        # This converts constraints to parameterized costs and analyzes the combined problem.
        # The structure is fixed; only lagrange_multipliers and penalty_params will vary.
        augmented_structure = self._analyze_augmented_problem(problem, constraint_dim)

        if self.verbose:
            jax_log(
                "Augmented Lagrangian: initial snorm={snorm:.4e}, csupn={csupn:.4e}, penalty={penalty:.4e}, constraint_dim={dim}",
                snorm=initial_snorm,
                csupn=initial_csupn,
                penalty=penalty_initial,
                dim=constraint_dim,
            )

        # Pre-allocate history arrays for outer loop (JIT-friendly).
        max_outer = self.config.max_iterations
        constraint_violation_history = jnp.zeros(max_outer)
        constraint_violation_history = constraint_violation_history.at[0].set(
            initial_csupn
        )
        penalty_history = jnp.zeros(max_outer)
        penalty_history = penalty_history.at[0].set(penalty_initial)
        inner_iterations_count = jnp.zeros(max_outer, dtype=jnp.int32)

        # Pre-allocate inner_summary with fixed-size arrays based on inner solver max iterations.
        max_inner = self.inner_solver.termination.max_iterations
        initial_summary = SolveSummary(
            iterations=jnp.array(0, dtype=jnp.int32),
            cost_history=jnp.zeros(max_inner),
            lambda_history=jnp.zeros(max_inner),
            termination_criteria=jnp.array([False, False, False, False]),
            termination_deltas=jnp.zeros(3),
        )

        # Create initial state.
        state = _AugmentedLagrangianState(
            vals=initial_vals,
            lagrange_multipliers=lagrange_multipliers,
            penalty_params=penalty_params,
            snorm=initial_snorm,
            snorm_prev=initial_snorm,  # Set prev = current initially
            constraint_values_prev=h_vals,  # Initial constraint values for per-constraint progress
            constraint_violation=initial_csupn,
            initial_snorm=initial_snorm,
            outer_iteration=0,
            inner_summary=initial_summary,
            constraint_violation_history=constraint_violation_history,
            penalty_history=penalty_history,
            inner_iterations_count=inner_iterations_count,
        )

        # Run outer loop using while_loop for JIT compilation.
        def cond_fn(state: _AugmentedLagrangianState) -> jax.Array:
            """Continue if not converged and under max iterations.

            ALGENCAN convergence: max(snorm, csupn) <= tol_abs AND snorm/snorm0 < tol_rel

            We always run at least one iteration to optimize the cost even when
            constraints are initially satisfied (e.g., inactive inequality constraints).
            """
            # Always run at least one iteration
            first_iteration = state.outer_iteration < 1

            converged_absolute = (
                jnp.maximum(state.snorm, state.constraint_violation)
                < self.config.tolerance_absolute
            )
            converged_relative = (
                state.snorm / (state.initial_snorm + 1e-6)
                < self.config.tolerance_relative
            )
            converged = converged_absolute & converged_relative
            under_max_iters = state.outer_iteration < self.config.max_iterations
            return (first_iteration | ~converged) & under_max_iters

        def body_fn(state: _AugmentedLagrangianState) -> _AugmentedLagrangianState:
            """Perform one outer iteration."""
            return self._step(problem, augmented_structure, state)

        state = jax.lax.while_loop(cond_fn, body_fn, state)

        # Check final convergence for logging.
        converged_absolute = (
            jnp.maximum(state.snorm, state.constraint_violation)
            < self.config.tolerance_absolute
        )
        converged_relative = (
            state.snorm / (state.initial_snorm + 1e-10)
            < self.config.tolerance_relative
        )

        if self.verbose:
            if converged_absolute and converged_relative:
                jax_log(
                    "Augmented Lagrangian converged @ outer iteration {i}: snorm={snorm:.4e}, csupn={csupn:.4e}",
                    i=state.outer_iteration,
                    snorm=state.snorm,
                    csupn=state.constraint_violation,
                )
            else:
                jax_log(
                    "Augmented Lagrangian: max iterations reached. Final snorm={snorm:.4e}, csupn={csupn:.4e}",
                    snorm=state.snorm,
                    csupn=state.constraint_violation,
                )

        if return_summary:
            return state.vals, state.inner_summary
        else:
            return state.vals

    def _compute_snorm_csupn(
        self,
        problem: AnalyzedLeastSquaresProblem,
        h_vals: jax.Array,
        lagrange_multipliers: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute complementarity (snorm) and constraint violation (csupn), both infinity-norm."""
        snorm_parts = []
        csupn_parts = []
        offset = 0

        for i, stacked_constraint in enumerate(problem.stacked_constraints):
            constraint_count = problem.constraint_counts[i]
            constraint_flat_dim = stacked_constraint.constraint_flat_dim
            total_dim = constraint_count * constraint_flat_dim
            h_slice = h_vals[offset : offset + total_dim]
            lambda_slice = lagrange_multipliers[offset : offset + total_dim]

            if stacked_constraint.constraint_type == "leq_zero":
                # Inequality: snorm = |min(-g, λ)|, csupn = max(0, g).
                comp = jnp.minimum(-h_slice, lambda_slice)
                snorm_parts.append(jnp.max(jnp.abs(comp)))
                csupn_parts.append(jnp.max(jnp.maximum(0.0, h_slice)))
            else:
                # Equality: snorm = csupn = |h|.
                snorm_parts.append(jnp.max(jnp.abs(h_slice)))
                csupn_parts.append(jnp.max(jnp.abs(h_slice)))

            offset += total_dim

        if len(snorm_parts) == 0:
            return jnp.array(0.0), jnp.array(0.0)

        snorm = jnp.max(jnp.array(snorm_parts))
        csupn = jnp.max(jnp.array(csupn_parts))
        return snorm, csupn

    def _compute_per_constraint_violation(
        self,
        problem: AnalyzedLeastSquaresProblem,
        h_vals: jax.Array,
    ) -> jax.Array:
        """Per-element violation: |h(x)| for equality, max(0, g(x)) for inequality."""
        violation_parts = []
        offset = 0

        for i, stacked_constraint in enumerate(problem.stacked_constraints):
            constraint_count = problem.constraint_counts[i]
            constraint_flat_dim = stacked_constraint.constraint_flat_dim
            total_dim = constraint_count * constraint_flat_dim
            h_slice = h_vals[offset : offset + total_dim]

            if stacked_constraint.constraint_type == "leq_zero":
                violation_parts.append(jnp.maximum(0.0, h_slice))
            else:
                violation_parts.append(jnp.abs(h_slice))

            offset += total_dim

        return jnp.concatenate(violation_parts)

    def _step(
        self,
        problem: AnalyzedLeastSquaresProblem,
        augmented_structure: AnalyzedLeastSquaresProblem,
        state: _AugmentedLagrangianState,
    ) -> _AugmentedLagrangianState:
        """Perform one outer iteration of the Augmented Lagrangian method (ALGENCAN-style).

        Args:
            problem: Original problem with constraints (for constraint evaluation).
            augmented_structure: Pre-analyzed augmented problem with parameterized costs.
            state: Current state of the Augmented Lagrangian solver.

        Returns:
            Updated state after one outer iteration.
        """
        # Update AL params in pre-analyzed problem. Split flat arrays into per-group tuples.
        from ._core import AugmentedLagrangianParams

        assert augmented_structure.augmented_lagrangian_params is not None
        lambda_arrays = []
        penalty_arrays = []
        offset = 0
        for (
            lambda_array
        ) in augmented_structure.augmented_lagrangian_params.lagrange_multipliers:
            dim = lambda_array.shape[0]
            lambda_arrays.append(
                state.lagrange_multipliers[offset : offset + dim]
            )
            penalty_arrays.append(state.penalty_params[offset : offset + dim])
            offset += dim

        al_params = AugmentedLagrangianParams(
            lagrange_multipliers=tuple(lambda_arrays),
            penalty_params=tuple(penalty_arrays),
        )

        augmented_problem = jdc.replace(
            augmented_structure,
            augmented_lagrangian_params=al_params,
        )

        # Solve inner unconstrained problem. Tolerance scales with snorm.
        inner_tolerance = self.config.inner_tolerance_factor * state.snorm
        inner_termination = jdc.replace(
            self.inner_solver.termination,
            cost_tolerance=jnp.maximum(inner_tolerance, 1e-6),
            gradient_tolerance=jnp.maximum(inner_tolerance, 1e-6),
        )
        inner_solver_updated = jdc.replace(
            self.inner_solver,
            termination=inner_termination,
        )

        vals_updated, inner_summary = inner_solver_updated.solve(
            augmented_problem, state.vals, return_summary=True
        )

        # Evaluate constraints at new solution.
        h_vals = problem.compute_constraint_values(vals_updated)

        # Update Lagrange multipliers: λ_new = λ_old + ρ * h(x).
        lagrange_multipliers_updated = (
            state.lagrange_multipliers + state.penalty_params * h_vals
        )

        # Project multipliers: non-negative for inequalities, then apply safeguard bounds.
        offset = 0
        lambda_arrays_projected = []
        for i, stacked_constraint in enumerate(problem.stacked_constraints):
            constraint_count = problem.constraint_counts[i]
            constraint_flat_dim = stacked_constraint.constraint_flat_dim
            total_dim = constraint_count * constraint_flat_dim

            lambda_slice = lagrange_multipliers_updated[offset : offset + total_dim]

            if stacked_constraint.constraint_type == "leq_zero":
                lambda_slice = jnp.maximum(0.0, lambda_slice)
            lambda_slice = jnp.clip(
                lambda_slice, self.config.lambda_min, self.config.lambda_max
            )

            lambda_arrays_projected.append(lambda_slice)
            offset += total_dim

        lagrange_multipliers_updated = jnp.concatenate(lambda_arrays_projected)

        # Compute snorm (complementarity) and csupn (constraint violation).
        snorm_new, csupn_new = self._compute_snorm_csupn(
            problem, h_vals, lagrange_multipliers_updated
        )

        # Per-constraint violation for penalty updates.
        constraint_violation_per = self._compute_per_constraint_violation(
            problem, h_vals
        )
        constraint_violation_prev_per = self._compute_per_constraint_violation(
            problem, state.constraint_values_prev
        )

        # Only increase penalty for constraints that didn't improve sufficiently.
        insufficient_progress_per = constraint_violation_per > (
            self.config.violation_reduction_threshold * constraint_violation_prev_per
        )

        # If inner solver failed, increase all penalties.
        inner_hit_max_iters = inner_summary.termination_criteria[3]
        inner_made_progress = inner_summary.termination_criteria[0]
        rhorestart_needed = inner_hit_max_iters & ~inner_made_progress

        # Update penalties for constraints with insufficient progress.
        penalty_params_updated = jnp.where(
            insufficient_progress_per | rhorestart_needed,
            jnp.minimum(
                state.penalty_params * self.config.penalty_factor,
                self.config.penalty_max,
            ),
            state.penalty_params,
        )

        if self.verbose:
            jax_log(
                " AL outer iter {i}: snorm={snorm:.4e}, csupn={csupn:.4e}, max_rho={max_rho:.4e}, inner_iters={inner_iters}",
                i=state.outer_iteration,
                snorm=snorm_new,
                csupn=csupn_new,
                max_rho=jnp.max(penalty_params_updated),
                inner_iters=inner_summary.iterations,
                ordered=True,
            )

        next_idx = state.outer_iteration + 1
        return jdc.replace(
            state,
            vals=vals_updated,
            lagrange_multipliers=lagrange_multipliers_updated,
            penalty_params=penalty_params_updated,
            snorm=snorm_new,
            snorm_prev=state.snorm,
            constraint_values_prev=h_vals,
            constraint_violation=csupn_new,
            outer_iteration=state.outer_iteration + 1,
            inner_summary=inner_summary,
            constraint_violation_history=state.constraint_violation_history.at[
                next_idx
            ].set(csupn_new),
            penalty_history=state.penalty_history.at[next_idx].set(
                jnp.max(penalty_params_updated)
            ),
            inner_iterations_count=state.inner_iterations_count.at[next_idx].set(
                inner_summary.iterations
            ),
        )
