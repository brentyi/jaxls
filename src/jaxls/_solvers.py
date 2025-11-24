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

    from ._core import AnalyzedLeastSquaresProblem, CustomJacobianCache


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
    """Configuration for Augmented Lagrangian method for constrained optimization."""

    penalty_initial: float | jax.Array | jdc.Static[Literal["auto"]] = "auto"
    """Initial penalty parameter. If 'auto', scales as initial_cost / 100."""

    penalty_factor: float | jax.Array = 10.0
    """Factor to multiply penalty parameter when constraint violation stagnates.
    Based on literature, typical values range from 2 to 10."""

    penalty_max: float | jax.Array = 1e8
    """Maximum penalty parameter to prevent numerical issues."""

    tolerance_absolute: float | jax.Array = 1e-6
    """Absolute tolerance for constraint violation: ||h(x)|| < tol_abs."""

    tolerance_relative: float | jax.Array = 1e-4
    """Relative tolerance for constraint violation: ||h(x)|| / ||h(x0)|| < tol_rel."""

    max_iterations: jdc.Static[int] = 20
    """Maximum number of augmented Lagrangian iterations (outer loop)."""

    inner_tolerance_factor: float | jax.Array = 0.1
    """Inner NLLS solver tolerance as fraction of current constraint violation."""

    violation_reduction_threshold: float | jax.Array = 0.25
    """Increase penalty if constraint violation doesn't reduce by this fraction."""


@jdc.pytree_dataclass
class _AugmentedLagrangianState:
    """State for outer Augmented Lagrangian loop."""

    vals: VarValues
    """Current variable values."""

    lagrange_multipliers: jax.Array
    """Lagrange multipliers for each constraint."""

    penalty_param: float | jax.Array
    """Current penalty parameter (ρ in literature, μ in ALGENCAN)."""

    constraint_violation: jax.Array
    """Current constraint violation ||h(x)||."""

    initial_violation: jax.Array
    """Initial constraint violation ||h(x0)|| for relative tolerance."""

    outer_iteration: int
    """Current outer iteration number."""

    inner_summary: SolveSummary
    """Summary from most recent inner solver iteration."""


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
        """Solve constrained optimization problem using Augmented Lagrangian.

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

        # Compute initial constraint violation.
        h_vals = problem.compute_constraint_values(initial_vals)
        constraint_dim = len(h_vals)
        initial_violation = jnp.linalg.norm(h_vals)

        # Initialize Lagrange multipliers (zeros, as recommended by literature).
        lagrange_multipliers = jnp.zeros(constraint_dim)

        # Initialize penalty parameter.
        if self.config.penalty_initial == "auto":
            # Auto-scale based on initial cost.
            residual_vector_result = problem.compute_residual_vector(initial_vals)
            if isinstance(residual_vector_result, tuple):
                residual_vector = residual_vector_result[0]
            else:
                residual_vector = residual_vector_result
            initial_cost = jnp.sum(residual_vector**2)
            penalty_param = initial_cost / 100.0
        else:
            penalty_param = self.config.penalty_initial

        if self.verbose:
            jax_log(
                "Augmented Lagrangian: initial violation={violation:.4e}, penalty={penalty:.4e}, constraint_dim={dim}",
                violation=initial_violation,
                penalty=penalty_param,
                dim=constraint_dim,
            )

        # Create initial state.
        state = _AugmentedLagrangianState(
            vals=initial_vals,
            lagrange_multipliers=lagrange_multipliers,
            penalty_param=penalty_param,
            constraint_violation=initial_violation,
            initial_violation=initial_violation,
            outer_iteration=0,
            inner_summary=SolveSummary(
                iterations=jnp.array(0),
                cost_history=jnp.zeros(0),
                lambda_history=jnp.zeros(0),
                termination_criteria=jnp.array([False, False, False, False]),
                termination_deltas=jnp.zeros(3),
            ),
        )

        # Run outer loop.
        converged_absolute = False
        converged_relative = False
        for _ in range(self.config.max_iterations):
            state = self._step(problem, state)

            # Check convergence.
            converged_absolute = state.constraint_violation < self.config.tolerance_absolute
            converged_relative = (
                state.constraint_violation / state.initial_violation
                < self.config.tolerance_relative
            )

            if converged_absolute and converged_relative:
                if self.verbose:
                    jax_log(
                        "Augmented Lagrangian converged @ outer iteration {i}: violation={violation:.4e}",
                        i=state.outer_iteration,
                        violation=state.constraint_violation,
                    )
                break

        if self.verbose and not (converged_absolute and converged_relative):
            jax_log(
                "Augmented Lagrangian: max iterations reached. Final violation={violation:.4e}",
                violation=state.constraint_violation,
            )

        if return_summary:
            return state.vals, state.inner_summary
        else:
            return state.vals

    def _step(
        self,
        problem: AnalyzedLeastSquaresProblem,
        state: _AugmentedLagrangianState,
    ) -> _AugmentedLagrangianState:
        """Perform one outer iteration of the Augmented Lagrangian method.

        Args:
            problem: The analyzed least squares problem with constraints.
            state: Current state of the Augmented Lagrangian solver.

        Returns:
            Updated state after one outer iteration.
        """
        # Create augmented problem by converting constraints to costs.
        augmented_problem = self._create_augmented_problem(
            problem, state.lagrange_multipliers, state.penalty_param
        )

        # Solve inner unconstrained problem.
        # Adjust inner solver tolerance based on constraint violation.
        inner_tolerance = self.config.inner_tolerance_factor * state.constraint_violation
        inner_termination = jdc.replace(
            self.inner_solver.termination,
            cost_tolerance=jnp.maximum(inner_tolerance, 1e-8),
            gradient_tolerance=jnp.maximum(inner_tolerance, 1e-8),
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
        constraint_violation = jnp.linalg.norm(h_vals)

        # Update Lagrange multipliers: λ_new = λ_old + ρ * h(x)
        lagrange_multipliers_updated = state.lagrange_multipliers + state.penalty_param * h_vals

        # Update penalty parameter if constraint violation didn't decrease enough.
        violation_reduction = (
            (state.constraint_violation - constraint_violation) /
            (state.constraint_violation + 1e-10)
        )
        penalty_updated = jnp.where(
            violation_reduction < self.config.violation_reduction_threshold,
            jnp.minimum(
                state.penalty_param * self.config.penalty_factor,
                self.config.penalty_max,
            ),
            state.penalty_param,
        )

        if self.verbose:
            jax_log(
                " AL outer iter {i}: violation={violation:.4e}, penalty={penalty:.4e}, inner_iters={inner_iters}, reduction={reduction:.2%}",
                i=state.outer_iteration,
                violation=constraint_violation,
                penalty=penalty_updated,
                inner_iters=inner_summary.iterations,
                reduction=violation_reduction,
                ordered=True,
            )

        return _AugmentedLagrangianState(
            vals=vals_updated,
            lagrange_multipliers=lagrange_multipliers_updated,
            penalty_param=penalty_updated,
            constraint_violation=constraint_violation,
            initial_violation=state.initial_violation,
            outer_iteration=state.outer_iteration + 1,
            inner_summary=inner_summary,
        )

    def _create_augmented_problem(
        self,
        problem: AnalyzedLeastSquaresProblem,
        lagrange_multipliers: jax.Array,
        penalty_param: float | jax.Array,
    ) -> AnalyzedLeastSquaresProblem:
        """Create augmented problem by adding constraint residuals to costs.

        For each constraint h_i(x), we add a residual:
            r_i = sqrt(ρ) * (h_i(x) + λ_i / ρ)

        This converts the augmented Lagrangian:
            L(x, λ, ρ) = f(x) + λᵀh(x) + (ρ/2)||h(x)||²

        Into an unconstrained least squares problem:
            min ||[r_original; r_constraint]||²

        Args:
            problem: Original problem with constraints.
            lagrange_multipliers: Current Lagrange multipliers.
            penalty_param: Current penalty parameter.

        Returns:
            Augmented problem with constraints converted to costs.
        """
        from ._core import Cost, LeastSquaresProblem

        # Convert each stacked constraint to a Cost.
        # Track offset into lagrange_multipliers array.
        lambda_offset = 0
        constraint_costs = []

        for stacked_constraint in problem.stacked_constraints:
            # Dimension of each constraint instance.
            constraint_flat_dim = stacked_constraint.constraint_flat_dim

            # Get slice of multipliers for this constraint group.
            # Each constraint in the args has its own multiplier slice.
            num_constraints = stacked_constraint.args[0][0].id.shape[0] if not isinstance(
                stacked_constraint.args[0][0].id, int
            ) else 1

            total_constraint_dim = num_constraints * constraint_flat_dim

            lambda_slice = lagrange_multipliers[
                lambda_offset : lambda_offset + total_constraint_dim
            ]

            # Create augmented cost function.
            # We capture constraint, lambda_slice, and penalty_param in the closure.
            def make_augmented_cost(constraint, lambdas, rho):
                """Factory to create augmented cost with captured parameters."""

                def augmented_residual_fn(vals, *args):
                    """Compute augmented constraint residual: sqrt(ρ) * (h(x) + λ/ρ)"""
                    h_val = constraint.compute_constraint_flat(vals, *args)
                    return jnp.sqrt(rho) * (h_val + lambdas / rho)

                return Cost(
                    compute_residual=augmented_residual_fn,
                    args=constraint.args,
                )

            constraint_costs.append(
                make_augmented_cost(stacked_constraint, lambda_slice, penalty_param)
            )

            lambda_offset += total_constraint_dim

        # Extract variables from original problem.
        variables = []
        for var_type in problem.sorted_ids_from_var_type.keys():
            ids = problem.sorted_ids_from_var_type[var_type]
            for var_id in ids:
                variables.append(var_type(var_id))

        # Combine original costs and constraint costs, then re-analyze.
        # Note: This is inefficient (re-analysis every outer iteration), but correct.
        # Future optimization: cache and reuse sparsity structure.
        # Extract original costs from analyzed problem (we need the original Cost objects).
        original_costs = []
        cost_offset = 0
        for analyzed_cost, count in zip(problem.stacked_costs, problem.cost_counts):
            for i in range(count):
                # Extract individual cost from stacked analyzed cost.
                # We need to slice the args properly.
                def extract_cost_at_index(analyzed, idx):
                    """Extract a single Cost from a stacked _AnalyzedCost."""
                    # Get args for this specific index.
                    single_args = jax.tree.map(lambda x: x[idx], analyzed.args)
                    return Cost(
                        compute_residual=analyzed.compute_residual,
                        args=single_args,
                        jac_mode=analyzed.jac_mode,
                        jac_batch_size=analyzed.jac_batch_size,
                        jac_custom_fn=analyzed.jac_custom_fn,
                        jac_custom_with_cache_fn=analyzed.jac_custom_with_cache_fn,
                        name=analyzed.name,
                    )

                original_costs.append(extract_cost_at_index(analyzed_cost, i))

        augmented_problem_unanalyzed = LeastSquaresProblem(
            costs=original_costs + constraint_costs,
            variables=variables,
            constraints=(),  # No constraints in augmented problem
        )

        return augmented_problem_unanalyzed.analyze()
