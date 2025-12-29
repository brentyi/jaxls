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

from ._augmented_lagrangian import (
    AugmentedLagrangianConfig,
    AugmentedLagrangianState,
    check_al_convergence,
    initialize_al_state,
    update_al_state,
    update_problem_al_params,
)
from ._sparse_matrices import BlockRowSparseMatrix, SparseCooMatrix, SparseCsrMatrix
from ._variables import VarTypeOrdering, VarValues
from .utils import jax_log

if TYPE_CHECKING:
    import sksparse.cholmod

    from ._problem import AnalyzedLeastSquaresProblem, _CostInfo


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

    # Factorize and solve.
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
    """History of non-augmented costs (l2_squared terms only, excludes constraint penalties)."""
    lambda_history: jax.Array


@jdc.pytree_dataclass
class _SolutionState:
    """State associated with a single solution point."""

    vals: VarValues
    cost_info: _CostInfo
    cg_state: _ConjugateGradientState | None


@jdc.pytree_dataclass
class _LmInnerState:
    """State for inner loop that tries different lambda values.

    For Levenberg-Marquardt, this loop searches for a good lambda. For
    Gauss-Newton (trust_region=None), the loop runs exactly once with lambda=0.
    """

    lambd: float | jax.Array
    accepted: jax.Array
    sol_proposed: _SolutionState
    local_delta: jax.Array
    summary: SolveSummary


@jdc.pytree_dataclass
class _LmOuterState:
    """State for outer loop. Each iteration corresponds to one accepted
    step."""

    solution: _SolutionState
    summary: SolveSummary
    lambd: float | jax.Array
    jacobian_scaler: jax.Array
    al_state: AugmentedLagrangianState | None  # AL state when constraints present.


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
    augmented_lagrangian: AugmentedLagrangianConfig | None = None
    """Configuration for Augmented Lagrangian method. Set when constraints are present."""

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
        cost_info = problem._compute_cost_info(vals)

        cost_history = jnp.zeros(self.termination.max_iterations)
        cost_history = cost_history.at[0].set(cost_info.cost_nonconstraint)
        lambda_history = jnp.zeros(self.termination.max_iterations)
        if self.trust_region is not None:
            lambda_history = lambda_history.at[0].set(self.trust_region.lambda_initial)

        # Initialize AL state if constraints are present.
        al_state: AugmentedLagrangianState | None = None
        if self.augmented_lagrangian is not None:
            al_state = initialize_al_state(
                problem,
                vals,
                self.augmented_lagrangian,
                verbose=self.verbose,
            )
            # Update problem with initial AL params.
            problem = update_problem_al_params(problem, al_state)
            # Recompute cost with updated AL params.
            cost_info = problem._compute_cost_info(vals)
            cost_history = cost_history.at[0].set(cost_info.cost_nonconstraint)

        state = _LmOuterState(
            solution=_SolutionState(
                vals=vals,
                cost_info=cost_info,
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
            ),
            summary=SolveSummary(
                iterations=jnp.array(0),
                cost_history=cost_history,
                lambda_history=lambda_history,
                termination_criteria=jnp.array([False, False, False]),
                termination_deltas=jnp.zeros(3),
            ),
            lambd=self.trust_region.lambda_initial
            if self.trust_region is not None
            else 0.0,
            jacobian_scaler=jnp.ones(problem._tangent_dim),
            al_state=al_state,
        )

        # Optimization.
        if self.termination.early_termination:

            def should_continue(state: _LmOuterState) -> jax.Array:
                # Basic checks: not NaN, within iteration limit.
                basic_checks = ~jnp.isnan(state.solution.cost_info.cost_total) & (
                    state.summary.iterations < self.termination.max_iterations
                )

                # For unconstrained: stop when termination criteria met or step rejected.
                if self.augmented_lagrangian is None:
                    return basic_checks & ~jnp.any(state.summary.termination_criteria)

                # For constrained: check AL convergence.
                assert state.al_state is not None
                al_abs, al_rel = check_al_convergence(
                    state.al_state, self.augmented_lagrangian
                )
                al_converged = al_abs & al_rel

                # For constrained problems, all termination criteria (cost, gradient,
                # parameter) should only trigger when AL has converged. This ensures
                # we don't stop when the original objective has converged but
                # constraints are not yet satisfied.
                # termination_criteria: [cost, gradient, parameter]
                lm_terminated = jnp.any(state.summary.termination_criteria)

                # Stop if:
                # - max_iters reached (regardless of AL), OR
                # - LM terminated AND AL converged
                should_stop = lm_terminated & al_converged
                return basic_checks & ~should_stop

            state = jax.lax.while_loop(
                cond_fun=should_continue,
                body_fun=lambda state: self.lm_outer_step(problem, state),
                init_val=state,
            )
        else:
            state = jax.lax.fori_loop(
                0,
                self.termination.max_iterations,
                body_fun=lambda _, state: self.lm_outer_step(problem, state),
                init_val=state,
            )
        if self.verbose:
            jax_log(
                "Terminated @ iteration #{i}: cost={cost:.4f} criteria={criteria}, term_deltas={cost_delta:.1e},{grad_mag:.1e},{param_delta:.1e}",
                i=state.summary.iterations,
                cost=state.solution.cost_info.cost_nonconstraint,
                criteria=state.summary.termination_criteria.astype(jnp.int32),
                cost_delta=state.summary.termination_deltas[0],
                grad_mag=state.summary.termination_deltas[1],
                param_delta=state.summary.termination_deltas[2],
            )

        if return_summary:
            return state.solution.vals, state.summary
        else:
            return state.solution.vals

    def lm_inner_step(
        self,
        problem: AnalyzedLeastSquaresProblem,
        inner_state: _LmInnerState,
        sol_prev: _SolutionState,
        jacobian_scaler: jax.Array,
        A_blocksparse: BlockRowSparseMatrix,
        A_multiply: Callable[[jax.Array], jax.Array],
        AT_multiply: Callable[[jax.Array], jax.Array],
        ATb: jax.Array,
    ) -> _LmInnerState:
        """Levenberg-Marquardt inner step. Tries one lambda value."""
        # Compute lambda for this iteration.
        # - For LM: multiply previous lambda by factor (initial state is pre-divided)
        # - For GN: keep lambda at 0
        if self.trust_region is not None:
            lambd = jnp.minimum(
                inner_state.lambd * self.trust_region.lambda_factor,
                self.trust_region.lambda_max,
            )
        else:
            lambd = inner_state.lambd  # stays at 0

        # Log optimizer state for debugging.
        if self.verbose:
            self._log_state(problem, sol_prev, inner_state.summary.iterations, lambd)

        # Solve the linear system.
        cg_state: _ConjugateGradientState | None = None
        if (
            isinstance(self.linear_solver, ConjugateGradientConfig)
            or self.linear_solver == "conjugate_gradient"
        ):
            # Use default CG config if specified as a string, otherwise use the provided config.
            cg_config = (
                ConjugateGradientConfig()
                if self.linear_solver == "conjugate_gradient"
                else self.linear_solver
            )
            assert isinstance(sol_prev.cg_state, _ConjugateGradientState)
            local_delta, cg_state = cg_config._solve(
                problem,
                A_blocksparse,
                # We could also use (lambd * ATA_diagonals * vec) for
                # scale-invariant damping. But this is hard to match with CHOLMOD.
                # Add 1e-5 regularization to match dense_cholesky behavior.
                lambda vec: AT_multiply(A_multiply(vec)) + (lambd + 1e-5) * vec,
                ATb=ATb,
                prev_linear_state=sol_prev.cg_state,
            )
        elif self.linear_solver == "cholmod":
            # Use CHOLMOD for direct solve.
            A_csr = SparseCsrMatrix(
                jnp.concatenate(
                    [
                        block_row.blocks_concat.flatten()
                        for block_row in A_blocksparse.block_rows
                    ],
                    axis=0,
                ),
                problem._jac_coords_csr,
            )
            local_delta = _cholmod_solve(A_csr, ATb, lambd=lambd)
        elif self.linear_solver == "dense_cholesky":
            A_dense = A_blocksparse.to_dense()
            ATA = A_dense.T @ A_dense
            diag_idx = jnp.arange(ATA.shape[0])
            ATA = ATA.at[diag_idx, diag_idx].add(lambd)
            cho_factor = jax.scipy.linalg.cho_factor(ATA)
            local_delta = jax.scipy.linalg.cho_solve(cho_factor, ATb)
        else:
            assert_never(self.linear_solver)

        scaled_local_delta = local_delta * jacobian_scaler
        proposed_vals = sol_prev.vals._retract(
            scaled_local_delta, problem._tangent_ordering
        )
        proposed_cost_info = problem._compute_cost_info(proposed_vals)

        # Compute step acceptance.
        # - For GN: always accept.
        # - For LM: accept if step quality meets threshold.
        if self.trust_region is None:
            accepted = jnp.array(True)
        else:
            cost_predicted = jnp.sum(
                (
                    A_blocksparse.multiply(scaled_local_delta)
                    + sol_prev.cost_info.residual_vector
                )
                ** 2
            )
            predicted_reduction = sol_prev.cost_info.cost_total - cost_predicted
            actual_reduction = (
                sol_prev.cost_info.cost_total - proposed_cost_info.cost_total
            )
            step_quality = actual_reduction / predicted_reduction
            accepted = ~jnp.isnan(proposed_cost_info.cost_total) & (
                step_quality >= self.trust_region.step_quality_min
            )

        iterations = inner_state.summary.iterations + 1
        term_criteria, term_deltas = self.termination._check_convergence(
            sol_prev,
            cost_nonconstraint_updated=proposed_cost_info.cost_nonconstraint,
            tangent=local_delta * jacobian_scaler,
            tangent_ordering=problem._tangent_ordering,
            ATb=ATb,
            iterations=iterations,
        )
        with jdc.copy_and_mutate(inner_state) as next:
            next.lambd = lambd
            next.accepted = accepted
            next.sol_proposed = _SolutionState(
                vals=proposed_vals,
                cost_info=proposed_cost_info,
                cg_state=cg_state,
            )
            next.local_delta = local_delta
            next.summary = SolveSummary(
                iterations=iterations,
                termination_criteria=term_criteria,
                termination_deltas=term_deltas,
                cost_history=next.summary.cost_history.at[iterations].set(
                    proposed_cost_info.cost_nonconstraint
                ),
                lambda_history=next.summary.lambda_history.at[iterations].set(lambd),
            )

        return next

    def lm_outer_step(
        self,
        problem: AnalyzedLeastSquaresProblem,
        state: _LmOuterState,
    ) -> _LmOuterState:
        """Levenberg-Marquardt outer step. Each outer step applies one parameter update."""
        sol_prev = state.solution

        # Update problem with current AL params if constraints present.
        if self.augmented_lagrangian is not None and state.al_state is not None:
            problem = update_problem_al_params(problem, state.al_state)

        # Get nonzero values of Jacobian.
        A_blocksparse = problem._compute_jac_values(
            sol_prev.vals, sol_prev.cost_info.jac_cache
        )

        # Compute Jacobian scaler on first iteration only. We use jnp.where to
        # avoid double JIT compilation (one trace for first=True, one for first=False).
        with jdc.copy_and_mutate(state, validate=False) as state:
            state.jacobian_scaler = jnp.where(
                state.summary.iterations == 0,
                1.0 / (1.0 + A_blocksparse.compute_column_norms()) + 1.0,
                state.jacobian_scaler,
            )
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
                values=jac_values, coords=problem._jac_coords_coo
            ).as_jax_bcoo()
            AT_coo = A_coo.transpose()
            A_multiply = lambda vec: A_coo @ vec
            AT_multiply = lambda vec: AT_coo @ vec
        elif self.sparse_mode == "csr":
            A_csr = SparseCsrMatrix(
                values=jac_values, coords=problem._jac_coords_csr
            ).as_jax_bcsr()
            A_multiply = lambda vec: A_csr @ vec
            AT_multiply_ = jax.linear_transpose(
                A_multiply, jnp.zeros((A_blocksparse.shape[1],))
            )
            AT_multiply = lambda vec: AT_multiply_(vec)[0]
        else:
            assert_never(self.sparse_mode)

        # Compute right-hand side of normal equation.
        ATb = -AT_multiply(sol_prev.cost_info.residual_vector)

        # Inner loop: search for a good lambda value.
        #
        # For Levenberg-Marquardt, we try increasing lambdas until the step is
        # accepted or lambda is maxed out. For Gauss-Newton (trust_region=None),
        # lambda stays at 0 and the step is always accepted, so the loop runs
        # exactly once.
        #
        # We use the "pre-divide" trick: initialize with lambd/factor so that
        # the first multiplication in lm_inner_step gives us the correct starting
        # lambda. For GN, lambda starts and stays at 0.
        if self.trust_region is not None:
            init_lambd = state.lambd / self.trust_region.lambda_factor
            lambda_max = self.trust_region.lambda_max
        else:
            init_lambd = jnp.zeros(())
            lambda_max = jnp.inf

        # Reset termination criteria for this outer step. The criteria from the
        # previous outer step should not prevent the inner loop from running.
        with jdc.copy_and_mutate(state.summary, validate=False) as init_summary:
            init_summary.termination_criteria = jnp.array([False, False, False])

        init_inner_state = _LmInnerState(
            lambd=init_lambd,
            accepted=jnp.array(False),
            # Dummy values - will be overwritten by first lm_inner_step call.
            sol_proposed=sol_prev,
            local_delta=jnp.zeros_like(ATb),
            summary=init_summary,
        )
        inner_state_final = jax.lax.while_loop(
            cond_fun=lambda s: (
                # Exit early if converged (cost, gradient, or parameter criteria).
                ~s.accepted
                & ~jnp.any(s.summary.termination_criteria)
                & (s.lambd < lambda_max)
                & (s.summary.iterations < self.termination.max_iterations)
            ),
            body_fun=lambda s: self.lm_inner_step(
                problem,
                s,
                sol_prev,
                state.jacobian_scaler,
                A_blocksparse,
                A_multiply,
                AT_multiply,
                ATb,
            ),
            init_val=init_inner_state,
        )

        # For LM, decrease lambda if step was good.
        if self.trust_region is not None:
            lambd_next = jnp.maximum(
                self.trust_region.lambda_min,
                jnp.where(
                    inner_state_final.accepted,
                    inner_state_final.lambd / self.trust_region.lambda_factor,
                    inner_state_final.lambd,
                ),
            )
        else:
            lambd_next = inner_state_final.lambd  # stays at 0

        # Build next state from inner loop result.
        with jdc.copy_and_mutate(state) as state_next:
            state_next.solution = jax.tree.map(
                lambda new, old: jnp.where(inner_state_final.accepted, new, old),
                inner_state_final.sol_proposed,
                sol_prev,
            )
            state_next.lambd = lambd_next

        # Debug: log step acceptance and ATb_norm.
        if self.verbose:
            jax_log(
                "     accepted={accepted} ATb_norm={atb_norm:.2e} cost_prev={cost_prev:.4f} cost_new={cost_new:.4f}",
                accepted=inner_state_final.accepted,
                atb_norm=jnp.linalg.norm(ATb),
                cost_prev=sol_prev.cost_info.cost_total,
                cost_new=inner_state_final.sol_proposed.cost_info.cost_total,
                ordered=True,
            )

        # Update AL state when inner problem has converged (gradient small or terminated).
        # Note: we update even if the step was rejected - if gradient is small, inner has converged.
        if self.augmented_lagrangian is not None and state_next.al_state is not None:
            should_update_al = (
                jnp.linalg.norm(ATb) < self.augmented_lagrangian.inner_solve_tolerance
            ) | jnp.any(inner_state_final.summary.termination_criteria)
            state_next = jax.lax.cond(
                should_update_al,
                lambda s: self._update_al_state_and_recompute(problem, s),
                lambda s: s,
                state_next,
            )

        # Copy summary from inner state (convergence and histories already updated).
        with jdc.copy_and_mutate(state_next) as state_next:
            state_next.summary = inner_state_final.summary
        return state_next

    def _update_al_state_and_recompute(
        self,
        problem: AnalyzedLeastSquaresProblem,
        state: _LmOuterState,
    ) -> _LmOuterState:
        """Update Augmented Lagrangian state and recompute cost.

        Args:
            problem: The analyzed problem with constraints.
            state: Current outer state.

        Returns:
            Updated outer state with new AL params and recomputed cost.
        """
        assert self.augmented_lagrangian is not None
        assert state.al_state is not None

        al_state_updated = update_al_state(
            problem,
            state.solution.vals,
            state.al_state,
            self.augmented_lagrangian,
            verbose=self.verbose,
        )

        problem_updated = update_problem_al_params(problem, al_state_updated)
        new_cost_info = problem_updated._compute_cost_info(state.solution.vals)

        with jdc.copy_and_mutate(state) as state_updated:
            state_updated.al_state = al_state_updated
            state_updated.solution.cost_info = new_cost_info
        return state_updated

    @staticmethod
    def _log_state(
        problem: AnalyzedLeastSquaresProblem,
        sol: _SolutionState,
        iterations: jax.Array,
        lambd: float | jax.Array,
    ) -> None:
        if sol.cg_state is None:
            jax_log(
                " step #{i}: cost={cost:.4f} lambd={lambd:.4f}",
                i=iterations,
                cost=sol.cost_info.cost_nonconstraint,
                lambd=lambd,
                ordered=True,
            )
        else:
            jax_log(
                " step #{i}: cost={cost:.4f} lambd={lambd:.4f} inexact_tol={inexact_tol:.1e}",
                i=iterations,
                cost=sol.cost_info.cost_nonconstraint,
                lambd=lambd,
                inexact_tol=sol.cg_state.eta,
                ordered=True,
            )
        residual_index = 0
        for f, count in zip(problem._stacked_costs, problem._cost_counts):
            stacked_dim = count * f.residual_flat_dim
            partial_cost = jnp.sum(
                sol.cost_info.residual_vector[
                    residual_index : residual_index + stacked_dim
                ]
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
    lambda_max: float | jax.Array = 1e6
    """Maximum damping factor. Only used for Levenberg-Marquardt."""
    step_quality_min: float | jax.Array = 1e-3
    """Minimum step quality for Levenberg-Marquardt. Only used for Levenberg-Marquardt."""


@jdc.pytree_dataclass
class TerminationConfig:
    # Termination criteria.
    max_iterations: jdc.Static[int] = 100
    """Maximum number of optimization steps. For constrained problems, this is
    the maximum iterations per inner solve (not total iterations)."""
    early_termination: jdc.Static[bool] = True
    """If set to `True`, terminate when any of the tolerances are met. If
    `False`, always run `max_iterations` steps."""
    cost_tolerance: float | jax.Array = 1e-5
    """We terminate if `|cost change| / cost < cost_tolerance`. For constrained
    problems, this acts as a floor for the adaptive inner solver tolerance."""
    gradient_tolerance: float | jax.Array = 1e-4
    """We terminate if `norm_inf(x - rplus(x, linear delta)) < gradient_tolerance`.
    For constrained problems, this acts as a floor for the adaptive inner solver
    tolerance."""
    gradient_tolerance_start_step: int | jax.Array = 10
    """When to start checking the gradient tolerance condition. Helps solve precision
    issues caused by inexact Newton steps."""
    parameter_tolerance: float | jax.Array = 1e-6
    """We terminate if `norm_2(linear delta) < (norm2(x) + parameter_tolerance) * parameter_tolerance`."""

    def _check_convergence(
        self,
        sol_prev: _SolutionState,
        cost_nonconstraint_updated: jax.Array,
        tangent: jax.Array,
        tangent_ordering: VarTypeOrdering,
        ATb: jax.Array,
        iterations: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Check for convergence!"""

        # Cost tolerance: use cost_nonconstraint for meaningful convergence check.
        # For constrained problems, the total cost changes with each AL update,
        # but cost_nonconstraint (original objective) provides stable convergence.
        cost_reldelta = (
            jnp.abs(cost_nonconstraint_updated - sol_prev.cost_info.cost_nonconstraint)
            / sol_prev.cost_info.cost_nonconstraint
        )
        converged_cost = cost_reldelta < self.cost_tolerance

        # Gradient tolerance: infinity norm of the gradient step.
        flat_vals = jax.flatten_util.ravel_pytree(sol_prev.vals)[0]
        gradient_mag = jnp.max(
            jnp.abs(
                flat_vals
                - jax.flatten_util.ravel_pytree(
                    sol_prev.vals._retract(ATb, tangent_ordering)
                )[0]
            )
        )
        converged_gradient = jnp.where(
            iterations >= self.gradient_tolerance_start_step,
            gradient_mag < self.gradient_tolerance,
            False,
        )

        # Parameter tolerance.
        param_delta = jnp.linalg.norm(jnp.abs(tangent)) / (
            jnp.linalg.norm(flat_vals) + self.parameter_tolerance
        )
        converged_parameters = param_delta < self.parameter_tolerance

        # Check termination flags. We'll terminate if any of the conditions are met.
        term_flags = jnp.array(
            [converged_cost, converged_gradient, converged_parameters]
        )
        return term_flags, jnp.array([cost_reldelta, gradient_mag, param_delta])
