from __future__ import annotations
from typing import Any

from typing_extensions import assert_never

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
    check_al_convergence,
    initialize_al_state,
    update_al_state,
    update_problem_al_params,
)
from ._sparse_matrices import SparseCooMatrix, SparseCsrMatrix
from .utils import jax_log


_cholmod_analyze_cache: Any = {}


def _cholmod_solve(A: Any, ATb: Any, lambd: Any) -> Any:
    return jax.pure_callback(
        _cholmod_solve_on_host,
        ATb,
        A,
        ATb,
        lambd,
        vmap_method="sequential",
    )


def _cholmod_solve_on_host(
    A: Any,
    ATb: Any,
    lambd: Any,
) -> Any:
    import sksparse.cholmod

    A_T_scipy = scipy.sparse.csc_matrix(
        (A.values, A.coords.indices, A.coords.indptr), shape=A.coords.shape[::-1]
    )

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

    cost = cost.cholesky_AAt(
        A_T_scipy,
        beta=lambd + 1e-5,
    )
    return cost.solve_A(ATb)


@jdc.pytree_dataclass
class _ConjugateGradientState:
    ATb_norm_prev: Any
    eta: Any


@jdc.pytree_dataclass
class ConjugateGradientConfig:
    tolerance_min: Any = 1e-7
    tolerance_max: Any = 1e-2

    eisenstat_walker_gamma: Any = 0.9
    eisenstat_walker_alpha: Any = 2.0

    preconditioner: jdc.Static[Any] = "block_jacobi"

    def _solve(
        self,
        problem: Any,
        A_blocksparse: Any,
        ATA_multiply: Any,
        ATb: Any,
        prev_linear_state: Any,
    ) -> Any:
        assert len(ATb.shape) == 1, "ATb should be 1D!"

        if self.preconditioner == "block_jacobi":
            preconditioner = make_block_jacobi_precoditioner(problem, A_blocksparse)
        elif self.preconditioner == "point_jacobi":
            preconditioner = make_point_jacobi_precoditioner(A_blocksparse)
        elif self.preconditioner is None:
            preconditioner = lambda x: x
        else:
            assert_never(self.preconditioner)

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

        initial_x = jnp.zeros(ATb.shape)
        solution_values, _ = jax.scipy.sparse.linalg.cg(
            A=ATA_multiply,
            b=ATb,
            x0=initial_x,
            maxiter=len(initial_x),
            tol=current_eta,
            M=preconditioner,
        )
        return solution_values, _ConjugateGradientState(
            ATb_norm_prev=ATb_norm, eta=current_eta
        )


@jdc.pytree_dataclass
class SolveSummary:
    iterations: Any
    termination_criteria: Any
    termination_deltas: Any
    cost_history: Any
    lambda_history: Any


@jdc.pytree_dataclass
class _SolutionState:
    vals: Any
    cost_info: Any
    cg_state: Any


@jdc.pytree_dataclass
class _LmInnerState:
    lambd: Any
    accepted: Any
    sol_proposed: Any
    local_delta: Any
    summary: Any


@jdc.pytree_dataclass
class _LmOuterState:
    solution: Any
    summary: Any
    lambd: Any
    jacobian_scaler: Any
    al_state: Any


@jdc.pytree_dataclass
class NonlinearSolver:
    linear_solver: jdc.Static[Any]
    trust_region: Any
    termination: Any
    conjugate_gradient_config: Any
    sparse_mode: jdc.Static[Any]
    verbose: jdc.Static[Any]
    augmented_lagrangian: Any = None

    @jdc.jit
    def solve(
        self,
        problem: Any,
        initial_vals: Any,
        return_summary: jdc.Static[Any] = False,
    ) -> Any:
        vals = initial_vals
        cost_info = problem._compute_cost_info(vals)

        cost_history = jnp.zeros(self.termination.max_iterations)
        cost_history = cost_history.at[0].set(cost_info.cost_nonconstraint)
        lambda_history = jnp.zeros(self.termination.max_iterations)
        if self.trust_region is not None:
            lambda_history = lambda_history.at[0].set(self.trust_region.lambda_initial)

        al_state: Any = None
        if self.augmented_lagrangian is not None:
            al_state = initialize_al_state(
                problem,
                vals,
                self.augmented_lagrangian,
                verbose=self.verbose,
            )

            problem = update_problem_al_params(problem, al_state)

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

        if self.termination.early_termination:

            def should_continue(state: Any) -> Any:
                basic_checks = ~jnp.isnan(state.solution.cost_info.cost_total) & (
                    state.summary.iterations < self.termination.max_iterations
                )

                if self.augmented_lagrangian is None:
                    return basic_checks & ~jnp.any(state.summary.termination_criteria)

                assert state.al_state is not None
                al_abs, al_rel = check_al_convergence(
                    state.al_state, self.augmented_lagrangian
                )
                al_converged = al_abs & al_rel

                lm_terminated = jnp.any(state.summary.termination_criteria)

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
        problem: Any,
        inner_state: Any,
        sol_prev: Any,
        jacobian_scaler: Any,
        A_blocksparse: Any,
        A_multiply: Any,
        AT_multiply: Any,
        ATb: Any,
    ) -> Any:
        if self.trust_region is not None:
            lambd = jnp.minimum(
                inner_state.lambd * self.trust_region.lambda_factor,
                self.trust_region.lambda_max,
            )
        else:
            lambd = inner_state.lambd

        if self.verbose:
            self._log_state(problem, sol_prev, inner_state.summary.iterations, lambd)

        cg_state: Any = None
        if (
            isinstance(self.linear_solver, ConjugateGradientConfig)
            or self.linear_solver == "conjugate_gradient"
        ):
            cg_config = (
                ConjugateGradientConfig()
                if self.linear_solver == "conjugate_gradient"
                else self.linear_solver
            )
            assert isinstance(sol_prev.cg_state, _ConjugateGradientState)
            local_delta, cg_state = cg_config._solve(
                problem,
                A_blocksparse,
                lambda vec: AT_multiply(A_multiply(vec)) + (lambd + 1e-5) * vec,
                ATb=ATb,
                prev_linear_state=sol_prev.cg_state,
            )
        elif self.linear_solver == "cholmod":
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
        problem: Any,
        state: Any,
    ) -> Any:
        sol_prev = state.solution

        if self.augmented_lagrangian is not None and state.al_state is not None:
            problem = update_problem_al_params(problem, state.al_state)

        A_blocksparse = problem._compute_jac_values(
            sol_prev.vals, sol_prev.cost_info.jac_cache
        )

        with jdc.copy_and_mutate(state, validate=False) as state:
            state.jacobian_scaler = jnp.where(
                state.summary.iterations == 0,
                1.0 / (1.0 + A_blocksparse.compute_column_norms()) + 1.0,
                state.jacobian_scaler,
            )
        A_blocksparse = A_blocksparse.scale_columns(state.jacobian_scaler)

        jac_values = jnp.concatenate(
            [
                block_row.blocks_concat.flatten()
                for block_row in A_blocksparse.block_rows
            ],
            axis=0,
        )

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

        ATb = -AT_multiply(sol_prev.cost_info.residual_vector)

        if self.trust_region is not None:
            init_lambd = state.lambd / self.trust_region.lambda_factor
            lambda_max = self.trust_region.lambda_max
        else:
            init_lambd = jnp.zeros(())
            lambda_max = jnp.inf

        with jdc.copy_and_mutate(state.summary, validate=False) as init_summary:
            init_summary.termination_criteria = jnp.array([False, False, False])

        init_inner_state = _LmInnerState(
            lambd=init_lambd,
            accepted=jnp.array(False),
            sol_proposed=sol_prev,
            local_delta=jnp.zeros_like(ATb),
            summary=init_summary,
        )
        inner_state_final = jax.lax.while_loop(
            cond_fun=lambda s: (
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
            lambd_next = inner_state_final.lambd

        with jdc.copy_and_mutate(state) as state_next:
            state_next.solution = jax.tree.map(
                lambda new, old: jnp.where(inner_state_final.accepted, new, old),
                inner_state_final.sol_proposed,
                sol_prev,
            )
            state_next.lambd = lambd_next

        if self.verbose:
            jax_log(
                "     accepted={accepted} ATb_norm={atb_norm:.2e} cost_prev={cost_prev:.4f} cost_new={cost_new:.4f}",
                accepted=inner_state_final.accepted,
                atb_norm=jnp.linalg.norm(ATb),
                cost_prev=sol_prev.cost_info.cost_total,
                cost_new=inner_state_final.sol_proposed.cost_info.cost_total,
                ordered=True,
            )

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

        with jdc.copy_and_mutate(state_next) as state_next:
            state_next.summary = inner_state_final.summary
        return state_next

    def _update_al_state_and_recompute(
        self,
        problem: Any,
        state: Any,
    ) -> Any:
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
        problem: Any,
        sol: Any,
        iterations: Any,
        lambd: Any,
    ) -> Any:
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
    lambda_initial: Any = 5e-4
    lambda_factor: Any = 2.0
    lambda_min: Any = 1e-5
    lambda_max: Any = 1e6
    step_quality_min: Any = 1e-3


@jdc.pytree_dataclass
class TerminationConfig:
    max_iterations: jdc.Static[Any] = 100
    early_termination: jdc.Static[Any] = True
    cost_tolerance: Any = 1e-5
    gradient_tolerance: Any = 1e-4
    gradient_tolerance_start_step: Any = 10
    parameter_tolerance: Any = 1e-6

    def _check_convergence(
        self,
        sol_prev: Any,
        cost_nonconstraint_updated: Any,
        tangent: Any,
        tangent_ordering: Any,
        ATb: Any,
        iterations: Any,
    ) -> Any:
        cost_reldelta = (
            jnp.abs(cost_nonconstraint_updated - sol_prev.cost_info.cost_nonconstraint)
            / sol_prev.cost_info.cost_nonconstraint
        )
        converged_cost = cost_reldelta < self.cost_tolerance

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

        param_delta = jnp.linalg.norm(jnp.abs(tangent)) / (
            jnp.linalg.norm(flat_vals) + self.parameter_tolerance
        )
        converged_parameters = param_delta < self.parameter_tolerance

        term_flags = jnp.array(
            [converged_cost, converged_gradient, converged_parameters]
        )
        return term_flags, jnp.array([cost_reldelta, gradient_mag, param_delta])
