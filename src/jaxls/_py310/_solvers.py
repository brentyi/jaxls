from __future__ import annotations
from typing import Any


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
class _NonlinearSolverState:
    vals: Any
    cost: Any
    residual_vector: Any
    summary: Any
    lambd: Any

    jac_cache: Any

    cg_state: Any

    jacobian_scaler: Any


@jdc.pytree_dataclass
class NonlinearSolver:
    linear_solver: jdc.Static[Any]
    trust_region: Any
    termination: Any
    conjugate_gradient_config: Any
    sparse_mode: jdc.Static[Any]
    verbose: jdc.Static[Any]

    @jdc.jit
    def solve(
        self,
        problem: Any,
        initial_vals: Any,
        return_summary: jdc.Static[Any] = False,
    ) -> Any:
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
                1,
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
        problem: Any,
        state: Any,
        first: Any,
    ) -> Any:
        if self.verbose:
            self._log_state(problem, state)

        A_blocksparse = problem._compute_jac_values(state.vals, state.jac_cache)

        if first:
            with jdc.copy_and_mutate(state, validate=False) as state:
                state.jacobian_scaler = (
                    1.0 / (1.0 + A_blocksparse.compute_column_norms()) * 0.0 + 1.0
                )
        assert state.jacobian_scaler is not None
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

        ATb = -AT_multiply(state.residual_vector)

        linear_state = None
        if (
            isinstance(self.linear_solver, ConjugateGradientConfig)
            or self.linear_solver == "conjugate_gradient"
        ):
            cg_config = (
                ConjugateGradientConfig()
                if self.linear_solver == "conjugate_gradient"
                else self.linear_solver
            )
            assert isinstance(state.cg_state, _ConjugateGradientState)
            local_delta, linear_state = cg_config._solve(
                problem,
                A_blocksparse,
                lambda vec: AT_multiply(A_multiply(vec)) + state.lambd * vec,
                ATb=ATb,
                prev_linear_state=state.cg_state,
            )
        elif self.linear_solver == "cholmod":
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

        if self.trust_region is None:
            with jdc.copy_and_mutate(state) as state_next:
                if linear_state is not None:
                    state_next.cg_state = linear_state

                state_next.vals = proposed_vals
                state_next.residual_vector = proposed_residual_vector
                state_next.cost = proposed_cost
                accept_flag = None

        else:
            step_quality = (proposed_cost - state.cost) / (
                jnp.sum(
                    (A_blocksparse.multiply(scaled_local_delta) + state.residual_vector)
                    ** 2
                )
                - state.cost
            )
            accept_flag = step_quality >= self.trust_region.step_quality_min

            with jdc.copy_and_mutate(state) as state_accept:
                if linear_state is not None:
                    state_accept.cg_state = linear_state

                state_accept.vals = proposed_vals
                state_accept.residual_vector = proposed_residual_vector
                state_accept.cost = proposed_cost
                state_accept.jac_cache = proposed_jac_cache
                state_accept.lambd = state.lambd / self.trust_region.lambda_factor

            with jdc.copy_and_mutate(state) as state_reject:
                state_reject.lambd = jnp.maximum(
                    self.trust_region.lambda_min,
                    jnp.minimum(
                        state.lambd * self.trust_region.lambda_factor,
                        self.trust_region.lambda_max,
                    ),
                )

            state_next = jax.tree.map(
                lambda x, y: x if (x is y) else jnp.where(accept_flag, x, y),
                state_accept,
                state_reject,
            )

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
    def _log_state(problem: Any, state: Any) -> Any:
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
    lambda_initial: Any = 5e-4
    lambda_factor: Any = 2.0
    lambda_min: Any = 1e-5
    lambda_max: Any = 1e10
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
        state_prev: Any,
        cost_updated: Any,
        tangent: Any,
        tangent_ordering: Any,
        ATb: Any,
        accept_flag: Any = None,
    ) -> Any:
        cost_absdelta = jnp.abs(cost_updated - state_prev.cost)
        cost_reldelta = cost_absdelta / state_prev.cost
        converged_cost = cost_reldelta < self.cost_tolerance

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

        param_delta = jnp.linalg.norm(jnp.abs(tangent)) / (
            jnp.linalg.norm(flat_vals) + self.parameter_tolerance
        )
        converged_parameters = param_delta < self.parameter_tolerance

        term_flags = jnp.array(
            [
                converged_cost,
                converged_gradient,
                converged_parameters,
                state_prev.summary.iterations >= (self.max_iterations - 1),
            ]
        )

        if accept_flag is not None:
            term_flags = term_flags.at[:3].set(
                jnp.logical_and(
                    term_flags[:3],
                    jnp.logical_or(accept_flag, cost_absdelta == 0.0),
                )
            )

        return term_flags, jnp.array([cost_reldelta, gradient_mag, param_delta])


@jdc.pytree_dataclass
class AugmentedLagrangianConfig:
    penalty_initial: Any = "auto"

    penalty_factor: Any = 10.0

    penalty_max: Any = 1e8

    penalty_min: Any = 1e-6

    tolerance_absolute: Any = 1e-6

    tolerance_relative: Any = 1e-4

    max_iterations: jdc.Static[Any] = 50

    violation_reduction_threshold: Any = 0.5

    lambda_min: Any = -1e8

    lambda_max: Any = 1e8


@jdc.pytree_dataclass
class _AugmentedLagrangianState:
    vals: Any

    lagrange_multipliers: Any

    penalty_params: Any

    snorm: Any

    snorm_prev: Any

    constraint_values_prev: Any

    constraint_violation: Any

    initial_snorm: Any

    outer_iteration: Any

    inner_summary: Any

    constraint_violation_history: Any

    penalty_history: Any

    inner_iterations_count: Any

    epsopk: Any


@jdc.pytree_dataclass
class AugmentedLagrangianSolver:
    config: Any

    inner_solver: Any

    verbose: jdc.Static[Any]

    def _extract_original_costs(self, problem: Any) -> Any:
        from ._core import Cost

        original_costs = []
        for stacked_cost in problem.stacked_costs:
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

    def _extract_variables(self, problem: Any) -> Any:
        variables = []
        for var_type, ids in problem.sorted_ids_from_var_type.items():
            for var_id in ids:
                variables.append(var_type(var_id))
        return variables

    def _analyze_augmented_problem(self, problem: Any, constraint_dim: Any) -> Any:
        from ._core import (
            AugmentedLagrangianParams,
            LeastSquaresProblem,
            create_augmented_constraint_cost,
        )

        constraint_dims = []
        constraint_costs = []

        for i, (stacked_constraint, constraint_count) in enumerate(
            zip(problem.stacked_constraints, problem.constraint_counts)
        ):
            constraint_flat_dim = stacked_constraint.residual_flat_dim
            total_dim = constraint_count * constraint_flat_dim
            constraint_dims.append(total_dim)

        lagrange_mult_arrays = tuple(jnp.zeros(dim) for dim in constraint_dims)
        penalty_param_arrays = tuple(jnp.ones(dim) for dim in constraint_dims)
        al_params = AugmentedLagrangianParams(
            lagrange_multipliers=lagrange_mult_arrays,
            penalty_params=penalty_param_arrays,
        )

        for i, (stacked_constraint, constraint_count) in enumerate(
            zip(problem.stacked_constraints, problem.constraint_counts)
        ):
            constraint_flat_dim = stacked_constraint.residual_flat_dim
            total_dim = constraint_count * constraint_flat_dim

            augmented_cost = create_augmented_constraint_cost(
                stacked_constraint, i, total_dim, al_params
            )
            constraint_costs.append(augmented_cost)

        original_costs = self._extract_original_costs(problem)
        variables = self._extract_variables(problem)

        if self.verbose:
            jax_log("Pre-analyzing augmented problem structure (one-time cost)...")

        augmented = LeastSquaresProblem(
            costs=original_costs + constraint_costs,
            variables=variables,
        ).analyze()

        return augmented

    def solve(
        self,
        problem: Any,
        initial_vals: Any,
        return_summary: jdc.Static[Any] = False,
    ) -> Any:
        assert len(problem.stacked_constraints) > 0, (
            "AugmentedLagrangianSolver requires constraints. "
            "Use NonlinearSolver for unconstrained problems."
        )

        h_vals = problem.compute_constraint_values(initial_vals)
        constraint_dim = len(h_vals)

        lagrange_multipliers = jnp.zeros(constraint_dim)

        initial_snorm, initial_csupn = self._compute_snorm_csupn(
            problem, h_vals, lagrange_multipliers
        )

        if self.config.penalty_initial == "auto":
            residual_vector_result = problem.compute_residual_vector(initial_vals)
            if isinstance(residual_vector_result, tuple):
                residual_vector = residual_vector_result[0]
            else:
                residual_vector = residual_vector_result
            initial_cost = jnp.sum(residual_vector**2)

            sum_c_squared = jnp.array(0.0)
            offset = 0
            for i, stacked_constraint in enumerate(problem.stacked_constraints):
                constraint_count = problem.constraint_counts[i]
                constraint_flat_dim = stacked_constraint.residual_flat_dim
                total_dim = constraint_count * constraint_flat_dim
                h_slice = h_vals[offset : offset + total_dim]

                if stacked_constraint.constraint_type == "leq_zero":
                    sum_c_squared = sum_c_squared + jnp.sum(
                        0.5 * jnp.maximum(0.0, h_slice) ** 2
                    )
                else:
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

        penalty_params = jnp.full(constraint_dim, penalty_initial)

        augmented_structure = self._analyze_augmented_problem(problem, constraint_dim)

        if self.verbose:
            jax_log(
                "Augmented Lagrangian: initial snorm={snorm:.4e}, csupn={csupn:.4e}, penalty={penalty:.4e}, constraint_dim={dim}",
                snorm=initial_snorm,
                csupn=initial_csupn,
                penalty=penalty_initial,
                dim=constraint_dim,
            )

        max_outer = self.config.max_iterations
        constraint_violation_history = jnp.zeros(max_outer)
        constraint_violation_history = constraint_violation_history.at[0].set(
            initial_csupn
        )
        penalty_history = jnp.zeros(max_outer)
        penalty_history = penalty_history.at[0].set(penalty_initial)
        inner_iterations_count = jnp.zeros(max_outer, dtype=jnp.int32)

        max_inner = self.inner_solver.termination.max_iterations
        initial_summary = SolveSummary(
            iterations=jnp.array(0, dtype=jnp.int32),
            cost_history=jnp.zeros(max_inner),
            lambda_history=jnp.zeros(max_inner),
            termination_criteria=jnp.array([False, False, False, False]),
            termination_deltas=jnp.zeros(3),
        )

        base_gradient_tol = self.inner_solver.termination.gradient_tolerance
        initial_epsopk = jnp.sqrt(base_gradient_tol)

        state = _AugmentedLagrangianState(
            vals=initial_vals,
            lagrange_multipliers=lagrange_multipliers,
            penalty_params=penalty_params,
            snorm=initial_snorm,
            snorm_prev=initial_snorm,
            constraint_values_prev=h_vals,
            constraint_violation=initial_csupn,
            initial_snorm=initial_snorm,
            outer_iteration=0,
            inner_summary=initial_summary,
            constraint_violation_history=constraint_violation_history,
            penalty_history=penalty_history,
            inner_iterations_count=inner_iterations_count,
            epsopk=initial_epsopk,
        )

        def cond_fn(state: Any) -> Any:
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

        def body_fn(state: Any) -> Any:
            return self._step(problem, augmented_structure, state)

        state = jax.lax.while_loop(cond_fn, body_fn, state)

        converged_absolute = (
            jnp.maximum(state.snorm, state.constraint_violation)
            < self.config.tolerance_absolute
        )
        converged_relative = (
            state.snorm / (state.initial_snorm + 1e-10) < self.config.tolerance_relative
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
        problem: Any,
        h_vals: Any,
        lagrange_multipliers: Any,
    ) -> Any:
        snorm_parts = []
        csupn_parts = []
        offset = 0

        for i, stacked_constraint in enumerate(problem.stacked_constraints):
            constraint_count = problem.constraint_counts[i]
            constraint_flat_dim = stacked_constraint.residual_flat_dim
            total_dim = constraint_count * constraint_flat_dim
            h_slice = h_vals[offset : offset + total_dim]
            lambda_slice = lagrange_multipliers[offset : offset + total_dim]

            if stacked_constraint.constraint_type == "leq_zero":
                comp = jnp.minimum(-h_slice, lambda_slice)
                snorm_parts.append(jnp.max(jnp.abs(comp)))
                csupn_parts.append(jnp.max(jnp.maximum(0.0, h_slice)))
            else:
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
        problem: Any,
        h_vals: Any,
    ) -> Any:
        violation_parts = []
        offset = 0

        for i, stacked_constraint in enumerate(problem.stacked_constraints):
            constraint_count = problem.constraint_counts[i]
            constraint_flat_dim = stacked_constraint.residual_flat_dim
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
        problem: Any,
        augmented_structure: Any,
        state: Any,
    ) -> Any:
        from ._core import AugmentedLagrangianParams

        constraint_dims = []
        for i, (stacked_constraint, constraint_count) in enumerate(
            zip(problem.stacked_constraints, problem.constraint_counts)
        ):
            constraint_flat_dim = stacked_constraint.residual_flat_dim
            total_dim = constraint_count * constraint_flat_dim
            constraint_dims.append(total_dim)

        lambda_arrays = []
        penalty_arrays = []
        offset = 0
        for dim in constraint_dims:
            lambda_arrays.append(state.lagrange_multipliers[offset : offset + dim])
            penalty_arrays.append(state.penalty_params[offset : offset + dim])
            offset += dim

        al_params = AugmentedLagrangianParams(
            lagrange_multipliers=tuple(lambda_arrays),
            penalty_params=tuple(penalty_arrays),
        )

        num_original_costs = len(problem.stacked_costs)
        updated_costs = list(augmented_structure.stacked_costs[:num_original_costs])

        constraint_idx = 0
        for cost in augmented_structure.stacked_costs[num_original_costs:]:
            (constraint_count,) = cost._get_batch_axes()

            def broadcast_to_batch(arr: Any) -> Any:
                return jnp.broadcast_to(arr, (constraint_count,) + arr.shape)

            al_params_broadcasted = AugmentedLagrangianParams(
                lagrange_multipliers=tuple(
                    broadcast_to_batch(arr) for arr in al_params.lagrange_multipliers
                ),
                penalty_params=tuple(
                    broadcast_to_batch(arr) for arr in al_params.penalty_params
                ),
            )
            updated_costs.append(jdc.replace(cost, al_params=al_params_broadcasted))
            constraint_idx += 1

        augmented_problem = jdc.replace(
            augmented_structure,
            stacked_costs=tuple(updated_costs),
        )

        inner_termination = jdc.replace(
            self.inner_solver.termination,
            cost_tolerance=state.epsopk,
            gradient_tolerance=state.epsopk,
        )
        inner_solver_updated = jdc.replace(
            self.inner_solver,
            termination=inner_termination,
        )

        vals_updated, inner_summary = inner_solver_updated.solve(
            augmented_problem, state.vals, return_summary=True
        )

        h_vals = problem.compute_constraint_values(vals_updated)

        lagrange_multipliers_updated = (
            state.lagrange_multipliers + state.penalty_params * h_vals
        )

        offset = 0
        lambda_arrays_projected = []
        for i, stacked_constraint in enumerate(problem.stacked_constraints):
            constraint_count = problem.constraint_counts[i]
            constraint_flat_dim = stacked_constraint.residual_flat_dim
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

        snorm_new, csupn_new = self._compute_snorm_csupn(
            problem, h_vals, lagrange_multipliers_updated
        )

        constraint_violation_per = self._compute_per_constraint_violation(
            problem, h_vals
        )
        constraint_violation_prev_per = self._compute_per_constraint_violation(
            problem, state.constraint_values_prev
        )

        insufficient_progress_per = constraint_violation_per > (
            self.config.violation_reduction_threshold * constraint_violation_prev_per
        )

        inner_hit_max_iters = inner_summary.termination_criteria[3]
        inner_made_progress = inner_summary.termination_criteria[0]
        rhorestart_needed = inner_hit_max_iters & ~inner_made_progress

        penalty_params_updated = jnp.where(
            insufficient_progress_per | rhorestart_needed,
            jnp.minimum(
                state.penalty_params * self.config.penalty_factor,
                self.config.penalty_max,
            ),
            state.penalty_params,
        )

        nlpsupn = inner_summary.termination_deltas[1]
        base_tol = self.inner_solver.termination.gradient_tolerance
        sqrt_base_tol = jnp.sqrt(base_tol)
        sqrt_constraint_tol = jnp.sqrt(self.config.tolerance_absolute)

        should_tighten = jnp.logical_and(
            snorm_new <= sqrt_constraint_tol,
            nlpsupn <= sqrt_base_tol,
        )
        epsopk_candidate = jnp.minimum(0.5 * nlpsupn, 0.1 * state.epsopk)
        epsopk_new = jnp.where(
            should_tighten,
            jnp.maximum(epsopk_candidate, base_tol),
            state.epsopk,
        )

        if self.verbose:
            jax_log(
                " AL outer iter {i}: snorm={snorm:.4e}, csupn={csupn:.4e}, max_rho={max_rho:.4e}, inner_iters={inner_iters}, epsopk={epsopk:.4e}",
                i=state.outer_iteration,
                snorm=snorm_new,
                csupn=csupn_new,
                max_rho=jnp.max(penalty_params_updated),
                inner_iters=inner_summary.iterations,
                epsopk=epsopk_new,
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
            epsopk=epsopk_new,
        )
