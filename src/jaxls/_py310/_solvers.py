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

    tolerance_absolute: Any = 1e-6

    tolerance_relative: Any = 1e-4

    max_iterations: jdc.Static[Any] = 50

    inner_tolerance_factor: Any = 0.1

    violation_reduction_threshold: Any = 0.25


@jdc.pytree_dataclass
class _AugmentedLagrangianState:
    vals: Any

    lagrange_multipliers: Any

    penalty_param: Any

    constraint_violation: Any

    initial_violation: Any

    outer_iteration: Any

    inner_summary: Any

    constraint_violation_history: Any

    penalty_history: Any

    inner_iterations_count: Any


@jdc.pytree_dataclass
class AugmentedLagrangianSolver:
    config: Any

    inner_solver: Any

    verbose: jdc.Static[Any]

    def _extract_original_costs(self, problem: Any) -> Any:
        from ._core import Cost

        original_costs = []
        for stacked_cost, count in zip(problem.stacked_costs, problem.cost_counts):
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
                variables.append(var_type(int(var_id)))
        return variables

    def _analyze_augmented_problem(self, problem: Any, constraint_dim: Any) -> Any:
        from ._core import LeastSquaresProblem, create_augmented_constraint_cost

        constraint_costs = []
        constraint_dims = []
        constraint_is_inequality = []

        for i, (stacked_constraint, constraint_count) in enumerate(
            zip(problem.stacked_constraints, problem.constraint_counts)
        ):
            constraint_flat_dim = stacked_constraint.constraint_flat_dim
            total_dim = constraint_count * constraint_flat_dim
            constraint_dims.append(total_dim)
            constraint_is_inequality.append(
                stacked_constraint.constraint_type == "leq_zero"
            )

            cost = create_augmented_constraint_cost(stacked_constraint, i, total_dim)
            constraint_costs.append(cost)

        original_costs = self._extract_original_costs(problem)

        variables = self._extract_variables(problem)

        if self.verbose:
            jax_log("Pre-analyzing augmented problem structure (one-time cost)...")

        augmented = LeastSquaresProblem(
            costs=original_costs + constraint_costs,
            variables=variables,
        ).analyze()

        from ._core import AugmentedLagrangianParams

        lagrange_mult_arrays = tuple(jnp.zeros(dim) for dim in constraint_dims)
        al_params = AugmentedLagrangianParams(
            lagrange_multipliers=lagrange_mult_arrays,
            penalty_param=jnp.array(1.0),
        )

        augmented = jdc.replace(augmented, augmented_lagrangian_params=al_params)

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
        initial_violation = jnp.linalg.norm(h_vals)

        lagrange_multipliers = jnp.zeros(constraint_dim)

        if self.config.penalty_initial == "auto":
            residual_vector_result = problem.compute_residual_vector(initial_vals)
            if isinstance(residual_vector_result, tuple):
                residual_vector = residual_vector_result[0]
            else:
                residual_vector = residual_vector_result
            initial_cost = jnp.sum(residual_vector**2)
            penalty_param = initial_cost / 100.0
        else:
            penalty_param = self.config.penalty_initial

        augmented_structure = self._analyze_augmented_problem(problem, constraint_dim)

        if self.verbose:
            jax_log(
                "Augmented Lagrangian: initial violation={violation:.4e}, penalty={penalty:.4e}, constraint_dim={dim}",
                violation=initial_violation,
                penalty=penalty_param,
                dim=constraint_dim,
            )

        max_outer = self.config.max_iterations
        constraint_violation_history = jnp.zeros(max_outer)
        constraint_violation_history = constraint_violation_history.at[0].set(
            initial_violation
        )
        penalty_history = jnp.zeros(max_outer)
        penalty_history = penalty_history.at[0].set(penalty_param)
        inner_iterations_count = jnp.zeros(max_outer, dtype=jnp.int32)

        max_inner = self.inner_solver.termination.max_iterations
        initial_summary = SolveSummary(
            iterations=jnp.array(0, dtype=jnp.int32),
            cost_history=jnp.zeros(max_inner),
            lambda_history=jnp.zeros(max_inner),
            termination_criteria=jnp.array([False, False, False, False]),
            termination_deltas=jnp.zeros(3),
        )

        state = _AugmentedLagrangianState(
            vals=initial_vals,
            lagrange_multipliers=lagrange_multipliers,
            penalty_param=penalty_param,
            constraint_violation=initial_violation,
            initial_violation=initial_violation,
            outer_iteration=0,
            inner_summary=initial_summary,
            constraint_violation_history=constraint_violation_history,
            penalty_history=penalty_history,
            inner_iterations_count=inner_iterations_count,
        )

        def cond_fn(state: Any) -> Any:
            converged_absolute = (
                state.constraint_violation < self.config.tolerance_absolute
            )
            converged_relative = (
                state.constraint_violation / state.initial_violation
                < self.config.tolerance_relative
            )
            converged = converged_absolute & converged_relative
            under_max_iters = state.outer_iteration < self.config.max_iterations
            return ~converged & under_max_iters

        def body_fn(state: Any) -> Any:
            return self._step(problem, augmented_structure, state)

        state = jax.lax.while_loop(cond_fn, body_fn, state)

        converged_absolute = state.constraint_violation < self.config.tolerance_absolute
        converged_relative = (
            state.constraint_violation / state.initial_violation
            < self.config.tolerance_relative
        )

        if self.verbose:
            if converged_absolute and converged_relative:
                jax_log(
                    "Augmented Lagrangian converged @ outer iteration {i}: violation={violation:.4e}",
                    i=state.outer_iteration,
                    violation=state.constraint_violation,
                )
            else:
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
        problem: Any,
        augmented_structure: Any,
        state: Any,
    ) -> Any:
        from ._core import AugmentedLagrangianParams

        lambda_arrays = []
        lambda_offset = 0
        for (
            lambda_array
        ) in augmented_structure.augmented_lagrangian_params.lagrange_multipliers:
            dim = lambda_array.shape[0]
            lambda_arrays.append(
                state.lagrange_multipliers[lambda_offset : lambda_offset + dim]
            )
            lambda_offset += dim

        al_params = AugmentedLagrangianParams(
            lagrange_multipliers=tuple(lambda_arrays),
            penalty_param=state.penalty_param,
        )

        augmented_problem = jdc.replace(
            augmented_structure,
            augmented_lagrangian_params=al_params,
        )

        inner_tolerance = (
            self.config.inner_tolerance_factor * state.constraint_violation
        )
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

        h_vals = problem.compute_constraint_values(vals_updated)
        constraint_violation = jnp.linalg.norm(h_vals)

        lagrange_multipliers_updated = (
            state.lagrange_multipliers + state.penalty_param * h_vals
        )

        lambda_offset = 0
        lambda_arrays_projected = []
        for i, stacked_constraint in enumerate(problem.stacked_constraints):
            constraint_count = problem.constraint_counts[i]
            constraint_flat_dim = stacked_constraint.constraint_flat_dim
            total_dim = constraint_count * constraint_flat_dim

            lambda_slice = lagrange_multipliers_updated[
                lambda_offset : lambda_offset + total_dim
            ]

            if stacked_constraint.constraint_type == "leq_zero":
                lambda_slice = jnp.maximum(0.0, lambda_slice)

            lambda_arrays_projected.append(lambda_slice)
            lambda_offset += total_dim

        lagrange_multipliers_updated = jnp.concatenate(lambda_arrays_projected)

        violation_reduction = (state.constraint_violation - constraint_violation) / (
            state.constraint_violation + 1e-10
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

        next_idx = state.outer_iteration + 1

        return jdc.replace(
            state,
            vals=vals_updated,
            lagrange_multipliers=lagrange_multipliers_updated,
            penalty_param=penalty_updated,
            constraint_violation=constraint_violation,
            outer_iteration=state.outer_iteration + 1,
            inner_summary=inner_summary,
            constraint_violation_history=state.constraint_violation_history.at[
                next_idx
            ].set(constraint_violation),
            penalty_history=state.penalty_history.at[next_idx].set(penalty_updated),
            inner_iterations_count=state.inner_iterations_count.at[next_idx].set(
                inner_summary.iterations
            ),
        )
