from __future__ import annotations

from typing import Any

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp

from ._solvers import SolveSummary
from .utils import jax_log


@jdc.pytree_dataclass
class AugmentedLagrangianParams:
    lagrange_multipliers: Any

    penalty_params: Any


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
        from ._core import LeastSquaresProblem

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
            with jdc.copy_and_mutate(cost) as cost_copy:
                cost_copy.al_params = al_params_broadcasted
            updated_costs.append(cost_copy)
            constraint_idx += 1

        with jdc.copy_and_mutate(augmented_structure) as augmented_problem:
            augmented_problem.stacked_costs = tuple(updated_costs)

        with jdc.copy_and_mutate(self.inner_solver.termination) as inner_termination:
            inner_termination.cost_tolerance = jnp.maximum(
                state.epsopk, self.inner_solver.termination.cost_tolerance
            )
            inner_termination.gradient_tolerance = jnp.maximum(
                state.epsopk, self.inner_solver.termination.gradient_tolerance
            )
        with jdc.copy_and_mutate(self.inner_solver) as inner_solver_updated:
            inner_solver_updated.termination = inner_termination

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
        with jdc.copy_and_mutate(state) as state_updated:
            state_updated.vals = vals_updated
            state_updated.lagrange_multipliers = lagrange_multipliers_updated
            state_updated.penalty_params = penalty_params_updated
            state_updated.snorm = snorm_new
            state_updated.snorm_prev = state.snorm
            state_updated.constraint_values_prev = h_vals
            state_updated.constraint_violation = csupn_new
            state_updated.outer_iteration = state.outer_iteration + 1
            state_updated.inner_summary = inner_summary
            state_updated.constraint_violation_history = (
                state.constraint_violation_history.at[next_idx].set(csupn_new)
            )
            state_updated.penalty_history = state.penalty_history.at[next_idx].set(
                jnp.max(penalty_params_updated)
            )
            state_updated.inner_iterations_count = state.inner_iterations_count.at[
                next_idx
            ].set(inner_summary.iterations)
            state_updated.epsopk = epsopk_new
        return state_updated


def create_augmented_constraint_cost(
    analyzed_constraint: Any,
    constraint_index: Any,
    total_dim: Any,
    al_params: Any,
) -> Any:
    from ._core import _AnalyzedCost

    is_inequality = analyzed_constraint.constraint_type == "leq_zero"
    constraint_flat_dim = analyzed_constraint.residual_flat_dim

    def augmented_residual_fn(
        vals: Any,
        *args_with_index_and_params,
    ) -> Any:
        args = args_with_index_and_params[:-2]
        instance_index = args_with_index_and_params[-2]
        al_params_inner: Any = args_with_index_and_params[-1]

        constraint_val = analyzed_constraint.compute_residual(vals, *args).flatten()

        start_idx = instance_index * constraint_flat_dim
        lambdas = jax.lax.dynamic_slice(
            al_params_inner.lagrange_multipliers[constraint_index],
            (start_idx,),
            (constraint_flat_dim,),
        )
        rho = jax.lax.dynamic_slice(
            al_params_inner.penalty_params[constraint_index],
            (start_idx,),
            (constraint_flat_dim,),
        )

        if is_inequality:
            return jnp.sqrt(rho) * jnp.maximum(0.0, constraint_val + lambdas / rho)
        else:
            return jnp.sqrt(rho) * (constraint_val + lambdas / rho)

    constraint_count = total_dim // constraint_flat_dim

    instance_indices = jnp.arange(constraint_count)

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

    return _AnalyzedCost(
        compute_residual=augmented_residual_fn,
        args=(*analyzed_constraint.args, instance_indices),
        jac_mode="auto",
        jac_batch_size=None,
        jac_custom_fn=None,
        jac_custom_with_cache_fn=None,
        name=f"augmented_{analyzed_constraint._get_name()}",
        num_variables=analyzed_constraint.num_variables,
        sorted_ids_from_var_type=analyzed_constraint.sorted_ids_from_var_type,
        residual_flat_dim=constraint_flat_dim,
        constraint_type=analyzed_constraint.constraint_type,
        al_params=al_params_broadcasted,
    )
