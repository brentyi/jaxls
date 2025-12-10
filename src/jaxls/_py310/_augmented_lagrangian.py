from __future__ import annotations
from typing import Any

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp

from ._core import AugmentedLagrangianParams
from ._solvers import SolveSummary
from .utils import jax_log


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

    constraint_values_prev: Any

    snorm: Any

    snorm_prev: Any

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

    def solve(
        self,
        problem: Any,
        initial_vals: Any,
        return_summary: jdc.Static[Any] = False,
    ) -> Any:
        constraint_costs = [
            cost for cost in problem.stacked_costs if cost.kind != "l2_squared"
        ]
        assert len(constraint_costs) > 0, (
            "AugmentedLagrangianSolver requires constraints (mode != 'cost'). "
            "Use NonlinearSolver for unconstrained problems."
        )

        h_vals = problem._compute_constraint_values(initial_vals)

        lagrange_multipliers = tuple(jnp.zeros_like(h) for h in h_vals)

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
            for cost, h_group in zip(constraint_costs, h_vals):
                if cost.kind in ("constraint_leq_zero", "constraint_geq_zero"):
                    sum_c_squared = sum_c_squared + jnp.sum(
                        0.5 * jnp.maximum(0.0, h_group) ** 2
                    )
                else:
                    sum_c_squared = sum_c_squared + jnp.sum(0.5 * h_group**2)

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

        penalty_params = tuple(jnp.full_like(h, penalty_initial) for h in h_vals)

        constraint_dim = sum(h.size for h in h_vals)
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
            constraint_values_prev=h_vals,
            snorm=initial_snorm,
            snorm_prev=initial_snorm,
            constraint_violation=initial_csupn,
            initial_snorm=initial_snorm,
            outer_iteration=0,
            inner_summary=initial_summary,
            constraint_violation_history=constraint_violation_history,
            penalty_history=penalty_history,
            inner_iterations_count=inner_iterations_count,
            epsopk=initial_epsopk,
        )

        def cond_fun(state: Any) -> Any:
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

        state = jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=lambda state: self._step(problem, state),
            init_val=state,
        )

        converged_absolute = (
            jnp.maximum(state.snorm, state.constraint_violation)
            < self.config.tolerance_absolute
        )
        converged_relative = (
            state.snorm / (state.initial_snorm + 1e-10) < self.config.tolerance_relative
        )

        if self.verbose:
            jax_log(
                "Augmented Lagrangian finished @ outer iteration {i}: converged_absolute={abs}, converged_relative={rel}, snorm={snorm:.4e}, csupn={csupn:.4e}",
                i=state.outer_iteration,
                abs=converged_absolute,
                rel=converged_relative,
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

        constraint_costs = [
            cost for cost in problem.stacked_costs if cost.kind != "l2_squared"
        ]

        for cost, h_group, lambda_group in zip(
            constraint_costs, h_vals, lagrange_multipliers
        ):
            if cost.kind in ("constraint_leq_zero", "constraint_geq_zero"):
                comp = jnp.minimum(-h_group, lambda_group)
                snorm_parts.append(jnp.max(jnp.abs(comp)))
                csupn_parts.append(jnp.max(jnp.maximum(0.0, h_group)))
            else:
                snorm_parts.append(jnp.max(jnp.abs(h_group)))
                csupn_parts.append(jnp.max(jnp.abs(h_group)))

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

        constraint_costs = [
            cost for cost in problem.stacked_costs if cost.kind != "l2_squared"
        ]

        for cost, h_group in zip(constraint_costs, h_vals):
            if cost.kind in ("constraint_leq_zero", "constraint_geq_zero"):
                violation_parts.append(jnp.maximum(0.0, h_group))
            else:
                violation_parts.append(jnp.abs(h_group))

        return tuple(violation_parts)

    def _step(
        self,
        problem: Any,
        state: Any,
    ) -> Any:
        updated_costs = []
        constraint_group_idx = 0

        for cost in problem.stacked_costs:
            if cost.kind == "l2_squared":
                updated_costs.append(cost)
            else:
                current_al_params: Any = cost.args[0]

                (constraint_count,) = cost._get_batch_axes()
                constraint_flat_dim = cost.residual_flat_dim

                lambda_flat = state.lagrange_multipliers[constraint_group_idx]
                penalty_flat = state.penalty_params[constraint_group_idx]

                lambda_reshaped = lambda_flat.reshape(
                    constraint_count, constraint_flat_dim
                )
                penalty_reshaped = penalty_flat.reshape(
                    constraint_count, constraint_flat_dim
                )

                al_params_updated = AugmentedLagrangianParams(
                    lagrange_multipliers=lambda_reshaped,
                    penalty_params=penalty_reshaped,
                    original_args=current_al_params.original_args,
                    constraint_index=current_al_params.constraint_index,
                )

                with jdc.copy_and_mutate(cost) as cost_copy:
                    cost_copy.args = (al_params_updated,)
                updated_costs.append(cost_copy)
                constraint_group_idx += 1

        with jdc.copy_and_mutate(problem) as updated_problem:
            updated_problem.stacked_costs = tuple(updated_costs)

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
            updated_problem, state.vals, return_summary=True
        )

        h_vals = problem._compute_constraint_values(vals_updated)

        constraint_costs = [
            cost for cost in problem.stacked_costs if cost.kind != "l2_squared"
        ]
        lagrange_multipliers_updated = []
        for cost, lambda_group, penalty_group, h_group in zip(
            constraint_costs,
            state.lagrange_multipliers,
            state.penalty_params,
            h_vals,
        ):
            lambda_new = lambda_group + penalty_group * h_group
            if cost.kind in ("constraint_leq_zero", "constraint_geq_zero"):
                lambda_new = jnp.maximum(0.0, lambda_new)
            lambda_new = jnp.clip(
                lambda_new, self.config.lambda_min, self.config.lambda_max
            )
            lagrange_multipliers_updated.append(lambda_new)
        lagrange_multipliers_updated = tuple(lagrange_multipliers_updated)

        snorm_new, csupn_new = self._compute_snorm_csupn(
            problem, h_vals, lagrange_multipliers_updated
        )

        constraint_violation_per = self._compute_per_constraint_violation(
            problem, h_vals
        )
        constraint_violation_prev_per = self._compute_per_constraint_violation(
            problem, state.constraint_values_prev
        )

        inner_hit_max_iters = inner_summary.termination_criteria[3]
        inner_made_progress = inner_summary.termination_criteria[0]
        rhorestart_needed = inner_hit_max_iters & ~inner_made_progress

        penalty_params_updated = []
        for violation, violation_prev, penalty_group in zip(
            constraint_violation_per,
            constraint_violation_prev_per,
            state.penalty_params,
        ):
            insufficient_progress = violation > (
                self.config.violation_reduction_threshold * violation_prev
            )
            penalty_new = jnp.where(
                insufficient_progress | rhorestart_needed,
                jnp.minimum(
                    penalty_group * self.config.penalty_factor,
                    self.config.penalty_max,
                ),
                penalty_group,
            )
            penalty_params_updated.append(penalty_new)
        penalty_params_updated = tuple(penalty_params_updated)

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

        max_penalty = jnp.max(jnp.array([jnp.max(p) for p in penalty_params_updated]))

        if self.verbose:
            jax_log(
                " AL outer iter {i}: snorm={snorm:.4e}, csupn={csupn:.4e}, max_rho={max_rho:.4e}, inner_iters={inner_iters}, epsopk={epsopk:.4e}",
                i=state.outer_iteration,
                snorm=snorm_new,
                csupn=csupn_new,
                max_rho=max_penalty,
                inner_iters=inner_summary.iterations,
                epsopk=epsopk_new,
                ordered=True,
            )

        next_idx = state.outer_iteration + 1
        with jdc.copy_and_mutate(state) as state_updated:
            state_updated.vals = vals_updated
            state_updated.lagrange_multipliers = lagrange_multipliers_updated
            state_updated.penalty_params = penalty_params_updated
            state_updated.constraint_values_prev = h_vals
            state_updated.snorm = snorm_new
            state_updated.snorm_prev = state.snorm
            state_updated.constraint_violation = csupn_new
            state_updated.outer_iteration = state.outer_iteration + 1
            state_updated.inner_summary = inner_summary
            state_updated.constraint_violation_history = (
                state.constraint_violation_history.at[next_idx].set(csupn_new)
            )
            state_updated.penalty_history = state.penalty_history.at[next_idx].set(
                max_penalty
            )
            state_updated.inner_iterations_count = state.inner_iterations_count.at[
                next_idx
            ].set(inner_summary.iterations)
            state_updated.epsopk = epsopk_new
        return state_updated
