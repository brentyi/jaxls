from __future__ import annotations
from typing import Any

import jax_dataclasses as jdc
from jax import numpy as jnp
from jax.nn import relu

from .utils import jax_log


@jdc.pytree_dataclass
class AugmentedLagrangianConfig:
    penalty_factor: Any = 4.0

    penalty_max: Any = 1e7

    penalty_min: Any = 1e-6

    penalty_initial: Any = None

    tolerance_absolute: Any = 1e-5

    tolerance_relative: Any = 1e-4

    violation_reduction_threshold: Any = 0.5

    lambda_min: Any = -1e7

    lambda_max: Any = 1e7

    inner_solve_tolerance: Any = 1e-2


@jdc.pytree_dataclass
class AugmentedLagrangianState:
    lagrange_multipliers: Any

    penalty_params: Any

    constraint_values_prev: Any

    is_inequality: jdc.Static[Any]

    snorm: Any

    constraint_violation: Any

    snorm_initial: Any


def _compute_snorm_csupn(
    h_vals: Any,
    lagrange_multipliers: Any,
    is_inequality: Any,
) -> Any:
    snorm_parts = []
    csupn_parts = []

    for h_group, lambda_group, is_ineq in zip(
        h_vals, lagrange_multipliers, is_inequality
    ):
        if is_ineq:
            comp = jnp.minimum(-h_group, lambda_group)
            snorm_parts.append(jnp.max(jnp.abs(comp)))
            csupn_parts.append(jnp.max(relu(h_group)))
        else:
            max_abs_h = jnp.max(jnp.abs(h_group))
            snorm_parts.append(max_abs_h)
            csupn_parts.append(max_abs_h)

    if len(snorm_parts) == 0:
        return jnp.array(0.0), jnp.array(0.0)

    snorm = jnp.max(jnp.array(snorm_parts))
    csupn = jnp.max(jnp.array(csupn_parts))
    return snorm, csupn


def initialize_al_state(
    problem: Any,
    initial_vals: Any,
    config: Any,
    verbose: Any = False,
) -> Any:
    is_inequality = tuple(
        cost.kind in ("constraint_leq_zero", "constraint_geq_zero")
        for cost in problem._stacked_costs
        if cost.kind != "l2_squared"
    )
    assert len(is_inequality) > 0, (
        "initialize_al_state requires constraints (kind != 'l2_squared')."
    )

    h_vals = problem._compute_constraint_values(initial_vals)
    lagrange_multipliers = tuple(jnp.zeros_like(h) for h in h_vals)
    initial_snorm, initial_csupn = _compute_snorm_csupn(
        h_vals, lagrange_multipliers, is_inequality
    )

    residual_vector = problem.compute_residual_vector(initial_vals)
    initial_cost = jnp.sum(residual_vector**2)

    if config.penalty_initial is not None:
        penalty_params = tuple(
            jnp.full(h.shape[0], config.penalty_initial) for h in h_vals
        )
    else:
        cost_scale = 10.0 * jnp.maximum(1.0, jnp.abs(initial_cost))

        penalty_params_list = []
        for h_group, is_ineq in zip(h_vals, is_inequality):
            if is_ineq:
                max_violation = jnp.max(relu(h_group), axis=1)
            else:
                max_violation = jnp.max(jnp.abs(h_group), axis=1)
            c_squared = 0.5 * max_violation**2
            penalty_group = jnp.clip(
                cost_scale / jnp.maximum(1.0, c_squared),
                config.penalty_min,
                config.penalty_max,
            )
            penalty_params_list.append(penalty_group)
        penalty_params = tuple(penalty_params_list)

    if verbose:
        constraint_dim = sum(h.size for h in h_vals)
        max_penalty = jnp.max(jnp.array([jnp.max(p) for p in penalty_params]))
        jax_log(
            "Augmented Lagrangian: initial snorm={snorm:.4e}, csupn={csupn:.4e}, max_rho={max_rho:.4e}, constraint_dim={dim}",
            snorm=initial_snorm,
            csupn=initial_csupn,
            max_rho=max_penalty,
            dim=constraint_dim,
        )

    return AugmentedLagrangianState(
        lagrange_multipliers=lagrange_multipliers,
        penalty_params=penalty_params,
        constraint_values_prev=h_vals,
        is_inequality=is_inequality,
        snorm=initial_snorm,
        constraint_violation=initial_csupn,
        snorm_initial=initial_snorm,
    )


def update_al_state(
    problem: Any,
    vals: Any,
    al_state: Any,
    config: Any,
    verbose: Any = False,
) -> Any:
    h_vals = problem._compute_constraint_values(vals)

    lagrange_multipliers_updated = []
    penalty_params_updated = []
    for lambda_group, penalty_group, h_group, h_prev, is_ineq in zip(
        al_state.lagrange_multipliers,
        al_state.penalty_params,
        h_vals,
        al_state.constraint_values_prev,
        al_state.is_inequality,
    ):
        lambda_new = lambda_group + penalty_group[:, None] * h_group
        if is_ineq:
            lambda_new = relu(lambda_new)
        lambda_new = jnp.clip(lambda_new, config.lambda_min, config.lambda_max)
        lagrange_multipliers_updated.append(lambda_new)

        if is_ineq:
            violation = jnp.max(relu(h_group), axis=1)
            violation_prev = jnp.max(relu(h_prev), axis=1)
        else:
            violation = jnp.max(jnp.abs(h_group), axis=1)
            violation_prev = jnp.max(jnp.abs(h_prev), axis=1)

        insufficient_progress = violation > (
            config.violation_reduction_threshold * violation_prev
        )
        penalty_new = jnp.where(
            insufficient_progress,
            jnp.minimum(penalty_group * config.penalty_factor, config.penalty_max),
            penalty_group,
        )
        penalty_params_updated.append(penalty_new)

    lagrange_multipliers_updated = tuple(lagrange_multipliers_updated)
    penalty_params_updated = tuple(penalty_params_updated)

    snorm_new, csupn_new = _compute_snorm_csupn(
        h_vals, lagrange_multipliers_updated, al_state.is_inequality
    )

    if verbose:
        max_penalty = jnp.max(jnp.array([jnp.max(p) for p in penalty_params_updated]))
        jax_log(
            " AL update: snorm={snorm:.4e}, csupn={csupn:.4e}, max_rho={max_rho:.4e}",
            snorm=snorm_new,
            csupn=csupn_new,
            max_rho=max_penalty,
            ordered=True,
        )

    return AugmentedLagrangianState(
        lagrange_multipliers=lagrange_multipliers_updated,
        penalty_params=penalty_params_updated,
        constraint_values_prev=h_vals,
        is_inequality=al_state.is_inequality,
        snorm=snorm_new,
        constraint_violation=csupn_new,
        snorm_initial=al_state.snorm_initial,
    )


def update_problem_al_params(
    problem: Any,
    al_state: Any,
) -> Any:
    from ._core import AugmentedLagrangianParams

    updated_costs = []
    constraint_group_idx = 0

    for cost in problem._stacked_costs:
        if cost.kind == "l2_squared":
            updated_costs.append(cost)
        else:
            assert len(cost.args) == 1
            current_al_params: Any = cost.args[0]

            with jdc.copy_and_mutate(cost) as cost_copy:
                cost_copy.args = (
                    AugmentedLagrangianParams(
                        lagrange_multipliers=al_state.lagrange_multipliers[
                            constraint_group_idx
                        ],
                        penalty_params=al_state.penalty_params[constraint_group_idx],
                        original_args=current_al_params.original_args,
                        constraint_index=current_al_params.constraint_index,
                    ),
                )
            updated_costs.append(cost_copy)
            constraint_group_idx += 1

    with jdc.copy_and_mutate(problem) as updated_problem:
        updated_problem._stacked_costs = tuple(updated_costs)

    return updated_problem


def check_al_convergence(
    al_state: Any,
    config: Any,
) -> Any:
    converged_absolute = (
        jnp.maximum(al_state.snorm, al_state.constraint_violation)
        < config.tolerance_absolute
    )

    converged_relative = (
        al_state.snorm / jnp.maximum(al_state.snorm_initial, 1.0)
        < config.tolerance_relative
    )
    return converged_absolute, converged_relative
