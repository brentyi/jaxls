from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from jax.nn import relu

from ._variables import VarValues
from .utils import jax_log

if TYPE_CHECKING:
    from ._core import AnalyzedLeastSquaresProblem, AugmentedLagrangianParams


@jdc.pytree_dataclass
class AugmentedLagrangianConfig:
    """Configuration for Augmented Lagrangian solver (ALGENCAN-style)."""

    penalty_factor: float | jax.Array = 4.0
    """Penalty multiplier when constraint progress stagnates."""

    penalty_max: float | jax.Array = 1e7
    """Maximum penalty parameter."""

    penalty_min: float | jax.Array = 1e-6
    """Minimum penalty parameter."""

    penalty_initial: float | jax.Array | None = None
    """Initial penalty parameter. If None, uses ALGENCAN-style heuristic:
    rho = 10 * max(1, |f|) / max(1, 0.5 * c²). Set to a fixed value (e.g., 1.0)
    to override the automatic initialization."""

    tolerance_absolute: float | jax.Array = 1e-5
    """Absolute convergence tolerance: max(snorm, csupn) < tol."""

    tolerance_relative: float | jax.Array = 1e-4
    """Relative convergence tolerance: snorm / snorm_initial < tol."""

    violation_reduction_threshold: float | jax.Array = 0.9
    """Increase penalty if violation > threshold * previous_violation.
    E.g., 0.9 requires ~10% reduction per update to avoid penalty growth.
    Use higher values (e.g., 0.99) for more lenient penalty updates."""

    penalty_decrease_threshold: float | jax.Array = 0.1
    """Decrease penalty if violation < threshold * previous_violation.
    E.g., 0.1 means 90%+ reduction allows penalty decrease.
    Set to 0.0 to disable penalty decrease."""

    penalty_decrease_factor: float | jax.Array = 0.25
    """Factor to multiply penalty by when constraints are well-satisfied.
    E.g., 0.25 means penalty decreases by 4x when violation drops significantly."""

    lambda_min: float | jax.Array = -1e7
    """Minimum Lagrange multiplier (safeguard)."""

    lambda_max: float | jax.Array = 1e7
    """Maximum Lagrange multiplier (safeguard)."""

    inner_solve_tolerance: float | jax.Array = 1e-2
    """Only update AL parameters when inner problem has converged.
    Update when cost_reldelta < this tolerance, meaning the LM solver
    has approximately solved the current augmented subproblem."""


@jdc.pytree_dataclass
class AugmentedLagrangianState:
    """State for Augmented Lagrangian method, stored in solver state when constraints present."""

    lagrange_multipliers: tuple[jax.Array, ...]
    """Lagrange multipliers, one array per constraint group.
    Each has shape (constraint_count, constraint_flat_dim)."""

    penalty_params: tuple[jax.Array, ...]
    """Per-instance penalty parameters, one array per constraint group.
    Each has shape (constraint_count,) - one scalar penalty per constraint instance."""

    constraint_values_prev: tuple[jax.Array, ...]
    """Previous constraint values for per-constraint progress, one array per constraint group.
    Each has shape (constraint_count, constraint_flat_dim)."""

    is_inequality: jdc.Static[tuple[bool, ...]]
    """Whether each constraint group is an inequality (True) or equality (False)."""

    snorm: jax.Array
    """Complementarity measure (infinity-norm). Equality: |h(x)|, inequality: |min(-g(x), lam)|."""

    constraint_violation: jax.Array
    """Constraint violation (infinity-norm). Equality: max|h(x)|, inequality: max(0, g(x))."""

    snorm_initial: jax.Array
    """Initial snorm for relative tolerance check."""

    cost_at_last_update: jax.Array
    """Cost when AL parameters were last updated, for inner solve detection."""


def _compute_snorm_csupn(
    h_vals: tuple[jax.Array, ...],
    lagrange_multipliers: tuple[jax.Array, ...],
    is_inequality: tuple[bool, ...],
) -> tuple[jax.Array, jax.Array]:
    """Compute complementarity (snorm) and constraint violation (csupn), both infinity-norm."""
    snorm_parts = []
    csupn_parts = []

    for h_group, lambda_group, is_ineq in zip(
        h_vals, lagrange_multipliers, is_inequality
    ):
        if is_ineq:
            # Inequality: snorm = |min(-g, lambda)|, csupn = max(0, g).
            comp = jnp.minimum(-h_group, lambda_group)
            snorm_parts.append(jnp.max(jnp.abs(comp)))
            csupn_parts.append(jnp.max(relu(h_group)))
        else:
            # Equality: snorm = csupn = |h|.
            max_abs_h = jnp.max(jnp.abs(h_group))
            snorm_parts.append(max_abs_h)
            csupn_parts.append(max_abs_h)

    if len(snorm_parts) == 0:
        return jnp.array(0.0), jnp.array(0.0)

    snorm = jnp.max(jnp.array(snorm_parts))
    csupn = jnp.max(jnp.array(csupn_parts))
    return snorm, csupn


def initialize_al_state(
    problem: AnalyzedLeastSquaresProblem,
    initial_vals: VarValues,
    config: AugmentedLagrangianConfig,
    verbose: bool = False,
) -> AugmentedLagrangianState:
    """Initialize Augmented Lagrangian state.

    Args:
        problem: The analyzed problem with constraints.
        initial_vals: Initial variable values.
        config: AL configuration.
        verbose: Whether to log initialization info.

    Returns:
        Initialized AL state.
    """
    # Determine which constraint groups are inequalities.
    is_inequality = tuple(
        cost.kind in ("constraint_leq_zero", "constraint_geq_zero")
        for cost in problem._stacked_costs
        if cost.kind != "l2_squared"
    )
    assert len(is_inequality) > 0, (
        "initialize_al_state requires constraints (kind != 'l2_squared')."
    )

    # Compute initial constraint values (tuple of arrays, one per group).
    h_vals = problem._compute_constraint_values(initial_vals)
    lagrange_multipliers = tuple(jnp.zeros_like(h) for h in h_vals)
    initial_snorm, initial_csupn = _compute_snorm_csupn(
        h_vals, lagrange_multipliers, is_inequality
    )

    # Compute initial cost for cost_at_last_update tracking.
    residual_vector = problem.compute_residual_vector(initial_vals)
    initial_cost = jnp.sum(residual_vector**2)

    # Initialize penalty parameters (per-instance, not per-element).
    # h_vals has shape (constraint_count, constraint_flat_dim) per group.
    # penalty_params has shape (constraint_count,) per group - one scalar per instance.
    if config.penalty_initial is not None:
        # User-specified fixed initial penalty.
        penalty_params = tuple(
            jnp.full(h.shape[0], config.penalty_initial) for h in h_vals
        )
    else:
        # ALGENCAN-style per-instance formula:
        # rho_i = 10 * max(1, |f|) / max(1, 0.5 * max_c_i^2)
        # where max_c_i is the max violation across elements of constraint i.
        cost_scale = 10.0 * jnp.maximum(1.0, jnp.abs(initial_cost))

        penalty_params_list = []
        for h_group, is_ineq in zip(h_vals, is_inequality):
            # Aggregate violation across elements (axis=1) using max.
            if is_ineq:
                # For inequality: max of positive part across elements.
                max_violation = jnp.max(relu(h_group), axis=1)
            else:
                # For equality: max of absolute value across elements.
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
        cost_at_last_update=initial_cost,
    )


def update_al_state(
    problem: AnalyzedLeastSquaresProblem,
    vals: VarValues,
    al_state: AugmentedLagrangianState,
    config: AugmentedLagrangianConfig,
    current_cost: jax.Array,
    step_accepted: jax.Array | bool = True,
    verbose: bool = False,
) -> AugmentedLagrangianState:
    """Update AL state after an LM step.

    Only updates Lagrange multipliers and penalties when the inner problem
    has approximately converged (cost_reldelta < inner_solve_tolerance).
    This prevents the cost surface from changing too frequently.

    Args:
        problem: The analyzed problem with constraints.
        vals: Current variable values.
        al_state: Current AL state.
        config: AL configuration.
        current_cost: Current total cost (for inner convergence check).
        step_accepted: Whether the LM step was accepted.
        verbose: Whether to log update info.

    Returns:
        Updated AL state.
    """
    # Evaluate constraints at current solution (tuple of arrays).
    h_vals = problem._compute_constraint_values(vals)

    # Check if we should update AL parameters: cost hasn't changed much since last update.
    # This indicates the inner LM problem has approximately converged for the current AL landscape.
    cost_reldelta = jnp.abs(current_cost - al_state.cost_at_last_update) / jnp.maximum(
        jnp.abs(al_state.cost_at_last_update), 1e-8
    )
    al_should_update = cost_reldelta < config.inner_solve_tolerance

    # Update Lagrange multipliers and penalties in a single pass.
    # Only actually update when inner problem has converged.
    # Note: penalty_group has shape (constraint_count,) - per-instance.
    # lambda_group and h_group have shape (constraint_count, constraint_flat_dim).
    lagrange_multipliers_updated = []
    penalty_params_updated = []
    for lambda_group, penalty_group, h_group, h_prev, is_ineq in zip(
        al_state.lagrange_multipliers,
        al_state.penalty_params,
        h_vals,
        al_state.constraint_values_prev,
        al_state.is_inequality,
    ):
        # Compute new lambda: lambda_new = lambda_old + rho * h(x).
        # Broadcast penalty_group from (constraint_count,) to (constraint_count, 1).
        lambda_candidate = lambda_group + penalty_group[:, None] * h_group
        if is_ineq:
            lambda_candidate = relu(lambda_candidate)
        lambda_candidate = jnp.clip(lambda_candidate, config.lambda_min, config.lambda_max)
        # Update lambda when inner problem has converged (regardless of step acceptance).
        # This allows AL parameters to update even when LM rejects the step.
        lambda_new = jnp.where(al_should_update, lambda_candidate, lambda_group)
        lagrange_multipliers_updated.append(lambda_new)

        # Update penalty based on constraint progress (per-instance).
        # Aggregate violations across elements using max for per-instance decision.
        # - Increase if insufficient progress (max_violation > threshold * max_prev)
        # - Decrease if excellent progress (max_violation < decrease_threshold * max_prev)
        # - Otherwise keep the same
        if is_ineq:
            max_violation = jnp.max(relu(h_group), axis=1)
            max_violation_prev = jnp.max(relu(h_prev), axis=1)
        else:
            max_violation = jnp.max(jnp.abs(h_group), axis=1)
            max_violation_prev = jnp.max(jnp.abs(h_prev), axis=1)

        insufficient_progress = max_violation > (
            config.violation_reduction_threshold * max_violation_prev
        )
        excellent_progress = max_violation < (
            config.penalty_decrease_threshold * max_violation_prev
        )

        # Increase penalty for insufficient progress.
        penalty_increased = jnp.minimum(
            penalty_group * config.penalty_factor, config.penalty_max
        )
        # Decrease penalty for excellent progress.
        penalty_decreased = jnp.maximum(
            penalty_group * config.penalty_decrease_factor, config.penalty_min
        )

        penalty_candidate = jnp.where(
            insufficient_progress,
            penalty_increased,
            jnp.where(excellent_progress, penalty_decreased, penalty_group),
        )
        # Only update penalty when AL should update.
        penalty_new = jnp.where(al_should_update, penalty_candidate, penalty_group)
        penalty_params_updated.append(penalty_new)

    lagrange_multipliers_updated = tuple(lagrange_multipliers_updated)
    penalty_params_updated = tuple(penalty_params_updated)

    # Compute snorm (complementarity) and csupn (constraint violation).
    # Always use current multipliers for measurement.
    snorm_new, csupn_new = _compute_snorm_csupn(
        h_vals, lagrange_multipliers_updated, al_state.is_inequality
    )

    if verbose:
        max_penalty = jnp.max(jnp.array([jnp.max(p) for p in penalty_params_updated]))
        jax_log(
            " AL update: snorm={snorm:.4e}, csupn={csupn:.4e}, max_rho={max_rho:.4e}, al_update={al_upd}",
            snorm=snorm_new,
            csupn=csupn_new,
            max_rho=max_penalty,
            al_upd=al_should_update,
            ordered=True,
        )

    # Update constraint_values_prev when step was accepted AND AL should update.
    constraint_values_prev_new = tuple(
        jnp.where(step_accepted & al_should_update, h_new, h_prev)
        for h_new, h_prev in zip(h_vals, al_state.constraint_values_prev)
    )

    # Always update cost_at_last_update to track step-to-step changes.
    cost_at_last_update_new = current_cost

    return AugmentedLagrangianState(
        lagrange_multipliers=lagrange_multipliers_updated,
        penalty_params=penalty_params_updated,
        constraint_values_prev=constraint_values_prev_new,
        is_inequality=al_state.is_inequality,
        snorm=snorm_new,
        constraint_violation=csupn_new,
        snorm_initial=al_state.snorm_initial,
        cost_at_last_update=cost_at_last_update_new,
    )


def update_problem_al_params(
    problem: AnalyzedLeastSquaresProblem,
    al_state: AugmentedLagrangianState,
) -> AnalyzedLeastSquaresProblem:
    """Update AL params on constraint costs in the problem.

    Args:
        problem: The analyzed problem with constraints.
        al_state: Current AL state with updated multipliers and penalties.

    Returns:
        Problem with updated AL params on constraint costs.
    """
    # Import here to avoid circular import at module load time.
    from ._core import AugmentedLagrangianParams

    updated_costs = []
    constraint_group_idx = 0

    for cost in problem._stacked_costs:
        if cost.kind == "l2_squared":
            # Regular cost, no update needed.
            updated_costs.append(cost)
        else:
            # Constraint-derived cost, update AL params.
            assert len(cost.args) == 1
            current_al_params: AugmentedLagrangianParams = cost.args[0]

            # Create updated AL params with new multipliers/penalties.
            # AL state already stores arrays with shape (constraint_count, constraint_flat_dim).
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
    al_state: AugmentedLagrangianState,
    config: AugmentedLagrangianConfig,
) -> tuple[jax.Array, jax.Array]:
    """Check AL convergence criteria.

    Args:
        al_state: Current AL state.
        config: AL configuration.

    Returns:
        Tuple of (converged_absolute, converged_relative) boolean arrays.
    """
    # Absolute convergence: snorm and constraint violation below threshold.
    converged_absolute = (
        jnp.maximum(al_state.snorm, al_state.constraint_violation)
        < config.tolerance_absolute
    )
    # Relative convergence: snorm reduced by tolerance_relative factor.
    # Use floor of 1.0 to handle initial_snorm ≈ 0 (already satisfied).
    converged_relative = (
        al_state.snorm / jnp.maximum(al_state.snorm_initial, 1.0)
        < config.tolerance_relative
    )
    return converged_absolute, converged_relative
