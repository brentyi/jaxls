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
    ``rho = 10 * max(1, |f|) / max(1, 0.5 * c^2)``. Set to a fixed value (e.g., 1.0) to
    override the automatic initialization."""

    tolerance_absolute: float | jax.Array = 1e-5
    """Absolute convergence tolerance: ``max(snorm, csupn) < tol``."""

    tolerance_relative: float | jax.Array = 1e-4
    """Relative convergence tolerance: ``snorm / snorm_initial < tol``."""

    violation_reduction_threshold: float | jax.Array = 0.5
    """Increase penalty if violation > threshold * previous_violation.
    E.g., 0.9 requires ~10% reduction per update to avoid penalty growth.
    Use higher values (e.g., 0.99) for more lenient penalty updates."""

    lambda_min: float | jax.Array = -1e7
    """Minimum Lagrange multiplier (safeguard)."""

    lambda_max: float | jax.Array = 1e7
    """Maximum Lagrange multiplier (safeguard)."""

    inner_solve_tolerance: float | jax.Array = 1e-2
    """Only update AL parameters when inner problem has converged.
    Update when ``||gradient|| < tolerance``, meaning the LM solver
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

    # Compute initial cost for penalty initialization heuristic.
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
    )


def update_al_state(
    problem: AnalyzedLeastSquaresProblem,
    vals: VarValues,
    al_state: AugmentedLagrangianState,
    config: AugmentedLagrangianConfig,
    verbose: bool = False,
) -> AugmentedLagrangianState:
    """Update AL state: Lagrange multipliers, penalties, and convergence metrics.

    Args:
        problem: The analyzed problem with constraints.
        vals: Current variable values.
        al_state: Current AL state.
        config: AL configuration.
        verbose: Whether to log update info.

    Returns:
        Updated AL state.
    """
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
        # Update lambda: lambda = lambda + rho * h(x).
        lambda_new = lambda_group + penalty_group[:, None] * h_group
        if is_ineq:
            lambda_new = relu(lambda_new)
        lambda_new = jnp.clip(lambda_new, config.lambda_min, config.lambda_max)
        lagrange_multipliers_updated.append(lambda_new)

        # Update penalty based on constraint progress.
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
