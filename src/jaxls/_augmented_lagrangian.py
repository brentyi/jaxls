from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, assert_never, overload

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp

from ._solvers import NonlinearSolver, SolveSummary
from ._variables import VarValues
from .utils import jax_log

if TYPE_CHECKING:
    from ._core import AnalyzedLeastSquaresProblem


@jdc.pytree_dataclass
class AugmentedLagrangianParams:
    """Parameters for a single augmented constraint cost.

    Each augmented cost gets its own params with arrays for just that constraint.
    The original constraint args are bundled here for type-safe access.
    """

    lagrange_multipliers: jax.Array
    """Lagrange multipliers for this constraint. Shape: (constraint_flat_dim,)."""

    penalty_params: jax.Array
    """Penalty parameters for this constraint. Shape: (constraint_flat_dim,)."""

    original_args: tuple[Any, ...]
    """The original constraint args to pass to compute_residual."""

    constraint_index: jdc.Static[int]
    """Index of this constraint in the original problem's stacked_constraints.

    This is needed because augmented costs may be reordered during re-analysis.
    """


@jdc.pytree_dataclass
class AugmentedLagrangianConfig:
    """Configuration for Augmented Lagrangian solver (ALGENCAN-style)."""

    penalty_initial: float | jax.Array | jdc.Static[Literal["auto"]] = "auto"
    """Initial penalty. If 'auto', computed from initial cost and constraint violation."""

    penalty_factor: float | jax.Array = 10.0
    """Penalty multiplier when constraint progress stagnates."""

    penalty_max: float | jax.Array = 1e8
    """Maximum penalty parameter."""

    penalty_min: float | jax.Array = 1e-6
    """Minimum penalty parameter."""

    tolerance_absolute: float | jax.Array = 1e-6
    """Absolute convergence tolerance: max(snorm, csupn) < tol."""

    tolerance_relative: float | jax.Array = 1e-4
    """Relative convergence tolerance: snorm / snorm_initial < tol."""

    max_iterations: jdc.Static[int] = 50
    """Maximum outer loop iterations."""

    violation_reduction_threshold: float | jax.Array = 0.5
    """Increase penalty if violation doesn't reduce by this fraction."""

    lambda_min: float | jax.Array = -1e8
    """Minimum Lagrange multiplier (safeguard)."""

    lambda_max: float | jax.Array = 1e8
    """Maximum Lagrange multiplier (safeguard)."""


@jdc.pytree_dataclass
class _AugmentedLagrangianState:
    """State for outer Augmented Lagrangian loop."""

    vals: VarValues
    """Current variable values."""

    lagrange_multipliers: tuple[jax.Array, ...]
    """Lagrange multipliers, one array per constraint group. Each has shape (count * flat_dim,)."""

    penalty_params: tuple[jax.Array, ...]
    """Per-constraint penalty parameters, one array per constraint group."""

    constraint_values_prev: tuple[jax.Array, ...]
    """Previous constraint values for per-constraint progress, one array per constraint group."""

    snorm: jax.Array
    """Complementarity measure (infinity-norm). Equality: |h(x)|, inequality: |min(-g(x), λ)|."""

    snorm_prev: jax.Array
    """Previous snorm for progress checking."""

    constraint_violation: jax.Array
    """Constraint violation (infinity-norm). Equality: max|h(x)|, inequality: max(0, g(x))."""

    initial_snorm: jax.Array
    """Initial snorm for relative tolerance check."""

    outer_iteration: int
    """Current outer iteration number."""

    inner_summary: SolveSummary
    """Summary from most recent inner solve."""

    # Pre-allocated history arrays for JIT compatibility.
    constraint_violation_history: jax.Array
    """Constraint violation history, shape (max_outer_iterations,)."""

    penalty_history: jax.Array
    """Max penalty history, shape (max_outer_iterations,)."""

    inner_iterations_count: jax.Array
    """Inner iteration counts, shape (max_outer_iterations,)."""

    epsopk: jax.Array
    """Current inner solver tolerance (ALGENCAN-style adaptive)."""


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

    def _extract_original_costs(self, problem: AnalyzedLeastSquaresProblem) -> list:
        """Extract original Cost objects from analyzed problem.

        We need to recreate Cost objects from the analyzed problem to combine
        with augmented constraint costs and re-analyze.
        """
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

    def _extract_variables(self, problem: AnalyzedLeastSquaresProblem) -> list:
        """Extract variable objects from analyzed problem."""
        variables = []
        for var_type, ids in problem.sorted_ids_from_var_type.items():
            for var_id in ids:
                variables.append(var_type(var_id))
        return variables

    def _analyze_augmented_problem(
        self, problem: AnalyzedLeastSquaresProblem
    ) -> AnalyzedLeastSquaresProblem:
        """Pre-analyze augmented problem structure once.

        Creates augmented costs from constraints with AL params baked in, then
        re-analyzes the combined problem. The structure is fixed; only the
        args (containing AugmentedLagrangianParams) will vary during optimization.

        Args:
            problem: Original analyzed problem with constraints.

        Returns:
            Analyzed augmented problem with AL params initialized.
        """
        from ._core import LeastSquaresProblem

        constraint_costs = []

        # Create augmented costs from constraints.
        for constraint_idx, (stacked_constraint, constraint_count) in enumerate(
            zip(problem.stacked_constraints, problem.constraint_counts)
        ):
            constraint_flat_dim = stacked_constraint.residual_flat_dim

            # Create per-constraint AL params with original args bundled.
            # Shape: (constraint_flat_dim,) for each array.
            al_params = AugmentedLagrangianParams(
                lagrange_multipliers=jnp.zeros(constraint_flat_dim),
                penalty_params=jnp.ones(constraint_flat_dim),
                original_args=stacked_constraint.args,
                constraint_index=constraint_idx,
            )

            # Broadcast to batch dimension (constraint_count,).
            # Note: original_args already have batch dimension from stacking.
            al_params_batched = AugmentedLagrangianParams(
                lagrange_multipliers=jnp.broadcast_to(
                    al_params.lagrange_multipliers,
                    (constraint_count, constraint_flat_dim),
                ),
                penalty_params=jnp.broadcast_to(
                    al_params.penalty_params, (constraint_count, constraint_flat_dim)
                ),
                original_args=al_params.original_args,  # Already batched
                constraint_index=constraint_idx,
            )

            augmented_cost = create_augmented_constraint_cost(
                stacked_constraint, al_params_batched
            )
            constraint_costs.append(augmented_cost)

        # Extract original costs and variables for re-analysis.
        original_costs = self._extract_original_costs(problem)
        variables = self._extract_variables(problem)

        if self.verbose:
            jax_log("Pre-analyzing augmented problem structure (one-time cost)...")

        # Re-analyze the combined problem to get correct Jacobian structure.
        augmented = LeastSquaresProblem(
            costs=original_costs + constraint_costs,
            variables=variables,
        ).analyze()

        return augmented

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
        """Solve constrained optimization problem using Augmented Lagrangian (ALGENCAN-style).

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

        # Compute initial constraint values (tuple of arrays, one per group).
        h_vals = problem._compute_constraint_values(initial_vals)

        # Initialize Lagrange multipliers (zeros, as recommended by literature).
        lagrange_multipliers = tuple(jnp.zeros_like(h) for h in h_vals)

        # Compute initial snorm and csupn. With λ=0, snorm = csupn for equalities.
        initial_snorm, initial_csupn = self._compute_snorm_csupn(
            problem, h_vals, lagrange_multipliers
        )

        # Initialize penalty parameter using ALGENCAN formula.
        if self.config.penalty_initial == "auto":
            # ALGENCAN formula: rho = 10 * max(1, |f|) / max(1, 0.5 * sum(c^2))
            residual_vector_result = problem.compute_residual_vector(initial_vals)
            if isinstance(residual_vector_result, tuple):
                residual_vector = residual_vector_result[0]
            else:
                residual_vector = residual_vector_result
            initial_cost = jnp.sum(residual_vector**2)

            # Sum of squared violations. For inequalities, only count g > 0.
            sum_c_squared = jnp.array(0.0)
            for stacked_constraint, h_group in zip(problem.stacked_constraints, h_vals):
                if stacked_constraint.constraint_type == "leq_zero":
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

        # Initialize per-constraint penalties (all same initially).
        penalty_params = tuple(jnp.full_like(h, penalty_initial) for h in h_vals)

        # Pre-analyze augmented problem structure once.
        augmented_structure = self._analyze_augmented_problem(problem)

        constraint_dim = sum(h.size for h in h_vals)
        if self.verbose:
            jax_log(
                "Augmented Lagrangian: initial snorm={snorm:.4e}, csupn={csupn:.4e}, penalty={penalty:.4e}, constraint_dim={dim}",
                snorm=initial_snorm,
                csupn=initial_csupn,
                penalty=penalty_initial,
                dim=constraint_dim,
            )

        # Pre-allocate history arrays.
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

        # Initialize inner tolerance (ALGENCAN-style: start with sqrt of target).
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

        def cond_fn(state: _AugmentedLagrangianState) -> jax.Array:
            """Check convergence. Always run at least one iteration."""
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

        def body_fn(state: _AugmentedLagrangianState) -> _AugmentedLagrangianState:
            """Perform one outer iteration."""
            return self._step(problem, augmented_structure, state)

        state = jax.lax.while_loop(cond_fn, body_fn, state)

        # Check final convergence for logging.
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
        problem: AnalyzedLeastSquaresProblem,
        h_vals: tuple[jax.Array, ...],
        lagrange_multipliers: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, jax.Array]:
        """Compute complementarity (snorm) and constraint violation (csupn), both infinity-norm."""
        snorm_parts = []
        csupn_parts = []

        for stacked_constraint, h_group, lambda_group in zip(
            problem.stacked_constraints, h_vals, lagrange_multipliers
        ):
            if stacked_constraint.constraint_type == "leq_zero":
                # Inequality: snorm = |min(-g, λ)|, csupn = max(0, g).
                comp = jnp.minimum(-h_group, lambda_group)
                snorm_parts.append(jnp.max(jnp.abs(comp)))
                csupn_parts.append(jnp.max(jnp.maximum(0.0, h_group)))
            else:
                # Equality: snorm = csupn = |h|.
                snorm_parts.append(jnp.max(jnp.abs(h_group)))
                csupn_parts.append(jnp.max(jnp.abs(h_group)))

        if len(snorm_parts) == 0:
            return jnp.array(0.0), jnp.array(0.0)

        snorm = jnp.max(jnp.array(snorm_parts))
        csupn = jnp.max(jnp.array(csupn_parts))
        return snorm, csupn

    def _compute_per_constraint_violation(
        self,
        problem: AnalyzedLeastSquaresProblem,
        h_vals: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ...]:
        """Per-element violation: |h(x)| for equality, max(0, g(x)) for inequality."""
        violation_parts = []

        for stacked_constraint, h_group in zip(problem.stacked_constraints, h_vals):
            if stacked_constraint.constraint_type == "leq_zero":
                violation_parts.append(jnp.maximum(0.0, h_group))
            else:
                violation_parts.append(jnp.abs(h_group))

        return tuple(violation_parts)

    def _step(
        self,
        problem: AnalyzedLeastSquaresProblem,
        augmented_structure: AnalyzedLeastSquaresProblem,
        state: _AugmentedLagrangianState,
    ) -> _AugmentedLagrangianState:
        """Perform one outer iteration of the Augmented Lagrangian method (ALGENCAN-style).

        Args:
            problem: Original problem with constraints (for constraint evaluation).
            augmented_structure: Pre-analyzed augmented problem with parameterized costs.
            state: Current state of the Augmented Lagrangian solver.

        Returns:
            Updated state after one outer iteration.
        """
        # Update AL params on each augmented cost.
        # Original costs are first, augmented costs are appended after.
        num_original_costs = len(problem.stacked_costs)
        updated_costs = list(augmented_structure.stacked_costs[:num_original_costs])

        # Augmented costs may be reordered; use constraint_index to get correct params.
        for cost in augmented_structure.stacked_costs[num_original_costs:]:
            current_al_params: AugmentedLagrangianParams = cost.args[0]
            constraint_idx = current_al_params.constraint_index

            (constraint_count,) = cost._get_batch_axes()
            constraint_flat_dim = problem.stacked_constraints[
                constraint_idx
            ].residual_flat_dim

            # Get this constraint group's lambda/penalty from state tuples.
            lambda_flat = state.lagrange_multipliers[constraint_idx]
            penalty_flat = state.penalty_params[constraint_idx]

            # Reshape to (constraint_count, constraint_flat_dim).
            lambda_reshaped = lambda_flat.reshape(constraint_count, constraint_flat_dim)
            penalty_reshaped = penalty_flat.reshape(
                constraint_count, constraint_flat_dim
            )

            # Create updated AL params with new multipliers/penalties.
            al_params_updated = AugmentedLagrangianParams(
                lagrange_multipliers=lambda_reshaped,
                penalty_params=penalty_reshaped,
                original_args=current_al_params.original_args,
                constraint_index=constraint_idx,
            )

            with jdc.copy_and_mutate(cost) as cost_copy:
                cost_copy.args = (al_params_updated,)
            updated_costs.append(cost_copy)

        with jdc.copy_and_mutate(augmented_structure) as augmented_problem:
            augmented_problem.stacked_costs = tuple(updated_costs)

        # Solve inner unconstrained problem using ALGENCAN-style adaptive tolerance.
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

        # Evaluate constraints at new solution (tuple of arrays).
        h_vals = problem._compute_constraint_values(vals_updated)

        # Update Lagrange multipliers: λ_new = λ_old + ρ * h(x).
        # Project multipliers: non-negative for inequalities, then apply safeguard bounds.
        lagrange_multipliers_updated = []
        for stacked_constraint, lambda_group, penalty_group, h_group in zip(
            problem.stacked_constraints,
            state.lagrange_multipliers,
            state.penalty_params,
            h_vals,
        ):
            lambda_new = lambda_group + penalty_group * h_group
            if stacked_constraint.constraint_type == "leq_zero":
                lambda_new = jnp.maximum(0.0, lambda_new)
            lambda_new = jnp.clip(
                lambda_new, self.config.lambda_min, self.config.lambda_max
            )
            lagrange_multipliers_updated.append(lambda_new)
        lagrange_multipliers_updated = tuple(lagrange_multipliers_updated)

        # Compute snorm (complementarity) and csupn (constraint violation).
        snorm_new, csupn_new = self._compute_snorm_csupn(
            problem, h_vals, lagrange_multipliers_updated
        )

        # Per-constraint violation for penalty updates.
        constraint_violation_per = self._compute_per_constraint_violation(
            problem, h_vals
        )
        constraint_violation_prev_per = self._compute_per_constraint_violation(
            problem, state.constraint_values_prev
        )

        # If inner solver failed, increase all penalties.
        inner_hit_max_iters = inner_summary.termination_criteria[3]
        inner_made_progress = inner_summary.termination_criteria[0]
        rhorestart_needed = inner_hit_max_iters & ~inner_made_progress

        # Update penalties for constraints with insufficient progress.
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

        # Update inner tolerance for next iteration (ALGENCAN-style).
        nlpsupn = inner_summary.termination_deltas[1]  # gradient magnitude
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

        # Compute max penalty using JAX operations (not Python max which doesn't work with tracers).
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


def create_augmented_constraint_cost(
    analyzed_constraint: Any,  # _AnalyzedCost, but avoid circular import
    al_params: AugmentedLagrangianParams,
) -> Any:  # Returns _AnalyzedCost
    """Create an augmented cost from a constraint for Augmented Lagrangian method.

    This creates an _AnalyzedCost object that converts a constraint into an augmented
    Lagrangian residual with per-constraint penalty parameters.

    For equality constraints h(x) = 0:
        r = sqrt(rho) * (h(x) + lambda/rho)

    For inequality constraints g(x) <= 0:
        r = sqrt(rho) * max(0, g(x) + lambda/rho)

    Args:
        analyzed_constraint: The analyzed constraint (as _AnalyzedCost) to convert.
        al_params: Augmented Lagrangian parameters bundled with original args.

    Returns:
        An _AnalyzedCost with args=(al_params,).
    """
    from ._core import _AnalyzedCost

    constraint_flat_dim = analyzed_constraint.residual_flat_dim

    def augmented_residual_fn(
        vals: VarValues,
        al_params_inner: AugmentedLagrangianParams,
    ) -> jax.Array:
        """Compute augmented constraint residual with per-constraint penalty."""
        constraint_val = analyzed_constraint.compute_residual(
            vals, *al_params_inner.original_args
        ).flatten()

        # For inequality constraints: only penalize when violated (max formulation)
        # For equality constraints: always penalize deviation
        lambdas = al_params_inner.lagrange_multipliers
        rho = al_params_inner.penalty_params
        if analyzed_constraint.constraint_type == "leq_zero":
            # g(x) <= 0: penalize only when g(x) + lambda/rho > 0
            return jnp.sqrt(rho) * jnp.maximum(0.0, constraint_val + lambdas / rho)
        elif analyzed_constraint.constraint_type == "eq_zero":
            # h(x) = 0: always penalize
            return jnp.sqrt(rho) * (constraint_val + lambdas / rho)
        else:
            assert_never(analyzed_constraint.constraint_type)

    return _AnalyzedCost(
        compute_residual=augmented_residual_fn,
        args=(al_params,),
        jac_mode="auto",
        jac_batch_size=None,
        jac_custom_fn=None,
        jac_custom_with_cache_fn=None,
        name=f"augmented_{analyzed_constraint._get_name()}",
        num_variables=analyzed_constraint.num_variables,
        sorted_ids_from_var_type=analyzed_constraint.sorted_ids_from_var_type,
        residual_flat_dim=constraint_flat_dim,
        constraint_type=analyzed_constraint.constraint_type,
    )
