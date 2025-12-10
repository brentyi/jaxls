from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

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
    """Parameters for augmented Lagrangian constraint costs (ALGENCAN-style).

    Each constraint group gets its own lagrange multiplier array and
    per-constraint penalty parameter array.
    """

    lagrange_multipliers: tuple[jax.Array, ...]
    """Lagrange multipliers for each constraint group."""

    penalty_params: tuple[jax.Array, ...]
    """Per-constraint penalty parameters for each constraint group.
    Each array has shape (constraint_count * constraint_flat_dim,)."""


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

    lagrange_multipliers: jax.Array
    """Lagrange multipliers, flat array of shape (constraint_dim,)."""

    penalty_params: jax.Array
    """Per-constraint penalty parameters, shape (constraint_dim,)."""

    snorm: jax.Array
    """Complementarity measure (infinity-norm). Equality: |h(x)|, inequality: |min(-g(x), λ)|."""

    snorm_prev: jax.Array
    """Previous snorm for progress checking."""

    constraint_values_prev: jax.Array
    """Previous constraint values for per-constraint progress, shape (constraint_dim,)."""

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
        self, problem: AnalyzedLeastSquaresProblem, constraint_dim: int
    ) -> AnalyzedLeastSquaresProblem:
        """Pre-analyze augmented problem structure once.

        Creates augmented costs from constraints with AL params baked in, then
        re-analyzes the combined problem. The structure is fixed; only the
        al_params on each augmented cost will vary during optimization.

        Args:
            problem: Original analyzed problem with constraints.
            constraint_dim: Total dimension of all constraints.

        Returns:
            Analyzed augmented problem with AL params initialized.
        """
        from ._core import LeastSquaresProblem

        # Build initial AL params and augmented costs.
        constraint_dims = []
        constraint_costs = []

        for i, (stacked_constraint, constraint_count) in enumerate(
            zip(problem.stacked_constraints, problem.constraint_counts)
        ):
            # stacked_constraint is now _AnalyzedCost
            constraint_flat_dim = stacked_constraint.residual_flat_dim
            total_dim = constraint_count * constraint_flat_dim
            constraint_dims.append(total_dim)

        # Create initial AL params.
        lagrange_mult_arrays = tuple(jnp.zeros(dim) for dim in constraint_dims)
        penalty_param_arrays = tuple(jnp.ones(dim) for dim in constraint_dims)
        al_params = AugmentedLagrangianParams(
            lagrange_multipliers=lagrange_mult_arrays,
            penalty_params=penalty_param_arrays,
        )

        # Create augmented costs from constraints.
        for i, (stacked_constraint, constraint_count) in enumerate(
            zip(problem.stacked_constraints, problem.constraint_counts)
        ):
            constraint_flat_dim = stacked_constraint.residual_flat_dim
            total_dim = constraint_count * constraint_flat_dim

            augmented_cost = create_augmented_constraint_cost(
                stacked_constraint, i, total_dim, al_params
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

        # Compute initial constraint values.
        h_vals = problem.compute_constraint_values(initial_vals)
        constraint_dim = len(h_vals)

        # Initialize Lagrange multipliers (zeros, as recommended by literature).
        lagrange_multipliers = jnp.zeros(constraint_dim)

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

        # Initialize per-constraint penalties (all same initially).
        penalty_params = jnp.full(constraint_dim, penalty_initial)

        # Pre-analyze augmented problem structure once.
        augmented_structure = self._analyze_augmented_problem(problem, constraint_dim)

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
        h_vals: jax.Array,
        lagrange_multipliers: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute complementarity (snorm) and constraint violation (csupn), both infinity-norm."""
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
                # Inequality: snorm = |min(-g, λ)|, csupn = max(0, g).
                comp = jnp.minimum(-h_slice, lambda_slice)
                snorm_parts.append(jnp.max(jnp.abs(comp)))
                csupn_parts.append(jnp.max(jnp.maximum(0.0, h_slice)))
            else:
                # Equality: snorm = csupn = |h|.
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
        problem: AnalyzedLeastSquaresProblem,
        h_vals: jax.Array,
    ) -> jax.Array:
        """Per-element violation: |h(x)| for equality, max(0, g(x)) for inequality."""
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
        # Split flat arrays into per-constraint-group tuples.

        # Compute constraint dims from the original problem.
        constraint_dims = []
        for i, (stacked_constraint, constraint_count) in enumerate(
            zip(problem.stacked_constraints, problem.constraint_counts)
        ):
            constraint_flat_dim = stacked_constraint.residual_flat_dim
            total_dim = constraint_count * constraint_flat_dim
            constraint_dims.append(total_dim)

        # Build lambda/penalty arrays for each constraint group.
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

        # Update al_params on each augmented cost (the costs added from constraints).
        # Original costs are first, augmented costs are appended after.
        # We need to broadcast al_params to match the batch dimension of each cost.
        num_original_costs = len(problem.stacked_costs)
        updated_costs = list(augmented_structure.stacked_costs[:num_original_costs])

        constraint_idx = 0
        for cost in augmented_structure.stacked_costs[num_original_costs:]:
            # Get constraint count from the cost's batch axes
            (constraint_count,) = cost._get_batch_axes()

            # Broadcast al_params to have batch dimension (constraint_count,)
            def broadcast_to_batch(arr: jax.Array) -> jax.Array:
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

        # Solve inner unconstrained problem using ALGENCAN-style adaptive tolerance.
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

        # Evaluate constraints at new solution.
        h_vals = problem.compute_constraint_values(vals_updated)

        # Update Lagrange multipliers: λ_new = λ_old + ρ * h(x).
        lagrange_multipliers_updated = (
            state.lagrange_multipliers + state.penalty_params * h_vals
        )

        # Project multipliers: non-negative for inequalities, then apply safeguard bounds.
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

        # Only increase penalty for constraints that didn't improve sufficiently.
        insufficient_progress_per = constraint_violation_per > (
            self.config.violation_reduction_threshold * constraint_violation_prev_per
        )

        # If inner solver failed, increase all penalties.
        inner_hit_max_iters = inner_summary.termination_criteria[3]
        inner_made_progress = inner_summary.termination_criteria[0]
        rhorestart_needed = inner_hit_max_iters & ~inner_made_progress

        # Update penalties for constraints with insufficient progress.
        penalty_params_updated = jnp.where(
            insufficient_progress_per | rhorestart_needed,
            jnp.minimum(
                state.penalty_params * self.config.penalty_factor,
                self.config.penalty_max,
            ),
            state.penalty_params,
        )

        # Update inner tolerance for next iteration (ALGENCAN-style).
        # Only tighten when making progress on both feasibility AND optimality.
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


def create_augmented_constraint_cost(
    analyzed_constraint: Any,  # _AnalyzedCost, but avoid circular import
    constraint_index: int,
    total_dim: int,
    al_params: AugmentedLagrangianParams,
) -> Any:  # Returns _AnalyzedCost
    """Create an augmented cost from a constraint for Augmented Lagrangian method.

    This creates an _AnalyzedCost object that converts a constraint into an augmented
    Lagrangian residual with per-constraint penalty parameters.

    For equality constraints h(x) = 0:
        r = sqrt(rho_i) * (h(x) + lambda_i/rho_i)

    For inequality constraints g(x) <= 0:
        r = sqrt(rho_i) * max(0, g(x) + lambda_i/rho_i)

    where lambda_i (Lagrange multipliers) and rho_i (per-constraint penalty parameters)
    are stored in the returned _AnalyzedCost's al_params field.

    Args:
        analyzed_constraint: The analyzed constraint (as _AnalyzedCost) to convert.
        constraint_index: Index of this constraint group (for accessing the right arrays).
        total_dim: Total dimension of lagrange multipliers for this constraint group
                   (constraint_count * constraint_flat_dim).
        al_params: Initial Augmented Lagrangian parameters.

    Returns:
        An _AnalyzedCost with al_params set.
    """
    from ._core import _AnalyzedCost

    is_inequality = analyzed_constraint.constraint_type == "leq_zero"
    constraint_flat_dim = analyzed_constraint.residual_flat_dim

    def augmented_residual_fn(
        vals: VarValues,
        *args_with_index_and_params,
    ) -> jax.Array:
        """Compute augmented constraint residual with per-constraint penalty.

        The second-to-last element of args is instance_index, the last is al_params.
        """
        # Split args: last is al_params, second-to-last is instance_index
        args = args_with_index_and_params[:-2]
        instance_index = args_with_index_and_params[-2]
        al_params_inner: AugmentedLagrangianParams = args_with_index_and_params[-1]

        # Compute constraint value using the original constraint function
        constraint_val = analyzed_constraint.compute_residual(vals, *args).flatten()

        # Get lambdas/rho for this specific instance
        # lambdas/rho are stored flat: [inst0_dim0, inst0_dim1, inst1_dim0, inst1_dim1, ...]
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

        # For inequality constraints: only penalize when violated (max formulation)
        # For equality constraints: always penalize deviation
        if is_inequality:
            # g(x) <= 0: penalize only when g(x) + lambda/rho > 0
            return jnp.sqrt(rho) * jnp.maximum(0.0, constraint_val + lambdas / rho)
        else:
            # h(x) = 0: always penalize
            return jnp.sqrt(rho) * (constraint_val + lambdas / rho)

    # Determine constraint count from total_dim and flat_dim
    constraint_count = total_dim // constraint_flat_dim

    # Add instance indices to args for proper slicing during vmap
    instance_indices = jnp.arange(constraint_count)

    # Broadcast al_params to have batch dimension (constraint_count,).
    # This ensures that when the cost object is vmapped, each instance gets the
    # full al_params (vmap will select the same values for each instance).
    def broadcast_to_batch(arr: jax.Array) -> jax.Array:
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
        compute_residual=augmented_residual_fn,  # type: ignore
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
