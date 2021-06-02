import jax
import jax_dataclasses
from jax import numpy as jnp

from .. import hints, sparse
from ..core._variable_assignments import VariableAssignments
from ._nonlinear_solver_base import NonlinearSolverState


@jax_dataclasses.pytree_dataclass
class _TrustRegionMixin:
    """Mixin that implements a step quality check for trust region solvers."""

    step_quality_min: hints.Scalar = 1e-3

    def compute_step_quality(
        self,
        A: sparse.SparseCooMatrix,
        proposed_cost: hints.Scalar,
        state_prev: NonlinearSolverState,
        step_vector: jnp.ndarray,
    ) -> bool:
        """Compute step quality ratio, often denoted $$\rho$$.
        This will be 1 when the cost drops linearly wrt the update step."""
        return (proposed_cost - state_prev.cost) / (
            jnp.sum((A @ step_vector + state_prev.residual_vector) ** 2)
            - state_prev.cost
        )


@jax_dataclasses.pytree_dataclass
class _TerminationCriteriaMixin:
    """Mixin for Ceres-style termination criteria."""

    cost_tolerance: float = 1e-5
    """We terminate if `|cost change| / cost < cost_tolerance`."""

    gradient_tolerance: float = 1e-9
    """We terminate if `norm_inf(x - rplus(x, linear delta)) < gradient_tolerance`."""

    gradient_tolerance_start_step: int = 10
    """When to start checking the gradient tolerance condition. Helps solve precision
    issues caused by inexact Newton steps."""

    parameter_tolerance: float = 1e-7
    """We terminate if `norm_2(linear delta) < (norm2(x) + parameter_tolerance) * parameter_tolerance`."""

    @jax.jit
    def check_convergence(
        self,
        state_prev: NonlinearSolverState,
        cost_updated: hints.Scalar,
        local_delta_assignments: VariableAssignments,
        negative_gradient: hints.Array,
    ) -> bool:
        """Check for convergence!"""

        # Cost tolerance
        converged_cost = (
            jnp.abs(cost_updated - state_prev.cost) / state_prev.cost
            < self.cost_tolerance
        )

        # Gradient tolerance
        converged_gradient = jnp.where(
            state_prev.iterations >= self.gradient_tolerance_start_step,
            jnp.max(
                state_prev.assignments.storage
                - state_prev.assignments.manifold_retract(
                    VariableAssignments(
                        storage=negative_gradient,
                        storage_metadata=local_delta_assignments.storage_metadata,
                    ),
                ).storage
            )
            < self.gradient_tolerance,
            False,
        )

        # Parameter tolerance
        converged_parameters = (
            jnp.linalg.norm(jnp.abs(local_delta_assignments.storage))
            < (
                jnp.linalg.norm(state_prev.assignments.storage)
                + self.parameter_tolerance
            )
            * self.parameter_tolerance
        )

        return jnp.logical_or(
            converged_cost,
            jnp.logical_or(
                converged_gradient,
                converged_parameters,
            ),
        )
