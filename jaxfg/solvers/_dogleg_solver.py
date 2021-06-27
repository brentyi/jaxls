from typing import TYPE_CHECKING

import jax_dataclasses
from jax import numpy as jnp
from overrides import overrides

from .. import hints, sparse
from ..core._variable_assignments import VariableAssignments
from ._mixins import _TerminationCriteriaMixin, _TrustRegionMixin
from ._nonlinear_solver_base import NonlinearSolverBase, NonlinearSolverState

if TYPE_CHECKING:
    from ..core._stacked_factor_graph import StackedFactorGraph


@jax_dataclasses.pytree_dataclass
class _DoglegState(NonlinearSolverState):
    """State passed between dogleg iterations."""

    radius: hints.Scalar


@jax_dataclasses.pytree_dataclass
class DoglegSolver(
    NonlinearSolverBase[_DoglegState],
    _TerminationCriteriaMixin,
    _TrustRegionMixin,
):
    """Traditional dogleg algorithm."""

    radius_initial: hints.Scalar = 1.0

    @overrides
    def _initialize_state(
        self,
        graph: "StackedFactorGraph",
        initial_assignments: VariableAssignments,
    ) -> _DoglegState:
        # Initialize
        cost, residual_vector = graph.compute_cost(initial_assignments)
        return _DoglegState(
            iterations=0,
            assignments=initial_assignments,
            cost=cost,
            residual_vector=residual_vector,
            done=False,
            radius=self.radius_initial,
        )

    @overrides
    def _step(
        self,
        graph: "StackedFactorGraph",
        state_prev: _DoglegState,
    ) -> _DoglegState:
        # There's currently some redundancy here: we only need to re-linearize and
        # compute new GN/SD update steps when updates are actually accepted
        self._hcb_print(
            lambda i, max_i, cost, radius: f"Iteration #{i}/{max_i}: cost={str(cost).ljust(15)} radius={str(radius)}",
            i=state_prev.iterations,
            max_i=self.max_iterations,
            cost=state_prev.cost,
            radius=state_prev.radius,
        )

        # Linearize graph
        A: sparse.SparseCooMatrix = graph.compute_whitened_residual_jacobian(
            assignments=state_prev.assignments,
            residual_vector=state_prev.residual_vector,
        )
        ATb = A.T @ -state_prev.residual_vector

        def compute_dogleg_step() -> jnp.ndarray:
            # Gauss-Newton step
            step_vector_gn: jnp.ndarray = self.linear_solver.solve_subproblem(
                A=A,
                ATb=ATb,
                lambd=0.0,
                iteration=state_prev.iterations,
            )

            # Steepest descent step
            step_vector_sd: jnp.ndarray = ATb

            # Blending parameters
            # Reference:
            # > METHODS FOR NON-LINEAR LEAST SQUARES PROBLEM, Madsen et al 2004.
            # > pg. 30~32
            alpha = jnp.sum(step_vector_sd ** 2) / jnp.sum((A @ step_vector_sd) ** 2)
            a = alpha * step_vector_sd
            b = step_vector_gn

            c = jnp.sum(a * (b - a))
            a_norm_sq = jnp.sum(a ** 2)
            b_minus_a_norm_sq = jnp.sum((b - a) ** 2)
            radius_sq = state_prev.radius ** 2
            radius_sq_minus_a_norm_sq = radius_sq - a_norm_sq
            sqrt_c_sq_plus = jnp.sqrt(
                c ** 2 + b_minus_a_norm_sq * radius_sq_minus_a_norm_sq
            )
            beta = jnp.where(
                c <= 0,
                (-c + sqrt_c_sq_plus) / b_minus_a_norm_sq,
                radius_sq_minus_a_norm_sq / (c + sqrt_c_sq_plus),
            )

            # Compute update step
            norm_gn = jnp.linalg.norm(step_vector_gn)
            norm_sd = jnp.linalg.norm(a)
            return jnp.where(
                # Use GN if it's within trust region
                norm_gn <= state_prev.radius,
                step_vector_gn,
                jnp.where(
                    # Use normed SD if neither are within trust region
                    norm_sd >= state_prev.radius,
                    state_prev.radius / norm_sd * a,
                    a + beta * (step_vector_gn - a),
                ),
            )

        step_vector = compute_dogleg_step()

        local_delta_assignments = VariableAssignments(
            storage=step_vector,
            storage_metadata=graph.local_storage_metadata,
        )

        # On-manifold retraction + solution check
        assignments_proposed = state_prev.assignments.manifold_retract(
            local_delta_assignments=local_delta_assignments
        )
        proposed_cost, residual_vector = graph.compute_cost(assignments_proposed)
        step_quality = self.compute_step_quality(
            A=A,
            proposed_cost=proposed_cost,
            state_prev=state_prev,
            step_vector=step_vector,
        )
        accept_flag = step_quality >= self.step_quality_min

        # Update trust region
        radius = jnp.where(
            step_quality < 0.25,
            state_prev.radius / 2.0,
            jnp.where(
                step_quality > 0.75,
                jnp.max(
                    jnp.array([state_prev.radius, 3.0 * jnp.linalg.norm(step_vector)])
                ),
                state_prev.radius,
            ),
        )

        # Get output assignments
        assignments = jax_dataclasses.replace(
            state_prev.assignments,
            storage=jnp.where(
                accept_flag,
                assignments_proposed.storage,
                state_prev.assignments.storage,
            ),
        )

        # Check for convergence
        done = jnp.logical_and(
            accept_flag,
            self.check_convergence(
                state_prev=state_prev,
                cost_updated=proposed_cost,
                local_delta_assignments=local_delta_assignments,
                negative_gradient=ATb,
            ),
        )

        return _DoglegState(
            iterations=state_prev.iterations + 1,
            assignments=assignments,
            radius=radius,
            cost=jnp.where(
                accept_flag, proposed_cost, state_prev.cost
            ),  # Use old cost if update is rejected
            residual_vector=residual_vector,
            done=done,
        )
