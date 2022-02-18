from typing import TYPE_CHECKING

import jax_dataclasses as jdc
from jax import numpy as jnp
from overrides import overrides

from .. import sparse
from ..core._variable_assignments import VariableAssignments
from ._mixins import _TerminationCriteriaMixin
from ._nonlinear_solver_base import NonlinearSolverBase, NonlinearSolverState

if TYPE_CHECKING:
    from ..core._stacked_factor_graph import StackedFactorGraph


@jdc.pytree_dataclass
class GaussNewtonSolver(
    NonlinearSolverBase[NonlinearSolverState],
    _TerminationCriteriaMixin,
):
    @overrides
    def _initialize_state(
        self,
        graph: "StackedFactorGraph",
        initial_assignments: VariableAssignments,
    ) -> NonlinearSolverState:
        # Initialize
        cost, residual_vector = graph.compute_cost(initial_assignments)
        return NonlinearSolverState(
            iterations=0,
            assignments=initial_assignments,
            cost=cost,
            residual_vector=residual_vector,
            done=False,
        )

    @overrides
    def _step(
        self,
        graph: "StackedFactorGraph",
        state_prev: NonlinearSolverState,
    ) -> NonlinearSolverState:
        """Linearize, solve linear subproblem, and update on manifold."""

        self._hcb_print(
            lambda i, max_i, cost: f"Iteration #{i}/{max_i}: cost={str(cost)}",
            i=state_prev.iterations,
            max_i=self.max_iterations,
            cost=state_prev.cost,
        )

        # Linearize graph
        A: sparse.SparseCooMatrix = graph.compute_whitened_residual_jacobian(
            assignments=state_prev.assignments,
            residual_vector=state_prev.residual_vector,
        )
        ATb = -(A.T @ state_prev.residual_vector)

        # Solve linear subproblem
        local_delta_assignments = VariableAssignments(
            storage=self.linear_solver.solve_subproblem(
                A=A,
                ATb=ATb,
                lambd=0.0,
                iteration=state_prev.iterations,
            ),
            storage_metadata=graph.local_storage_metadata,
        )

        # On-manifold retraction
        assignments = state_prev.assignments.manifold_retract(
            local_delta_assignments=local_delta_assignments,
        )

        # Check for convergence
        cost, residual_vector = graph.compute_cost(assignments)
        done = jnp.logical_or(
            self.check_exceeded_max_iterations(state_prev=state_prev),
            self.check_convergence(
                state_prev=state_prev,
                cost_updated=cost,
                local_delta_assignments=local_delta_assignments,
                negative_gradient=ATb,
            ),
        )

        return NonlinearSolverState(
            iterations=state_prev.iterations + 1,
            assignments=assignments,
            cost=cost,
            residual_vector=residual_vector,
            done=done,
        )
