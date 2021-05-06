import dataclasses
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
from overrides import overrides

from .. import sparse, utils
from ..core._variable_assignments import VariableAssignments
from ._nonlinear_solver_base import (
    NonlinearSolverBase,
    _NonlinearSolverState,
    _TerminationCriteriaMixin,
)

if TYPE_CHECKING:
    from ..core._stacked_factor_graph import StackedFactorGraph


@utils.register_dataclass_pytree
@dataclasses.dataclass
class GaussNewtonSolver(
    NonlinearSolverBase,
    _TerminationCriteriaMixin,
):
    @overrides
    def solve(
        self,
        graph: "StackedFactorGraph",
        initial_assignments: "VariableAssignments",
    ) -> "VariableAssignments":
        # Initialize
        assignments = initial_assignments
        cost, residual_vector = graph.compute_cost(assignments)

        state = _NonlinearSolverState(
            # Using device arrays instead of native types helps avoid redundant JIT
            # compilation
            iterations=jnp.array(0),
            assignments=assignments,
            cost=cost,
            residual_vector=residual_vector,
            done=jnp.array(False),
        )
        self._print(f"Starting solve with {self}, initial cost={state.cost}")

        # Optimization
        for i in range(self.max_iterations):
            # Gauss-newton step
            state = self._step(graph, state)
            self._print(f"Iteration #{i}: cost={str(state.cost).ljust(15)}")

            # Exit if either cost threshold is met
            if state.done:
                self._print("Terminating early!")
                break

        return state.assignments

    @jax.jit
    def _step(
        self,
        graph: "StackedFactorGraph",
        state_prev: _NonlinearSolverState,
    ) -> _NonlinearSolverState:
        """Linearize, solve linear subproblem, and update on manifold."""

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
        done = self.check_convergence(
            state_prev=state_prev,
            cost_updated=cost,
            local_delta_assignments=local_delta_assignments,
            negative_gradient=ATb,
        )

        return _NonlinearSolverState(
            iterations=state_prev.iterations + 1,
            assignments=assignments,
            cost=cost,
            residual_vector=residual_vector,
            done=done,
        )
