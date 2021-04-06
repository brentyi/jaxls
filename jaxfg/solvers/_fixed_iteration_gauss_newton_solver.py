import dataclasses
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
from overrides import overrides

from .. import sparse, utils
from ..core._variable_assignments import VariableAssignments
from ._nonlinear_solver_base import (
    NonlinearSolverBase,
    _InexactStepSolverMixin,
    _NonlinearSolverState,
)

if TYPE_CHECKING:
    from ..core._stacked_factor_graph import StackedFactorGraph


@utils.register_dataclass_pytree
@dataclasses.dataclass
class FixedIterationGaussNewtonSolver(
    NonlinearSolverBase,
    _InexactStepSolverMixin,
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

        return state.assignments

    @jax.jit
    def _step(
        self,
        graph: "StackedFactorGraph",
        state_prev: _NonlinearSolverState,
    ) -> _NonlinearSolverState:
        """Linearize, solve linear subproblem, and update on manifold."""

        # Linearize graph
        A: sparse.SparseCooMatrix = graph.compute_residual_jacobian(
            state_prev.assignments
        )
        ATb = A.T @ -state_prev.residual_vector

        # Solve linear subproblem
        local_delta_assignments = VariableAssignments(
            storage=sparse.linear_solve(
                A=A,
                ATb=ATb,
                initial_x=jnp.zeros(graph.local_storage_metadata.dim),
                tol=self.inexact_step_forcing_sequence(state_prev.iterations),
                lambd=0.0,
            ),
            storage_metadata=graph.local_storage_metadata,
        )

        # On-manifold retraction
        assignments = state_prev.assignments.manifold_retract(
            local_delta_assignments=local_delta_assignments
        )

        # Re-compute cost / residual
        cost, residual_vector = graph.compute_cost(assignments)
        done = False

        return _NonlinearSolverState(
            iterations=state_prev.iterations + 1,
            assignments=assignments,
            cost=cost,
            residual_vector=residual_vector,
            done=done,
        )
