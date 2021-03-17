import dataclasses
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
from overrides import overrides

from .. import types, utils
from ..core._variable_assignments import VariableAssignments
from . import _linear_utils
from ._nonlinear_solver_base import (
    NonlinearSolverBase,
    _InexactStepSolverMixin,
    _NonlinearSolverState,
    _TerminationCriteriaMixin,
)

if TYPE_CHECKING:
    from ..core._prepared_factor_graph import StackedFactorGraph


@utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class GaussNewtonSolver(
    NonlinearSolverBase,
    _InexactStepSolverMixin,
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
                print("Terminating early!")
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
        A: types.SparseMatrix = graph.compute_jacobian(state_prev.assignments)
        ATb = A.T @ -state_prev.residual_vector

        # Solve linear subproblem
        local_delta_assignments = VariableAssignments(
            storage=_linear_utils.sparse_linear_solve(
                A=A,
                ATb=ATb,
                initial_x=jnp.zeros(graph.local_storage_metadata.dim),
                tol=self.inexact_step_forcing_sequence(state_prev.iterations),
                lambd=0.0,
            ),
            storage_metadata=graph.local_storage_metadata,
        )

        # On-manifold retraction
        assignments = state_prev.assignments.apply_local_deltas(
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
