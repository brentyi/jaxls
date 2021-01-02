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
)

if TYPE_CHECKING:
    from ..core._prepared_factor_graph import PreparedFactorGraph


@utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class FixedIterationGaussNewtonSolver(NonlinearSolverBase, _InexactStepSolverMixin):
    @overrides
    def solve(
        self,
        graph: "PreparedFactorGraph",
        initial_assignments: "VariableAssignments",
    ) -> "VariableAssignments":
        # Initialize
        assignments = initial_assignments
        cost, residual_vector = graph.compute_cost(assignments)
        state = _NonlinearSolverState(
            iterations=0,
            assignments=assignments,
            cost=cost,
            residual_vector=residual_vector,
            done=False,
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
        graph: "PreparedFactorGraph",
        state_prev: _NonlinearSolverState,
    ) -> _NonlinearSolverState:
        """Linearize, solve linear subproblem, and update on manifold."""

        # Linearize graph
        A: types.SparseMatrix = _linear_utils.linearize_graph(
            graph, state_prev.assignments
        )
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
        assignments = _linear_utils.apply_local_deltas(
            state_prev.assignments,
            local_delta_assignments=local_delta_assignments,
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
