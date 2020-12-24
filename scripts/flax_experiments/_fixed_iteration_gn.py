import dataclasses
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
from jaxfg import types, utils
from jaxfg.core import VariableAssignments
from jaxfg.solvers import NonlinearSolverBase, _linear_utils
from overrides import overrides

if TYPE_CHECKING:
    from jaxfg.core import PreparedFactorGraph


@utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class _GaussNewtonState:
    """Helper for state passed between GN iterations."""

    assignments: VariableAssignments
    error: float
    error_vector: jnp.ndarray


@utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class FixedIterationGaussNewtonSolver(NonlinearSolverBase):
    @overrides
    def solve(
        self,
        graph: "PreparedFactorGraph",
        initial_assignments: VariableAssignments,
    ) -> VariableAssignments:
        # Initialize
        assignments = initial_assignments
        error, error_vector = graph.compute_sum_squared_error(assignments)
        state = _GaussNewtonState(
            assignments=assignments,
            error=error,
            error_vector=error_vector,
        )
        self._print(f"Starting solve with {self}, initial error={state.error}")

        # Optimization
        for i in range(self.max_iters):
            # Gauss-newton step
            state = self._step(graph, state)

            # Exit if either error threshold is met
            self._print(f"Iteration #{i}: error={str(state.error).ljust(15)}")

        return state.assignments

    @jax.jit
    def _step(
        self,
        graph: "PreparedFactorGraph",
        state_prev: _GaussNewtonState,
    ) -> _GaussNewtonState:
        """Linearize, solve linear subproblem, and update on manifold."""
        A: types.SparseMatrix = _linear_utils.linearize_graph(
            graph, state_prev.assignments
        )
        local_deltas = _linear_utils.sparse_linear_solve(
            A=A,
            initial_x=jnp.zeros(graph.local_storage_metadata.dim),
            b=-state_prev.error_vector,
            tol=self.atol,
            lambd=0.0,
            diagonal_damping=False,
        )
        assignments = _linear_utils.apply_local_deltas(
            state_prev.assignments,
            local_delta_assignments=VariableAssignments(
                storage=local_deltas, storage_metadata=graph.local_storage_metadata
            ),
        )
        error, error_vector = graph.compute_sum_squared_error(assignments)

        return _GaussNewtonState(
            assignments=assignments,
            error=error,
            error_vector=error_vector,
        )
