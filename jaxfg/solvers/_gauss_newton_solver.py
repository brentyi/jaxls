import dataclasses
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
from overrides import overrides

from .. import types, utils
from ..core._variable_assignments import VariableAssignments
from . import _linear_utils
from ._nonlinear_solver_base import NonlinearSolverBase

if TYPE_CHECKING:
    from .._prepared_factor_graph import PreparedFactorGraph


@utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class _GaussNewtonState:
    """Helper for state passed between GN iterations."""

    iterations: int
    assignments: "VariableAssignments"
    error: float
    error_vector: jnp.ndarray
    done: bool


@utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class GaussNewtonSolver(NonlinearSolverBase):
    @overrides
    def solve(
        self,
        graph: "PreparedFactorGraph",
        initial_assignments: "VariableAssignments",
    ) -> "VariableAssignments":
        # Initialize
        assignments = initial_assignments
        error, error_vector = graph.compute_sum_squared_error(assignments)
        state = _GaussNewtonState(
            iterations=0,
            assignments=assignments,
            error=error,
            error_vector=error_vector,
            done=False,
        )
        self._print(f"Starting solve with {self}, initial error={state.error}")

        # Optimization
        for i in range(self.max_iters):
            # Gauss-newton step
            state = self._step(graph, state)

            # Exit if either error threshold is met
            self._print(f"Iteration #{i}: error={str(state.error).ljust(15)}")
            if state.done:
                self._print("Terminating early!")
                break

        return state.assignments

    # @jax.jit
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
            tol=self.inexact_step_forcing_sequence(state_prev.iterations),
            atol=self.atol,
            lambd=0.0,
        )
        assignments = _linear_utils.apply_local_deltas(
            state_prev.assignments,
            local_delta_assignments=VariableAssignments(
                storage=local_deltas, storage_metadata=graph.local_storage_metadata
            ),
        )
        error, error_vector = graph.compute_sum_squared_error(assignments)

        # Determine whether or not we want to terminate
        error_delta = jnp.abs(state_prev.error - error)
        done = jnp.logical_or(
            error_delta < self.atol,
            error_delta / state_prev.error < self.rtol,
        )

        return _GaussNewtonState(
            iterations=state_prev.iterations + 1,
            assignments=assignments,
            error=error,
            error_vector=error_vector,
            done=done,
        )
