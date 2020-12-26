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
class _LevenbergMarqaurdtState:
    """Helper for state passed between LM iterations."""

    assignments: "VariableAssignments"
    lambd: float
    error: float
    error_vector: jnp.ndarray
    done: bool


@utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class LevenbergMarquardtSolver(NonlinearSolverBase):
    """Simple damped least-squares implementation."""

    lambda_initial: float = 1e-5
    lambda_factor: float = 2.0
    lambda_min: float = 1e-10
    lambda_max: float = 1e10

    @overrides
    def solve(
        self,
        graph: "PreparedFactorGraph",
        initial_assignments: "VariableAssignments",
    ) -> "VariableAssignments":
        # Initialize
        error_prev, error_vector = graph.compute_sum_squared_error(initial_assignments)
        self._print(f"Starting solve with {self}, initial error={error_prev}")

        state = _LevenbergMarqaurdtState(
            assignments=initial_assignments,
            lambd=self.lambda_initial,
            error=error_prev,
            error_vector=error_vector,
            done=False,
        )

        # Optimization
        for i in range(self.max_iters):
            # LM step
            state = self._step(graph, state)
            self._print(f"Iteration #{i}: error={str(state.error).ljust(15)}")
            if state.done:
                self._print("Terminating early!")
                break

        return state.assignments

    @jax.jit
    def _step(
        self,
        graph: "PreparedFactorGraph",
        state_prev: _LevenbergMarqaurdtState,
    ) -> _LevenbergMarqaurdtState:
        """Linearize, solve linear subproblem, and accept or reject update."""
        # There's currently some redundancy here: we only need to re-linearize when
        # updates are accepted.
        A: types.SparseMatrix = _linear_utils.linearize_graph(
            graph, state_prev.assignments
        )
        local_deltas = _linear_utils.sparse_linear_solve(
            A=A,
            initial_x=jnp.zeros(graph.local_storage_metadata.dim),
            b=-state_prev.error_vector,
            tol=self.rtol,
            atol=self.atol,
            lambd=state_prev.lambd,
        )
        assignments_proposed = _linear_utils.apply_local_deltas(
            state_prev.assignments,
            local_delta_assignments=VariableAssignments(
                storage=local_deltas, storage_metadata=graph.local_storage_metadata
            ),
        )
        error, error_vector = graph.compute_sum_squared_error(assignments_proposed)

        # Check if error dropped
        accept_flag = error <= state_prev.error

        # Update damping
        # In the future, we may consider more sophisticated lambda updates, eg:
        # > METHODS FOR NON-LINEAR LEAST SQUARES PROBLEM, Madsen et al 2004.
        # > pg. 27, Algorithm 3.16
        lambd = jnp.where(
            accept_flag,
            jnp.maximum(state_prev.lambd / self.lambda_factor, self.lambda_min),
            jnp.minimum(state_prev.lambd * self.lambda_factor, self.lambda_max),
        )

        # Get output assignments
        assignments = dataclasses.replace(
            state_prev.assignments,
            storage=jnp.where(
                accept_flag,
                assignments_proposed.storage,
                state_prev.assignments.storage,
            ),
        )

        # Determine whether or not we want to terminate
        error_delta = jnp.abs(state_prev.error - error)
        done = jnp.logical_and(
            accept_flag,
            jnp.logical_or(
                error_delta < self.atol,
                error_delta / state_prev.error < self.rtol,
            ),
        )

        return _LevenbergMarqaurdtState(
            assignments=assignments,
            lambd=lambd,
            error=error,
            error_vector=error_vector,
            done=done,
        )
