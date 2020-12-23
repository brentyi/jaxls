import abc
import dataclasses
from typing import TYPE_CHECKING, Tuple

import jax
import numpy as onp
from jax import numpy as jnp
from overrides import overrides

from .. import _types, _utils
from .._variable_assignments import VariableAssignments
from . import _linear

if TYPE_CHECKING:
    from .._prepared_factor_graph import PreparedFactorGraph


@dataclasses.dataclass(frozen=True)
class _NonlinearSolver:
    # See: https://github.com/python/mypy/issues/5374#issuecomment-650656381
    max_iters: int = 100
    verbose: bool = True
    atol: float = 1e-5
    rtol: float = 1e-5

    def _print(self, *args, **kwargs):
        """Prefixed printing helper. No-op if `verbose` is set to `False`."""
        if self.verbose:
            print(f"[{type(self).__name__}]", *args, **kwargs)


class NonlinearSolver(_NonlinearSolver, abc.ABC):
    @abc.abstractmethod
    def solve(
        self,
        graph: "PreparedFactorGraph",
        initial_assignments: "VariableAssignments",
    ) -> "VariableAssignments":
        """Run MAP inference on a factor graph."""


@_utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class GaussNewtonSolver(NonlinearSolver):
    @overrides
    def solve(
        self,
        graph: "PreparedFactorGraph",
        initial_assignments: "VariableAssignments",
    ) -> "VariableAssignments":
        # Initialize
        assignments = initial_assignments
        error_prev, error_vector = graph.compute_sum_squared_error(assignments)
        self._print(f"Starting solve with {self}, initial error={error_prev}")

        # Optimization
        for i in range(self.max_iters):
            # Gauss-newton step
            assignments, error, error_vector = self._step(
                graph, assignments, error_vector
            )

            # Exit if either error threshold is met
            error_delta = onp.abs(error_prev - error)
            self._print(
                f"Iteration #{i}: error={str(error).ljust(15)} error_delta={error_delta}"
            )
            if error_delta < self.atol or error_delta / error_prev < self.rtol:
                self._print("Terminating early!")
                break
            error_prev = error

        return assignments

    @jax.jit
    def _step(
        self,
        graph: "PreparedFactorGraph",
        assignments: "VariableAssignments",
        error_vector: jnp.ndarray,
    ) -> Tuple["VariableAssignments", float, jnp.ndarray]:
        """Linearize, solve linear subproblem, and update on manifold."""
        A: _types.SparseMatrix = _linear.linearize_graph(graph, assignments)
        local_deltas = _linear.sparse_linear_solve(
            A=A,
            initial_x=jnp.zeros(graph.local_storage_metadata.dim),
            b=-error_vector,
            tol=self.atol,
            lambd=0.0,
            diagonal_damping=False,
        )
        assignments = _linear.apply_local_deltas(
            assignments,
            local_delta_assignments=VariableAssignments(
                storage=local_deltas, storage_metadata=graph.local_storage_metadata
            ),
        )
        error, error_vector = graph.compute_sum_squared_error(assignments)
        return assignments, error, error_vector


@_utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class _LevenbergMarqaurdtState:
    assignments: "VariableAssignments"
    lambd: float
    error: float
    error_vector: jnp.ndarray
    done: bool

    # In the future, we may consider more sophisticated lambda updates, eg:
    # > METHODS FOR NON-LINEAR LEAST SQUARES PROBLEM, Madsen et al 2004.
    # > pg. 27, Algorithm 3.16


#
@jax.partial(_utils.register_dataclass_pytree, static_fields=("diagonal_damping",))
@dataclasses.dataclass(frozen=True)
class LevenbergMarquardtSolver(NonlinearSolver):
    """Simple damped least-squares implementation."""

    lambda_initial: float = 1e-5
    lambda_factor: float = 10.0
    lambda_min: float = 1e-10
    lambda_max: float = 1e10
    diagonal_damping: bool = True

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
            state: _LevenbergMarqaurdtState
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
    ) -> Tuple["VariableAssignments", _LevenbergMarqaurdtState]:
        """Linearize, solve linear subproblem, and accept or reject update."""
        # There's currently some redundancy here: we only need to re-linearize when
        # updates are accepted.
        A: _types.SparseMatrix = _linear.linearize_graph(graph, state_prev.assignments)
        local_deltas = _linear.sparse_linear_solve(
            A=A,
            initial_x=jnp.zeros(graph.local_storage_metadata.dim),
            b=-state_prev.error_vector,
            tol=self.atol,
            lambd=state_prev.lambd,
            diagonal_damping=False,
        )
        assignments_proposed = _linear.apply_local_deltas(
            state_prev.assignments,
            local_delta_assignments=VariableAssignments(
                storage=local_deltas, storage_metadata=graph.local_storage_metadata
            ),
        )
        error, error_vector = graph.compute_sum_squared_error(assignments_proposed)

        # Check if error dropped
        accept_flag = error <= state_prev.error

        # Update damping
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
