import dataclasses
from typing import TYPE_CHECKING, Tuple

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


@dataclasses.dataclass(frozen=True)
class FixedIterationGaussNewtonSolver(NonlinearSolverBase):
    @jax.partial(jax.jit, static_argnums=0)
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
        state, unused_errors = jax.lax.scan(
            f=lambda state, x: self._step(graph, state),
            init=state,
            xs=jnp.zeros(self.max_iters),
        )

        return state.assignments

    def _step(
        self,
        graph: "PreparedFactorGraph",
        state_prev: _GaussNewtonState,
    ) -> Tuple[_GaussNewtonState, float]:
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

        return (
            _GaussNewtonState(
                assignments=assignments,
                error=error,
                error_vector=error_vector,
            ),
            error,
        )
