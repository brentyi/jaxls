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
    from ..core._prepared_factor_graph import PreparedFactorGraph


@utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class _LevenbergMarqaurdtState(_NonlinearSolverState):
    """Helper for state passed between LM iterations."""

    lambd: float


@utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class LevenbergMarquardtSolver(
    NonlinearSolverBase, _InexactStepSolverMixin, _TerminationCriteriaMixin
):
    """Simple damped least-squares implementation."""

    lambda_initial: float = 5e-4
    lambda_factor: float = 2.0
    lambda_min: float = 1e-6
    lambda_max: float = 1e10

    @overrides
    def solve(
        self,
        graph: "PreparedFactorGraph",
        initial_assignments: "VariableAssignments",
    ) -> "VariableAssignments":
        # Initialize
        cost_prev, residual_vector = graph.compute_cost(initial_assignments)
        self._print(f"Starting solve with {self}, initial cost={cost_prev}")

        state = _LevenbergMarqaurdtState(
            iterations=0,
            assignments=initial_assignments,
            lambd=self.lambda_initial,
            cost=cost_prev,
            residual_vector=residual_vector,
            done=False,
        )

        # Optimization
        for i in range(self.max_iterations):
            # LM step
            state = self._step(graph, state)
            self._print(
                f"Iteration #{i}: cost={str(state.cost).ljust(15)} lambda={str(state.lambd)}"
            )
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
        A: types.SparseMatrix = graph.compute_jacobian(state_prev.assignments)
        ATb = A.T @ -state_prev.residual_vector
        local_delta_assignments = VariableAssignments(
            storage=_linear_utils.sparse_linear_solve(
                A=A,
                ATb=ATb,
                initial_x=jnp.zeros(graph.local_storage_metadata.dim),
                tol=self.inexact_step_forcing_sequence(state_prev.iterations),
                lambd=state_prev.lambd,
            ),
            storage_metadata=graph.local_storage_metadata,
        )
        assignments_proposed = state_prev.assignments.apply_local_deltas(
            local_delta_assignments=local_delta_assignments
        )
        cost, residual_vector = graph.compute_cost(assignments_proposed)

        # Check if cost dropped
        accept_flag = cost <= state_prev.cost

        # Update damping
        # In the future, we may consider more sophisticated lambda updates, eg:
        # > METHODS FOR NON-LINEAR LEAST SQUARES PROBLEM, Madsen et al 2004.
        # > pg. 27, Algorithm 3.16
        lambd = jnp.where(
            accept_flag,
            # If accept, decrease damping... note that we *don't* enforce any bounds here
            state_prev.lambd / self.lambda_factor,
            # If reject: increase lambda and enforce bounds
            jnp.maximum(
                self.lambda_min,
                jnp.minimum(state_prev.lambd * self.lambda_factor, self.lambda_max),
            ),
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

        # Check for convergence
        done = jnp.logical_and(
            accept_flag,
            self.check_convergence(
                state_prev=state_prev,
                cost_updated=cost,
                local_delta_assignments=local_delta_assignments,
                negative_gradient=ATb,
            ),
        )

        return _LevenbergMarqaurdtState(
            iterations=state_prev.iterations + 1,
            assignments=assignments,
            lambd=lambd,
            cost=jnp.where(
                accept_flag, cost, state_prev.cost
            ),  # Use old cost if update is rejected
            residual_vector=residual_vector,
            done=done,
        )
