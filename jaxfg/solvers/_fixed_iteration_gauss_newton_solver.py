import functools
from typing import TYPE_CHECKING

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from overrides import overrides

from .. import sparse
from ..core._variable_assignments import VariableAssignments
from ._nonlinear_solver_base import NonlinearSolverBase, NonlinearSolverState

if TYPE_CHECKING:
    from ..core._stacked_factor_graph import StackedFactorGraph


@jdc.pytree_dataclass
class FixedIterationGaussNewtonSolver(NonlinearSolverBase[NonlinearSolverState]):
    """Alternative version of Gauss-Newton solver, which ignores convergence checks."""

    unroll: bool = jdc.static_field(default=True)

    # To unroll the optimizer loop, we must have a concrete (static) iteration count
    iterations: int = jdc.static_field(default=10)

    @overrides
    def _initialize_state(
        self,
        graph: "StackedFactorGraph",
        initial_assignments: VariableAssignments,
    ) -> NonlinearSolverState:
        # Initialize
        cost, residual_vector = graph.compute_cost(initial_assignments)
        return NonlinearSolverState(
            iterations=0,
            assignments=initial_assignments,
            cost=cost,
            residual_vector=residual_vector,
            done=False,
        )

    @overrides
    def _step(
        self,
        graph: "StackedFactorGraph",
        state_prev: NonlinearSolverState,
    ) -> NonlinearSolverState:
        """Linearize, solve linear subproblem, and update on manifold."""

        self._hcb_print(
            lambda i, cost: f"Iteration #{i}: cost={str(cost)}",
            i=state_prev.iterations,
            cost=state_prev.cost,
        )

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
        done = state_prev.iterations >= (self.iterations - 1)

        return NonlinearSolverState(
            iterations=state_prev.iterations + 1,
            assignments=assignments,
            cost=cost,
            residual_vector=residual_vector,
            done=done,
        )

    @jax.jit
    @overrides
    def solve(
        self,
        graph: "StackedFactorGraph",
        initial_assignments: VariableAssignments,
    ) -> VariableAssignments:
        """Run MAP inference on a factor graph."""

        # Initialize
        assignments = initial_assignments
        cost, residual_vector = graph.compute_cost(assignments)
        state = self._initialize_state(graph, initial_assignments)

        # Optimization
        if self.unroll:
            for i in range(self.iterations):
                state = self._step(graph, state)
        else:
            state = jax.lax.while_loop(
                cond_fun=lambda state: jnp.logical_not(state.done),
                body_fun=functools.partial(self._step, graph),
                init_val=state,
            )

        self._hcb_print(
            lambda i, cost: f"Terminated @ iteration #{i}: cost={str(cost).ljust(15)}",
            i=state.iterations,
            cost=state.cost,
        )

        return state.assignments
