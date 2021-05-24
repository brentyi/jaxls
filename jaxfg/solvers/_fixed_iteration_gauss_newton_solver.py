import dataclasses
from typing import TYPE_CHECKING

import jax
from overrides import overrides

from .. import utils
from ..core._variable_assignments import VariableAssignments
from ._gauss_newton_solver import GaussNewtonSolver

if TYPE_CHECKING:
    from ..core._stacked_factor_graph import StackedFactorGraph


@utils.register_dataclass_pytree
@dataclasses.dataclass
class FixedIterationGaussNewtonSolver(GaussNewtonSolver):
    """Alternative version of Gauss-Newton solver, which ignores convergence checks."""

    unroll: bool = dataclasses.field(default=True, metadata=utils.static_field())

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
            for i in range(self.max_iterations):
                state = self._step(graph, state)
        else:
            state = jax.lax.fori_loop(
                lower=0,
                upper=self.max_iterations,
                body_fun=lambda _unused_i, state: self._step(graph, state),
                init_val=state,
            )

        self._hcb_print(
            lambda i, max_i, cost: f"Terminated @ iteration #{i}/{max_i}: cost={str(cost).ljust(15)}",
            i=state.iterations,
            max_i=self.max_iterations,
            cost=state.cost,
        )

        return state.assignments
