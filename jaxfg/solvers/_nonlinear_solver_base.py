import abc
from typing import TYPE_CHECKING, Callable, Generic, TypeVar, Union

import jax
import jax_dataclasses
from jax import numpy as jnp
from jax.experimental import host_callback as hcb
from overrides import EnforceOverrides

from .. import hints, sparse
from ..core._variable_assignments import VariableAssignments

if TYPE_CHECKING:
    from ..core._stacked_factor_graph import StackedFactorGraph

Int = Union[hints.Array, int]
Boolean = Union[hints.Array, bool]


@jax_dataclasses.dataclass
class NonlinearSolverState:
    """Standard state passed between nonlinear solve iterations."""

    iterations: Int
    assignments: "VariableAssignments"
    cost: hints.Scalar
    residual_vector: hints.Array
    done: Boolean


NonlinearSolverStateType = TypeVar(
    "NonlinearSolverStateType", bound=NonlinearSolverState
)


@jax_dataclasses.dataclass
class _NonlinearSolverBase:
    # For why we have two classes:
    # https://github.com/python/mypy/issues/5374#issuecomment-650656381

    """Nonlinear solver interface."""

    max_iterations: int = 100
    """Maximum number of iterations."""

    verbose: Boolean = jax_dataclasses.static_field(default=True)
    """Set to `True` to enable printing."""

    linear_solver: sparse.LinearSubproblemSolverBase = jax_dataclasses.field(
        default_factory=lambda: sparse.CholmodSolver()
    )
    """Solver to use for linear subproblems."""


class NonlinearSolverBase(
    _NonlinearSolverBase, Generic[NonlinearSolverStateType], abc.ABC, EnforceOverrides
):
    # To be overriden by subclasses

    @abc.abstractmethod
    def _initialize_state(
        self,
        graph: "StackedFactorGraph",
        initial_assignments: VariableAssignments,
    ) -> NonlinearSolverStateType:
        ...

    @abc.abstractmethod
    def _step(
        self,
        graph: "StackedFactorGraph",
        state_prev: NonlinearSolverStateType,
    ) -> NonlinearSolverStateType:
        """Single nonlinear optimization step."""

    # Shared

    @jax.jit
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
        state = jax.lax.while_loop(
            cond_fun=lambda state: jnp.logical_and(
                jnp.logical_not(state.done), state.iterations < self.max_iterations
            ),
            body_fun=jax.partial(self._step, graph),
            init_val=state,
        )

        self._hcb_print(
            lambda i, max_i, cost: f"Terminated @ iteration #{i}/{max_i}: cost={str(cost).ljust(15)}",
            i=state.iterations,
            max_i=self.max_iterations,
            cost=state.cost,
        )

        return state.assignments

    def _hcb_print(
        self,
        string_from_args: Callable[..., str],
        *args: hints.PyTree,
        **kwargs: hints.PyTree,
    ) -> None:
        """Helper for printer optimizer messages via host callbacks. No-op if `verbose`
        is set to `False`."""

        if not self.verbose:
            return

        hcb.id_tap(
            lambda args_kwargs, _unused_transforms: print(
                f"[{type(self).__name__}]",
                string_from_args(*args_kwargs[0], **args_kwargs[1]),
            ),
            (args, kwargs),
        )
