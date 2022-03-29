import abc
import functools
from typing import TYPE_CHECKING, Callable, Generic, TypeVar, Union

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from jax.experimental import host_callback as hcb
from overrides import EnforceOverrides

from .. import hints, sparse
from ..core._variable_assignments import VariableAssignments

if TYPE_CHECKING:
    from ..core._stacked_factor_graph import StackedFactorGraph

Int = Union[hints.Array, int]
Boolean = Union[hints.Array, bool]


@jdc.pytree_dataclass
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


@jdc.pytree_dataclass
class _NonlinearSolverBase:
    # For why we have two classes:
    # https://github.com/python/mypy/issues/5374#issuecomment-650656381

    """Nonlinear solver interface."""

    verbose: Boolean = jdc.static_field(default=True)
    """Set to `True` to enable printing."""

    linear_solver: sparse.LinearSubproblemSolverBase = jdc.field(
        default_factory=lambda: sparse.CholmodSolver()
    )
    """Solver to use for linear subproblems."""


class NonlinearSolverBase(
    _NonlinearSolverBase, Generic[NonlinearSolverStateType], abc.ABC, EnforceOverrides
):
    # To be overriden by subclasses.

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

    # Shared.

    @jax.jit
    def solve(
        self,
        graph: "StackedFactorGraph",
        initial_assignments: VariableAssignments,
    ) -> VariableAssignments:
        """Run MAP inference on a factor graph."""

        # Initialize. Note that the storage layout of the initial assignments may not
        # match what the graph expects.
        assignments = initial_assignments.update_storage_layout(graph.storage_layout)
        cost, residual_vector = graph.compute_cost(assignments)
        state = self._initialize_state(graph, assignments)

        # Optimization.
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

        # Return, but with the storage layout reverted.
        return state.assignments.update_storage_layout(
            initial_assignments.storage_layout
        )

    def _hcb_print(
        self,
        string_from_args: Callable[..., str],
        *args: hints.Pytree,
        **kwargs: hints.Pytree,
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
