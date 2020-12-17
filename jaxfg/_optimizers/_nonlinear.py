import abc
import dataclasses
from typing import TYPE_CHECKING

import numpy as onp
from overrides import overrides

if TYPE_CHECKING:
    from .._prepared_factor_graph import PreparedFactorGraph
    from .._variable_assignments import VariableAssignments

from . import _linear


@dataclasses.dataclass(frozen=True)
class NonlinearSolver(abc.ABC):
    max_iters: int = 100
    atol: float = 1e-5
    rtol: float = 1e-5
    verbose: bool = True

    @abc.abstractmethod
    def solve(
        self,
        graph: "PreparedFactorGraph",
        initial_assignments: "VariableAssignments",
    ) -> "VariableAssignments":
        """Run MAP inference on a factor graph."""

    def _print(self, *args, **kwargs):
        """Prefixed printing helper. No-op if `verbose` is set to `False`."""
        if self.verbose:
            print(f"[{type(self).__name__}]", *args, **kwargs)


@dataclasses.dataclass(frozen=True)
class GaussNewtonSolver(NonlinearSolver):
    @overrides
    def solve(
        self,
        graph: "PreparedFactorGraph",
        initial_assignments: "VariableAssignments",
    ) -> "VariableAssignments":

        self._print(f"Starting solve with {self}")

        # First iteration
        assignments, prev_error = _linear.gauss_newton_step(
            graph, initial_assignments, tol=self.atol * 10
        )
        self._print(f"Iteration #0: error={prev_error}")

        # Remaining iterations
        for i in range(self.max_iters - 1):
            assignments, error = _linear.gauss_newton_step(
                graph, assignments, tol=self.atol
            )

            # Exit if either error threshold is met
            error_delta = onp.abs(prev_error - error)
            self._print(
                f"Iteration #{i + 1}: error={str(error).ljust(15)} error_delta={error_delta}"
            )
            if error_delta < self.atol or error_delta / prev_error < self.rtol:
                self._print("Terminating early!")
                break
            prev_error = error

        return assignments
