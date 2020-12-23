import abc
import dataclasses
from typing import TYPE_CHECKING, Tuple

import jax
import numpy as onp
from jax import numpy as jnp
from overrides import overrides

from .. import _types, _utils
from .._variable_assignments import VariableAssignments
from . import _linear_utils

if TYPE_CHECKING:
    from .._prepared_factor_graph import PreparedFactorGraph


# For why we have two classes: https://github.com/python/mypy/issues/5374#issuecomment-650656381
@dataclasses.dataclass(frozen=True)
class _NonlinearSolverBase:
    max_iters: int = 100
    verbose: bool = True

    # Stopping criteria
    atol: float = 1e-4
    rtol: float = 1e-4

    def _print(self, *args, **kwargs):
        """Prefixed printing helper. No-op if `verbose` is set to `False`."""
        if self.verbose:
            print(f"[{type(self).__name__}]", *args, **kwargs)

class NonlinearSolverBase(_NonlinearSolverBase, abc.ABC):
    @abc.abstractmethod
    def solve(
        self,
        graph: "PreparedFactorGraph",
        initial_assignments: "VariableAssignments",
    ) -> "VariableAssignments":
        """Run MAP inference on a factor graph."""
