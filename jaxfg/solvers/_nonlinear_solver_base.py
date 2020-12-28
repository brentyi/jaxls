import abc
import dataclasses
from typing import TYPE_CHECKING, Tuple

import jax
import numpy as onp
from jax import numpy as jnp
from overrides import overrides

from .. import types, utils
from ..core._variable_assignments import VariableAssignments
from . import _linear_utils

if TYPE_CHECKING:
    from .._prepared_factor_graph import PreparedFactorGraph


# For why we have two classes: https://github.com/python/mypy/issues/5374#issuecomment-650656381
@dataclasses.dataclass(frozen=True)
class _NonlinearSolverBase:
    max_iters: int = 100
    verbose: bool = True

    inexact_step_eta: float = 1e-1
    """Forcing sequence parameter for inexact Newton steps. CG tolerance is set to
    `eta / iteration #`.

    For reference, see AN INEXACT LEVENBERG-MARQUARDT METHOD FOR LARGE SPARSE NONLINEAR
    LEAST SQUARES, Wright & Holt 1983."""

    # Termination criteria is same as GTSAM/minisam but feels pretty sketchy; looking
    # at delta magnitudes seems more theoretically grounded?
    #
    # We're also re-using these values for the CGLS tolerances which seems OK
    # practically, but these actually refer to completely different quantities.
    atol: float = 1e-4
    """Absolute termination threshold."""
    rtol: float = 1e-4
    """Relative termination threshold."""

    def inexact_step_forcing_sequence(self, iterations: int) -> float:
        """Get CGLS tolerance from zero-indexed iteration count."""
        return jnp.maximum(self.inexact_step_eta / (iterations + 1), self.rtol)

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
