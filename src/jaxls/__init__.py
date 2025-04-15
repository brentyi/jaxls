import sys
import warnings
from typing import TYPE_CHECKING, Any

from . import utils as utils
from ._core import AnalyzedLeastSquaresProblem as AnalyzedLeastSquaresProblem
from ._core import Cost as Cost
from ._core import LeastSquaresProblem as LeastSquaresProblem
from ._lie_group_variables import SE2Var as SE2Var
from ._lie_group_variables import SE3Var as SE3Var
from ._lie_group_variables import SO2Var as SO2Var
from ._lie_group_variables import SO3Var as SO3Var
from ._solvers import ConjugateGradientConfig as ConjugateGradientConfig
from ._solvers import TerminationConfig as TerminationConfig
from ._solvers import TrustRegionConfig as TrustRegionConfig
from ._variables import Var as Var
from ._variables import VarValues as VarValues

# Some shims for backwards compatibility.
if not TYPE_CHECKING:
    # Create a descriptor for Factor that shows a deprecation warning when accessed
    class _FactorDescriptor:
        def __get__(self, obj: Any, objtype: Any = None) -> Any:
            warnings.warn(
                "`jaxls.Factor` has been renamed to `jaxls.Cost`",
                DeprecationWarning,
                stacklevel=2,
            )
            return Cost

    # Create a descriptor for FactorGraph that shows a deprecation warning when accessed
    class _FactorGraphDescriptor:
        def __get__(self, obj: Any, objtype: Any = None) -> Any:
            warnings.warn(
                "`jaxls.FactorGraph.make(...)` has been replaced with `jaxls.LeastSquaresProblem(...).analyze()`",
                DeprecationWarning,
                stacklevel=2,
            )

            class FactorGraph:
                @staticmethod
                def make(factors, variables, use_onp=False):
                    if "factors" in kwargs:
                        kwargs["costs"] = kwargs.pop("factors")

                    warnings.warn(
                        "`jaxls.FactorGraph` has been renamed `jaxls.LeastSquaresProblem`",
                        DeprecationWarning,
                        stacklevel=2,
                    )

                    return LeastSquaresProblem(
                        costs=factors, variables=variables
                    ).analyze(use_onp)

            return FactorGraph

    # Use the descriptors to trigger warnings when Factor or FactorGraph are accessed
    sys.modules[__name__].__class__ = type(
        sys.modules[__name__].__class__.__name__,
        (sys.modules[__name__].__class__,),
        {
            "Factor": _FactorDescriptor(),
            "FactorGraph": _FactorGraphDescriptor(),
        },
    )
