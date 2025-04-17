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
            utils.print_deprecation_warning(
                "[bold]jaxls.Factor[/bold] has been renamed to [bold]jaxls.Cost[/bold]"
            )

            # Standard warning is now handled by print_deprecation_warning
            return Cost

    # Create a descriptor for FactorGraph that shows a deprecation warning when accessed
    class _FactorGraphDescriptor:
        def __get__(self, obj: Any, objtype: Any = None) -> Any:
            utils.print_deprecation_warning(
                "[bold]jaxls.FactorGraph.make(...)[/bold] has been replaced with [bold]jaxls.LeastSquaresProblem(...).analyze()[/bold]"
            )

            # Standard warning is now handled by print_deprecation_warning

            class FactorGraph:
                @staticmethod
                def make(factors, variables, use_onp=False):
                    utils.print_deprecation_warning(
                        "[bold]jaxls.FactorGraph[/bold] has been renamed [bold]jaxls.LeastSquaresProblem[/bold]"
                    )

                    # Standard warning is now handled by print_deprecation_warning

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
