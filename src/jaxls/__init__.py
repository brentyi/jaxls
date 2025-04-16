import inspect
import sys
import warnings
from typing import TYPE_CHECKING, Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

# Create a console for pretty printing
_console = Console(stderr=True) if _HAS_RICH else None


def _print_deprecation_warning(message: str, caller_frame):
    """Print a nicely formatted deprecation warning with code context."""
    # Get more context (5 lines)
    frame_info = inspect.getframeinfo(caller_frame, context=5)
    filename = frame_info.filename
    lineno = frame_info.lineno

    if _HAS_RICH:
        from rich.console import Group

        # Create the content for the panel
        warning_message = Text(message, style="yellow bold")

        # Add a horizontal rule for separation
        from rich.rule import Rule

        separator = Rule(style="dim")

        panel_content = [warning_message, separator]

        # Add code context if available
        if frame_info.code_context:
            # Get the original code context
            code_lines = frame_info.code_context
            start_line = lineno - (len(code_lines) // 2)

            # Insert a comment as the first line of the code context
            modified_code_lines = list(code_lines)
            modified_code_lines.insert(0, f"# Called from: {filename}:{lineno}\n")

            # Combine into a single string
            code = "".join(modified_code_lines)

            # Calculate the line number that should be highlighted
            # It's the original target line plus 1 (for our comment)
            highlight_line = start_line + (lineno - start_line) + 1

            # Create syntax highlighting with the current line highlighted
            syntax = Syntax(
                code,
                "python",
                line_numbers=True,
                start_line=start_line,  # Keep the original start line
                highlight_lines=[highlight_line],  # Highlight the actual code line
            )
            panel_content.append(syntax)

        # Create a group with all content
        content_group = Group(*panel_content)

        # Create a single panel with all information
        _console.print(
            Panel(
                content_group,
                title="[bold red]Deprecation warning[/bold red]",
                border_style="red",
                expand=False,
            )
        )
    else:
        # Fallback for when rich is not available
        print(f"\nDeprecation warning: {message}", file=sys.stderr)
        print("-" * 40, file=sys.stderr)

        if frame_info.code_context:
            print(f"# Called from: {filename}:{lineno}", file=sys.stderr)
            line_start = lineno - (len(frame_info.code_context) // 2)
            for i, line in enumerate(frame_info.code_context):
                line_num = line_start + i
                marker = ">" if line_num == lineno else " "
                print(f"{marker} {line_num}: {line.rstrip()}", file=sys.stderr)

    print("", file=sys.stderr)


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
            # Get caller's frame info
            caller_frame = inspect.currentframe().f_back
            if caller_frame:
                _print_deprecation_warning(
                    "`jaxls.Factor` has been renamed to `jaxls.Cost`", caller_frame
                )

            # Also emit standard warning
            warnings.warn(
                "`jaxls.Factor` has been renamed to `jaxls.Cost`",
                DeprecationWarning,
                stacklevel=2,
            )
            return Cost

    # Create a descriptor for FactorGraph that shows a deprecation warning when accessed
    class _FactorGraphDescriptor:
        def __get__(self, obj: Any, objtype: Any = None) -> Any:
            # Get caller's frame info
            caller_frame = inspect.currentframe().f_back
            if caller_frame:
                _print_deprecation_warning(
                    "`jaxls.FactorGraph.make(...)` has been replaced with `jaxls.LeastSquaresProblem(...).analyze()`",
                    caller_frame,
                )

            # Also emit standard warning
            warnings.warn(
                "`jaxls.FactorGraph.make(...)` has been replaced with `jaxls.LeastSquaresProblem(...).analyze()`",
                DeprecationWarning,
                stacklevel=2,
            )

            class FactorGraph:
                @staticmethod
                def make(factors, variables, use_onp=False):
                    # Get caller's frame info
                    caller_frame = inspect.currentframe().f_back
                    if caller_frame:
                        _print_deprecation_warning(
                            "`jaxls.FactorGraph` has been renamed `jaxls.LeastSquaresProblem`",
                            caller_frame,
                        )

                    # Also emit standard warning
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
