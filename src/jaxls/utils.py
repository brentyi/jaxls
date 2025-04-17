import contextlib
import inspect
import time
from functools import partial
from typing import Generator

import jax
import termcolor
from loguru import logger


@contextlib.contextmanager
def stopwatch(label: str = "unlabeled block") -> Generator[None, None, None]:
    """Context manager for measuring runtime."""
    start_time = time.time()
    print("\n========")
    print(f"Running ({label})")
    yield
    print(f"{termcolor.colored(str(time.time() - start_time), attrs=['bold'])} seconds")
    print("========")


def _log(fmt: str, *args, **kwargs) -> None:
    logger.bind(function="log").info(fmt, *args, **kwargs)


def jax_log(fmt: str, *args, **kwargs) -> None:
    """Emit a loguru info message from a JITed JAX function."""
    jax.debug.callback(partial(_log, fmt), *args, **kwargs)


def print_deprecation_warning(message: str, stack_level: int = 2) -> None:
    """Print a nicely formatted deprecation warning with code context.

    Args:
        message: The deprecation message to display
        stack_level: Number of frames to go back to find the caller
                    (default: 2, meaning the caller of the caller of this function)
    """
    # Get the caller's frame based on stack_level.
    frame = inspect.currentframe()
    for _ in range(stack_level):
        if not frame:
            return
        frame = frame.f_back

    if not frame:
        return

    # Get more context (5 lines).
    frame_info = inspect.getframeinfo(frame, context=5)
    filename = frame_info.filename
    lineno = frame_info.lineno

    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text

    console = Console(stderr=True)

    # Create the content for the panel.
    warning_message = Text.from_markup(message, style="yellow")

    # Add a horizontal rule for separation.
    from rich.rule import Rule

    separator = Rule(style="dim")

    panel_content = [warning_message, separator]

    # Add code context if available.
    if frame_info.code_context:
        # Get the original code context.
        code_lines = frame_info.code_context
        start_line = lineno - (len(code_lines) // 2)

        # Insert a comment as the first line of the code context.
        modified_code_lines = list(code_lines)

        # Combine into a single string.
        code = "".join(modified_code_lines)

        # Calculate the line number that should be highlighted.
        highlight_line = start_line + (lineno - start_line)

        # Create syntax highlighting with the current line highlighted.
        syntax = Syntax(
            code,
            "python",
            line_numbers=True,
            start_line=start_line,  # Keep the original start line.
            highlight_lines={highlight_line},  # Highlight the actual code line.
        )
        panel_content.append(syntax)

    # Create a group with all content.
    content_group = Group(*panel_content)

    # Create a single panel with all information.
    console.print(
        Panel(
            content_group,
            title="[bold red]Deprecation Warning[/bold red]",
            border_style="red",
            expand=False,
        )
    )
