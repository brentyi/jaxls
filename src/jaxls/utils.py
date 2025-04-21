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


def print_deprecation_warning(
    message0: str, message1: str | None = None, stack_level: int = 2
) -> None:
    """Print a nicely formatted deprecation warning with code context.

    Args:
        message0: The deprecation message to display. Goes above code.
        message1: An optional second message to display. Goes under code.
        stack_level: Number of frames to go back to find the caller.
                    (default: 2, meaning the caller of the caller of this function)
    """
    # Get the caller's frame based on stack_level.
    frame = inspect.currentframe()
    for _ in range(stack_level):
        if not frame:
            return
        frame = frame.f_back

    if frame is None:
        return

    # Get more context (2 lines).
    frame_info = inspect.getframeinfo(frame, context=2)
    filename = frame_info.filename
    lineno = frame_info.lineno

    if frame_info.code_context is None:
        return

    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text

    console = Console(stderr=True)
    panel_content = [
        Text.from_markup(message0),
        Syntax("# " + filename + ":" + str(lineno), "python"),
    ]

    # Add code context if available.
    if frame_info.code_context:
        # Get the original code context.
        code_lines = frame_info.code_context
        start_line = lineno - (len(code_lines) // 2)

        while start_line < 1:
            start_line += 1
            code_lines.pop()

        code = "".join(code_lines)

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

    # Add the second message if provided.
    if message1 is not None:
        panel_content.append(Text.from_markup(message1))

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
