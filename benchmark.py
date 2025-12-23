#!/usr/bin/env python
"""Benchmark jaxls examples and report solver metrics."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tyro
from rich.console import Console
from rich.table import Table


@dataclass
class BenchmarkConfig:
    """Benchmark jaxls examples and report metrics."""

    examples_dir: Path = Path("docs/source/examples")
    """Directory containing example notebooks."""

    output_txt: Path = Path("benchmark_results.txt")
    """Output text file path for results table."""

    category: str | None = None
    """Filter by category (robotics, portfolio, mechanics, vision, guide)."""

    timeout: float = 120.0
    """Timeout in seconds per notebook."""

    verbose: bool = False
    """Show detailed output during execution."""


@dataclass
class SolveResult:
    """Result from the first solve() call in a notebook."""

    notebook: str
    iterations: int
    final_cost: float
    success: bool = True
    error: str | None = None


def discover_notebooks(examples_dir: Path, category: str | None = None) -> list[Path]:
    """Find all example notebooks, optionally filtered by category."""
    notebooks = []
    for nb in examples_dir.rglob("*.ipynb"):
        # Exclude checkpoints
        if ".ipynb_checkpoints" in str(nb):
            continue
        # Filter by category if specified
        if category is not None:
            rel_path = nb.relative_to(examples_dir)
            parts = rel_path.parts
            if len(parts) == 0 or parts[0] != category:
                continue
        notebooks.append(nb)
    return sorted(notebooks)


def run_notebook_with_instrumentation(
    notebook_path: Path, timeout: float, verbose: bool
) -> SolveResult | None:
    """Execute a notebook with solve() instrumentation and collect metrics."""
    import json
    import nbformat

    notebook_name = notebook_path.stem

    # Read the notebook
    with open(notebook_path, "r") as f:
        nb = nbformat.read(f, as_version=4)

    # Instrumentation code to inject at the start of the notebook
    instrumentation_code = '''
import jax
import jaxls
import jaxls._core

_benchmark_result = None  # Only capture first solve
_original_solve = jaxls._core.AnalyzedLeastSquaresProblem.solve

def _instrumented_solve(self, *args, **kwargs):
    global _benchmark_result

    # Check if we're inside a JAX transform (vmap, jit tracing, etc.)
    import jax.core
    def _has_tracer(x):
        if isinstance(x, jax.core.Tracer):
            return True
        if hasattr(x, '__dict__'):
            for v in x.__dict__.values():
                if _has_tracer(v):
                    return True
        return False

    inside_transform = _has_tracer(self) or _has_tracer(args) or _has_tracer(kwargs)

    if inside_transform:
        # Inside JAX transform - just call original without instrumentation
        return _original_solve(self, *args, **kwargs)

    # Already captured first solve - just run original
    if _benchmark_result is not None:
        return _original_solve(self, *args, **kwargs)

    # Check if user wanted summary
    original_return_summary = kwargs.get("return_summary", False)

    # Force return_summary for metrics
    kwargs = dict(kwargs)
    kwargs["return_summary"] = True

    try:
        result = _original_solve(self, *args, **kwargs)
        solution, summary = result

        # Block until computation is complete
        jax.block_until_ready(solution)
        jax.block_until_ready(summary)

        _benchmark_result = {
            "iterations": int(summary.iterations),
            "final_cost": float(summary.cost_history[summary.iterations]),
            "success": True,
            "error": None,
        }

        # Return based on original expectation
        if original_return_summary:
            return solution, summary
        return solution
    except Exception as e:
        _benchmark_result = {
            "iterations": 0,
            "final_cost": float("nan"),
            "success": False,
            "error": str(e),
        }
        raise

jaxls._core.AnalyzedLeastSquaresProblem.solve = _instrumented_solve
'''

    # Finalization code to print results
    finalization_code = '''
import json as _benchmark_json
print("__BENCHMARK_RESULT_START__")
print(_benchmark_json.dumps(_benchmark_result))
print("__BENCHMARK_RESULT_END__")
'''

    # Create new cells for instrumentation
    instrumentation_cell = nbformat.v4.new_code_cell(source=instrumentation_code)
    finalization_cell = nbformat.v4.new_code_cell(source=finalization_code)

    # Insert instrumentation at the beginning, finalization at the end
    nb.cells.insert(0, instrumentation_cell)
    nb.cells.append(finalization_cell)

    # Write modified notebook to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ipynb", delete=False, dir=notebook_path.parent
    ) as f:
        nbformat.write(nb, f)
        temp_notebook_path = f.name

    try:
        # Execute the notebook using jupyter execute
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "execute",
                temp_notebook_path,
                "--inplace",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(notebook_path.parent.parent.parent.parent),  # Project root
            env={
                **dict(__import__("os").environ),
                "PYTHONPATH": "src",
            },
        )

        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

        # Read executed notebook and extract results from output
        with open(temp_notebook_path, "r") as f:
            executed_nb = nbformat.read(f, as_version=4)

        # Find the finalization cell output
        for cell in executed_nb.cells:
            if cell.cell_type != "code":
                continue
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream" and output.get("name") == "stdout":
                    text = output.get("text", "")
                    if "__BENCHMARK_RESULT_START__" in text:
                        start_marker = "__BENCHMARK_RESULT_START__"
                        end_marker = "__BENCHMARK_RESULT_END__"
                        start_idx = text.index(start_marker) + len(start_marker)
                        end_idx = text.index(end_marker)
                        result_json = text[start_idx:end_idx].strip()

                        r = json.loads(result_json)
                        if r is not None:
                            return SolveResult(
                                notebook=notebook_name,
                                iterations=r["iterations"],
                                final_cost=r["final_cost"],
                                success=r["success"],
                                error=r.get("error"),
                            )

        return None

    except subprocess.TimeoutExpired:
        return SolveResult(
            notebook=notebook_name,
            iterations=0,
            final_cost=float("nan"),
            success=False,
            error=f"Timeout after {timeout}s",
        )
    except Exception as e:
        return SolveResult(
            notebook=notebook_name,
            iterations=0,
            final_cost=float("nan"),
            success=False,
            error=str(e),
        )
    finally:
        Path(temp_notebook_path).unlink(missing_ok=True)


def display_results(results: list[SolveResult], console: Console) -> None:
    """Display results in a rich table."""
    table = Table(title="jaxls Benchmark Results")
    table.add_column("Notebook", style="cyan")
    table.add_column("Iterations", justify="right")
    table.add_column("Final Cost", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        status = "[green]✓[/green]" if r.success else f"[red]✗[/red]"
        cost_str = f"{r.final_cost:.6g}" if r.success else "N/A"
        iter_str = str(r.iterations) if r.success else "N/A"

        table.add_row(
            r.notebook,
            iter_str,
            cost_str,
            status,
        )

    console.print(table)

    # Summary
    total = len(results)
    successful = sum(1 for r in results if r.success)
    console.print(f"\n[bold]Summary:[/bold] {successful}/{total} notebooks with solve() calls")


def export_txt(results: list[SolveResult], output_path: Path) -> None:
    """Export results table to text file."""
    table = Table(title="jaxls Benchmark Results")
    table.add_column("Notebook", style="cyan")
    table.add_column("Iterations", justify="right")
    table.add_column("Final Cost", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        status = "✓" if r.success else "✗"
        cost_str = f"{r.final_cost:.6g}" if r.success else "N/A"
        iter_str = str(r.iterations) if r.success else "N/A"

        table.add_row(r.notebook, iter_str, cost_str, status)

    # Write to file using a Console with no color/styling
    with open(output_path, "w") as f:
        file_console = Console(file=f, force_terminal=False, no_color=True, width=80)
        file_console.print(table)
        file_console.print(
            f"\nSummary: {sum(1 for r in results if r.success)}/{len(results)} "
            "notebooks with solve() calls"
        )


def main(config: BenchmarkConfig) -> None:
    """Run benchmarks."""
    console = Console()

    # Discover notebooks
    console.print(f"[bold]Discovering notebooks in {config.examples_dir}...[/bold]")
    notebooks = discover_notebooks(config.examples_dir, config.category)

    if len(notebooks) == 0:
        console.print("[yellow]No notebooks found![/yellow]")
        return

    console.print(f"Found {len(notebooks)} notebooks")
    if config.category is not None:
        console.print(f"  (filtered by category: {config.category})")

    # Run benchmarks
    all_results: list[SolveResult] = []

    for i, nb_path in enumerate(notebooks):
        rel_path = nb_path.relative_to(config.examples_dir)
        console.print(
            f"\n[{i + 1}/{len(notebooks)}] Running [cyan]{rel_path}[/cyan]..."
        )

        result = run_notebook_with_instrumentation(
            nb_path, config.timeout, config.verbose
        )

        if result is None:
            console.print("  [yellow]No solve() calls detected[/yellow]")
        elif result.success:
            console.print(
                f"  [green]✓[/green] {result.iterations} iters, "
                f"cost={result.final_cost:.6g}"
            )
            all_results.append(result)
        else:
            console.print(f"  [red]✗[/red] {result.error}")
            all_results.append(result)

    # Display summary
    console.print("\n")
    display_results(all_results, console)

    # Export results table to text file
    export_txt(all_results, config.output_txt)
    console.print(f"\n[bold]Results exported to {config.output_txt}[/bold]")


if __name__ == "__main__":
    config = tyro.cli(BenchmarkConfig)
    main(config)
