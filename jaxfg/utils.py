import contextlib
import time
import warnings
from typing import Generator, TypeVar

import jax
import termcolor
from jax import numpy as jnp

T = TypeVar("T")


def pytree_stack(*trees: T, axis=0) -> T:
    """Stack pytrees along a specified axis."""
    warnings.warn("Deprecated!", DeprecationWarning, stacklevel=1)
    return jax.tree_map(lambda *arrays: jnp.stack(arrays, axis=axis), *trees)


def pytree_concatenate(*trees: T, axis=0) -> T:
    """Concatenate pytrees along a specified axis."""
    warnings.warn("Deprecated!", DeprecationWarning, stacklevel=1)
    return jax.tree_map(lambda *arrays: jnp.concatenate(arrays, axis=axis), *trees)


@contextlib.contextmanager
def stopwatch(label: str = "unlabeled block") -> Generator[None, None, None]:
    """Context manager for measuring runtime."""
    start_time = time.time()
    print("\n========")
    print(f"Running ({label})")
    yield
    print(f"{termcolor.colored(str(time.time() - start_time), attrs=['bold'])} seconds")
    print("========")
