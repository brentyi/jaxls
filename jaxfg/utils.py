import contextlib
import time
from typing import Generator, Type, TypeVar

import jax
import jax_dataclasses
import termcolor
from jax import numpy as jnp

T = TypeVar("T")


def pytree_stack(*trees: T, axis=0) -> T:
    """Stack PyTrees along a specified axis."""
    return jax.tree_multimap(lambda *arrays: jnp.stack(arrays, axis=axis), *trees)


def pytree_concatenate(*trees: T, axis=0) -> T:
    """Concatenate PyTrees along a specified axis."""
    return jax.tree_multimap(lambda *arrays: jnp.concatenate(arrays, axis=axis), *trees)


@contextlib.contextmanager
def stopwatch(label: str = "unlabeled block") -> Generator[None, None, None]:
    """Context manager for measuring runtime."""
    start_time = time.time()
    print("\n========")
    print(f"Running ({label})")
    yield
    print(f"{termcolor.colored(str(time.time() - start_time), attrs=['bold'])} seconds")
    print("========")


def _identity(x: T) -> T:
    return x


def register_dataclass_pytree(cls: Type[T]) -> Type[T]:
    """Legacy API. Deprecation warning for now. TODO: remove this"""

    import warnings

    warnings.warn(
        "@register_dataclass_pytree has been phased out -- use"
        " @jax_dataclasses.pytree_dataclass() instead!",
        DeprecationWarning,
        stacklevel=1,
    )
    return jax_dataclasses._dataclasses._register(cls)
