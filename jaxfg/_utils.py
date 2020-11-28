import abc
import contextlib
import dataclasses
import time
from typing import Generator, Tuple

import jax
import numpy as onp
import termcolor
from jax import numpy as jnp
from overrides import overrides


@contextlib.contextmanager
def stopwatch(label: str = "unlabeled block") -> Generator[None, None, None]:
    start_time = time.time()
    print(f"\n========")
    print(f"Running ({label})")
    yield
    print(f"{termcolor.colored(str(time.time() - start_time), attrs=['bold'])} seconds")
    print(f"========")


def immutable_dataclass(cls):
    """Decorator for defining immutable dataclasses."""

    # Hash based on object ID, rather than contents
    cls.__hash__ = object.__hash__

    return dataclasses.dataclass(cls, frozen=True)


def get_epsilon(x: jnp.ndarray) -> float:
    if x.dtype is jnp.dtype("float32"):
        return 1e-5
    elif x.dtype is jnp.dtype("float64"):
        return 1e-10
    else:
        assert False, f"Unexpected array type: {x.dtype}"
