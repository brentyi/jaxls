import abc
import contextlib
import dataclasses
import time
from typing import Any, Generator, Tuple, Type, TypeVar

import jax
import numpy as onp
import termcolor
from jax import numpy as jnp
from overrides import overrides

T = TypeVar("T")


def pytree_stack(*trees: T, axis=0) -> T:
    """Stack PyTrees along a specified axis."""
    return jax.tree_multimap(lambda *arrays: jnp.stack(arrays, axis=axis), *trees)


@contextlib.contextmanager
def stopwatch(label: str = "unlabeled block") -> Generator[None, None, None]:
    start_time = time.time()
    print(f"\n========")
    print(f"Running ({label})")
    yield
    print(f"{termcolor.colored(str(time.time() - start_time), attrs=['bold'])} seconds")
    print(f"========")


def register_dataclass_pytree(
    cls: Type[T], static_fields: Tuple[str, ...] = tuple()
) -> Type[T]:
    """Register a dataclass as a PyTree."""

    assert dataclasses.is_dataclass(cls)

    field: dataclasses.Field
    field_names = [field.name for field in dataclasses.fields(cls)]
    children_fields = [name for name in field_names if name not in static_fields]
    assert set(field_names) == set(children_fields) | set(static_fields)

    def _flatten(obj):
        return [getattr(obj, key) for key in children_fields], tuple(
            getattr(obj, key) for key in static_fields
        )

    def _unflatten(treedef, children):
        return cls(
            **dict(zip(children_fields, children)), **dict(zip(static_fields, treedef))
        )

    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)

    return cls


def hashable(cls: Type[T]) -> Type[T]:
    """Decorator for making classes hashable."""

    # Hash based on object ID, rather than contents
    cls.__hash__ = object.__hash__
    return cls


def get_epsilon(x: jnp.ndarray) -> float:
    if x.dtype is jnp.dtype("float32"):
        return 1e-5
    elif x.dtype is jnp.dtype("float64"):
        return 1e-10
    else:
        assert False, f"Unexpected array type: {x.dtype}"
