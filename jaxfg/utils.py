import contextlib
import dataclasses
import time
from typing import Dict, Generator, Tuple, Type, TypeVar

import jax
import termcolor
from flax import serialization
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


def register_dataclass_pytree(
    cls: Type[T], static_fields: Tuple[str, ...] = tuple()
) -> Type[T]:
    """Register a dataclass as a flax-serializable PyTree.

    For compatibility with function transformations in JAX (jit, grad, vmap, etc),
    arguments and return values must all be
    [PyTree](https://jax.readthedocs.io/en/latest/pytrees.html) containers; this
    decorator enables dataclasses to be used as valid PyTree nodes.

    Very similar to `flax.struct.dataclass`, but (a) adds support for static fields and
    (b) works better with non-Google tooling (mypy, jedi, etc).

    Args:
        cls (Type[T]): Dataclass to wrap.
        static_fields (Tuple[str, ...]): Any static field names as strings. Rather than
            including these fields as "children" of our dataclass, their values must be
            hashable and are considered part of the treedef.  Pass in using
            `@jax.partial()`.

    Returns:
        Type[T]: Decorated class.
    """

    assert dataclasses.is_dataclass(cls)

    # Get a list of fields in our dataclass
    field: dataclasses.Field
    field_names = [field.name for field in dataclasses.fields(cls)]
    children_fields = [name for name in field_names if name not in static_fields]
    assert set(field_names) == set(children_fields) | set(static_fields)

    # Define flatten, unflatten operations: this simple converts our dataclass to a list
    # of fields.
    def _flatten(obj):
        return [getattr(obj, key) for key in children_fields], tuple(
            getattr(obj, key) for key in static_fields
        )

    def _unflatten(treedef, children):
        return cls(
            **dict(zip(children_fields, children)), **dict(zip(static_fields, treedef))
        )

    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)

    # Serialization: this is mostly copied from `flax.struct.dataclass`
    def _to_state_dict(x: T):
        state_dict = {
            name: serialization.to_state_dict(getattr(x, name)) for name in field_names
        }
        return state_dict

    def _from_state_dict(x: T, state: Dict):
        """Restore the state of a data class."""
        state = state.copy()  # copy the state so we can pop the restored fields.
        updates = {}
        for name in field_names:
            if name not in state:
                raise ValueError(
                    f"Missing field {name} in state dict while restoring"
                    f" an instance of {cls.__name__}"
                )
            value = getattr(x, name)
            value_state = state.pop(name)
            updates[name] = serialization.from_state_dict(value, value_state)
        if state:
            names = ",".join(state.keys())
            raise ValueError(
                f'Unknown field(s) "{names}" in state dict while'
                f" restoring an instance of {cls.__name__}"
            )
        return dataclasses.replace(x, **updates)

    serialization.register_serialization_state(cls, _to_state_dict, _from_state_dict)

    return cls
