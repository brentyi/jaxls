import contextlib
import dataclasses
import time
from typing import Any, Callable, Dict, Generator, List, Type, TypeVar

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


def _identity(x: T) -> T:
    return x


def static_field(
    treedef_from_value: Callable[[T], Any] = _identity,
    value_from_treedef: Callable[[Any], T] = _identity,
) -> Dict[str, Any]:
    """Returns metadata dictionary for `dataclasses.field(metadata=...)`, which
    marks that a field should be treated as static and part of the treedef.

    This is often useful for fields that contain anything that's not a standard
    container of arrays, such as boolean flags used in non-JITable control flow.

    Optionally, we can specify a mapping from the value held by the field to its
    representation in the treedef. This can be helpful for unhashable types, or in cases
    where the specific identity of the field value doesn't matter. (for example: store
    only the type in the flattened treedef, and reconstruct the object when
    unflattening)
    """
    return {
        "pytree_static_value": True,
        "treedef_from_value": treedef_from_value,
        "value_from_treedef": value_from_treedef,
    }


def register_dataclass_pytree(cls: Type[T]) -> Type[T]:
    """Register a dataclass as a flax-serializable PyTree.

    For compatibility with function transformations in JAX (jit, grad, vmap, etc),
    arguments and return values must all be
    [PyTree](https://jax.readthedocs.io/en/latest/pytrees.html) containers; this
    decorator enables dataclasses to be used as valid PyTree nodes.

    Very similar to `flax.struct.dataclass`, but (a) adds support for static fields and
    (b) expects an external, explicit for @dataclass decorator for better static
    analysis support. The latter may change if
    [dataclass_transform](https://github.com/microsoft/pyright/blob/main/specs/dataclass_transforms.md)
    gains traction.

    We assume all registered classes retain the default dataclass constructor.

    Args:
        cls (Type[T]): Dataclass to wrap.
        make_immutable (bool): Set to `True` to make dataclass immutable.
    """

    assert dataclasses.is_dataclass(cls)

    # Determine which fields are static and part of the treedef, and which should be
    # registered as child nodes
    child_node_field_names: List[str] = []
    static_fields: List[dataclasses.Field] = []
    for field in dataclasses.fields(cls):
        if "pytree_static_value" in field.metadata:
            static_fields.append(field)
        else:
            child_node_field_names.append(field.name)

    # Define flatten, unflatten operations: this simple converts our dataclass to a list
    # of fields.
    def _flatten(obj):
        children = tuple(getattr(obj, key) for key in child_node_field_names)
        treedef = tuple(
            field.metadata["treedef_from_value"](getattr(obj, field.name))
            for field in static_fields
        )
        return children, treedef

    def _unflatten(treedef, children):
        return cls(
            **dict(zip(child_node_field_names, children)),
            **{
                field.name: field.metadata["value_from_treedef"](tdef)
                for field, tdef in zip(static_fields, treedef)
            },
        )

        # Alternative:
        #     return dataclasses.replace(
        #         cls.__new__(cls),
        #         **dict(zip(children_fields, children)),
        #         **dict(zip(static_fields_set, treedef)),
        #     )

    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)

    # Serialization: this is mostly copied from `flax.struct.dataclass`
    def _to_state_dict(x: T):
        state_dict = {
            name: serialization.to_state_dict(getattr(x, name))
            for name in child_node_field_names
        }
        return state_dict

    def _from_state_dict(x: T, state: Dict):
        state = state.copy()  # copy the state so we can pop the restored fields.
        updates = {}
        for name in child_node_field_names:
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

    # Make dataclass immutable after __init__ is called
    # Similar to dataclasses.dataclass(frozen=True), but a bit friendlier for custom
    # __init__ methods
    def _mark_immutable():
        original_init = cls.__init__ if hasattr(cls, "__init__") else None

        def disabled_setattr(*args, **kwargs):
            raise dataclasses.FrozenInstanceError(
                "Dataclass registered as PyTree is immutable!"
            )

        def new_init(self, *args, **kwargs):
            cls.__setattr__ = object.__setattr__
            if original_init is not None:
                original_init(self, *args, **kwargs)
            cls.__setattr__ = disabled_setattr

        cls.__setattr__ = disabled_setattr  # type: ignore
        cls.__init__ = new_init  # type: ignore

    _mark_immutable()

    return cls
