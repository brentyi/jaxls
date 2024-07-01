from __future__ import annotations

import abc
from typing import Any, Callable, ClassVar, Iterable, Self, override

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp


@jdc.pytree_dataclass
class Var[T]:
    """A symbolic representation of an optimization variable."""

    id: int | jax.Array

    # Class properties.
    # type ignores are for generics: https://github.com/python/typing/discussions/1424
    default: ClassVar[T]  # type: ignore
    """Default value for this variable."""
    parameter_dim: ClassVar[int]
    """Number of parameters in this variable type."""
    tangent_dim: ClassVar[int]
    """Dimension of the tangent space."""
    retract_fn: ClassVar[Callable[[T, jax.Array], T] | None]  # type: ignore
    """Retraction function for the manifold. None for Euclidean space."""

    def __init_subclass__(
        cls,
        default: T,
        tangent_dim: int | None,
        retract_fn: Callable[[T, jax.Array], T] | None,
    ) -> None:
        cls.default = default
        cls.parameter_dim = int(
            sum([onp.prod(leaf.size) for leaf in jax.tree.leaves(default)])
        )
        cls.tangent_dim = tangent_dim if tangent_dim is not None else cls.parameter_dim
        cls.retract_fn = retract_fn  # type: ignore
        super().__init_subclass__()

    @classmethod
    def allocate(cls, count: int, start_id: int = 0) -> tuple[Self, ...]:
        """Helper for allocating a sequence of variables."""
        return tuple(cls(i) for i in range(start_id, start_id + count))


@jdc.pytree_dataclass
class _VarOrderInfo:
    ids: jax.Array
    """IDs of variables, before sorting."""

    ids_sorted: jax.Array
    """Variable ID corresponding to element in some stacked array. Must be sorted.

    For example, if the ID array is:
        [17, 19, 27]

    And we call get_value(SomeVar(id=19)), we should return the variable value
    with index 2.
    """

    id_argsort: jax.Array
    """argsort() on the ID array. `ids_sorted = ids[id_argsort]`.

    Usage patterns:
        - ids[order_indices[var_type]] would sort the IDs of a specific
          variable type.
        - order_indices[var_type][0] would return the index to the variable
          with the smallest ID.
    """

    def index_from_id(self, id: int | jax.Array) -> jax.Array:
        return jnp.searchsorted(self.id_argsort, id)

    @staticmethod
    def make(vars: Iterable[Var]) -> dict[type[Var], _VarOrderInfo]:
        # Get list of variables for each type.
        vars_from_type = dict[type[Var], list[Var]]()
        for var in vars:
            if type(var) not in vars_from_type:
                vars_from_type[type(var)] = []
            vars_from_type[type(var)].append(var)

        # Get variable IDs.
        out = dict[type[Var], _VarOrderInfo]()

        for var_type, var_list in vars_from_type.items():
            ids = jnp.stack([var.id for var in var_list])
            order_indices = jnp.argsort(ids)

            out[var_type] = _VarOrderInfo(
                ids=ids[order_indices],
                ids_sorted=ids,
                id_argsort=order_indices,
            )
        return out


@jdc.pytree_dataclass
class VarValues:
    """A mapping from variables to variable values."""

    stacked_from_var_type: dict[type[Var], Any]
    """Stacked values for each variable type. Will be sorted by ID."""

    order_info: dict[type[Var], _VarOrderInfo]
    """Descriptions of how variable IDs are ordered."""

    def get_value[T](self, var: Var[T]) -> T:
        """Get the value of a specific variable."""
        assert getattr(var.id, "shape", None) == () or isinstance(var.id, int)
        var_type = type(var)
        index = self.order_info[var_type].index_from_id(var.id)
        return jax.tree.map(lambda x: x[index], self.stacked_from_var_type[var_type])

    def get_stacked_value[T](self, var_type: type[Var[T]]) -> T:
        """Get the value of all variables of a specific type."""
        return self.stacked_from_var_type[var_type]

    def __getitem__[T](self, var_or_type: Var[T] | type[Var[T]]) -> T:
        if isinstance(var_or_type, type):
            return self.get_stacked_value(var_or_type)
        else:
            assert isinstance(var_or_type, Var)
            return self.get_value(var_or_type)

    @staticmethod
    def from_dict(val_from_var: dict[Var, Any]) -> VarValues:
        """Create a `VarValues` object from a dictionary."""
        id_table = _VarOrderInfo.make(val_from_var.keys())

        # Get variable IDs.
        stacked_from_var_type = dict[type[Var], Any]()
        ids_from_var_type = dict[type[Var], jax.Array]()
        for var_type, var_list in vars_from_type.items():
            stacked_indices = jnp.stack([var.id for var in var_list])
            order_indices = jnp.argsort(stacked_indices)

            stacked_from_var_type[var_type] = jax.tree.map(
                lambda *leafs: jnp.stack(leafs, axis=0)[order_indices],
                [val_from_var[v] for v in var_list],
            )
            ids_from_var_type[var_type] = stacked_indices[order_indices]

        return VarValues(stacked_from_var_type, id_table)

    @staticmethod
    def from_defaults(vars: tuple[Var, ...]) -> VarValues:
        """Construct a `VarValues` object from default values for each variable type."""
        return VarValues.from_dict({v: v.default for v in vars})
