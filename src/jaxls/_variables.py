from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Iterable, Literal, cast, overload

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import flatten_util
from jax import numpy as jnp


@dataclass(frozen=True)
class VarTypeOrdering:
    """Object describing how variables are ordered within a `VarValues` object
    or tangent vector.

    We should use this instead of iterating directly over `dict[type[Var[Any]], T]`
    objects. It ensures correct tangent vector computation and also prevents
    dictionary ordering edge cases.

    Relevant: https://github.com/google/jax/issues/4085
    """

    order_from_type: dict[type[Var[Any]], int]

    def ordered_dict_items[T](
        self,
        var_type_mapping: dict[type[Var[Any]], T],
    ) -> list[tuple[type[Var[Any]], T]]:
        return sorted(
            var_type_mapping.items(), key=lambda x: self.order_from_type[x[0]]
        )


@jdc.pytree_dataclass
class Var[T]:
    """A symbolic representation of an optimization variable."""

    id: int | jax.Array

    # We would ideally annotate these as `ClassVar[T]`, but we can't.
    #
    # https://github.com/python/typing/discussions/1424
    default: ClassVar[Any]
    """Default value for this variable."""
    tangent_dim: ClassVar[int]
    """Dimension of the tangent space."""
    retract_fn: ClassVar[Callable[[Any, jax.Array], Any]]
    """Retraction function for the manifold. None for Euclidean space."""

    @overload
    def __init_subclass__[T_](
        cls,
        default: T_,
        retract_fn: Callable[[T_, jax.Array], T_],
        tangent_dim: int,
    ) -> None: ...

    @overload
    def __init_subclass__(
        cls,
        default: Any,
        retract_fn: None = None,
        tangent_dim: None = None,
    ) -> None: ...

    def __init_subclass__[T_](
        cls,
        default: T_,
        retract_fn: Callable[[T_, jax.Array], T_] | None = None,
        tangent_dim: int | None = None,
    ) -> None:
        cls.default = default
        if retract_fn is not None:
            assert tangent_dim is not None
            cls.tangent_dim = tangent_dim
            cls.retract_fn = retract_fn
        else:
            assert tangent_dim is None
            parameter_dim = int(
                sum([onp.prod(leaf.size) for leaf in jax.tree.leaves(default)])
            )
            cls.tangent_dim = parameter_dim
            cls.retract_fn = cls._euclidean_retract

        super().__init_subclass__()

        # Subclasses need to be registered as PyTrees.
        jdc.pytree_dataclass(cls)

    @staticmethod
    def _euclidean_retract(pytree: T, delta: jax.Array) -> T:
        # Euclidean retraction.
        flat, unravel = flatten_util.ravel_pytree(pytree)
        del flat
        return cast(T, jax.tree_map(jnp.add, pytree, unravel(delta)))


@jdc.pytree_dataclass
class VarValues:
    """A mapping from variables to variable values.

    Given a variable object `var` and a values of object `vals`, we can get the
    value by calling one of:

        # Equivalent.
        vals.get_value(var)
        vals[var]

    To get all values of a specific type `var_type`, use:

        # Equivalent.
        vals.get_stacked_value(var_type)
        vals[var_type]

    """

    vals_from_type: dict[type[Var[Any]], Any]
    """Stacked values for each variable type. Will be sorted by ID (ascending)."""

    ids_from_type: dict[type[Var[Any]], jax.Array]
    """Variable ID for each value, sorted in ascending order."""

    def get_value[T](self, var: Var[T]) -> T:
        """Get the value of a specific variable."""
        assert getattr(var.id, "shape", None) == () or isinstance(var.id, int)
        var_type = type(var)
        index = jnp.searchsorted(self.ids_from_type[var_type], var.id)
        return jax.tree.map(lambda x: x[index], self.vals_from_type[var_type])

    def get_stacked_value[T](self, var_type: type[Var[T]]) -> T:
        """Get the value of all variables of a specific type."""
        return self.vals_from_type[var_type]

    def __getitem__[T](self, var_or_type: Var[T] | type[Var[T]]) -> T:
        if isinstance(var_or_type, type):
            return self.get_stacked_value(var_or_type)
        else:
            assert isinstance(var_or_type, Var)
            return self.get_value(var_or_type)

    def __repr__(self) -> str:
        out_lines = list[str]()

        for var_type, ids in self.ids_from_type.items():
            for i in range(ids.shape[-1]):
                batch_axes = ids.shape[:-1]
                val = jax.tree_map(
                    lambda x: x.take(indices=i, axis=len(batch_axes)),
                    self.vals_from_type[var_type],
                )
                out_lines.append(
                    f"  {var_type.__name__}(" + f"{ids[..., i]}): ".ljust(8) + f"{val},"
                )

        return f"VarValues(\n{'\n'.join(out_lines)}\n)"

    @staticmethod
    def make[T](
        variables: Iterable[Var[T]],
        values: Iterable[T] | Literal["default"] = "default",
    ) -> VarValues:
        """Create a `VarValues` object. Entries in `vars` and entries in
        `values` have a 1:1 correspondence.

        We don't use a {var: value} dictionary because variables are not
        hashable.
        """
        variables = tuple(variables)
        if values == "default":
            ids_from_type = sort_and_stack_vars(variables)
            vals_from_type = {
                # This should be faster than jnp.stack().
                var_type: jax.tree_map(
                    lambda x: jnp.broadcast_to(x[None, ...], (len(ids), *x.shape)),
                    var_type.default,
                )
                for var_type, ids in ids_from_type.items()
            }
            return VarValues(vals_from_type=vals_from_type, ids_from_type=ids_from_type)
        else:
            values = tuple(values)
            assert len(variables) == len(values)
            ids_from_type, vals_from_type = sort_and_stack_vars(variables, values)
            return VarValues(vals_from_type=vals_from_type, ids_from_type=ids_from_type)

    def _get_subset(
        self,
        indices_from_type: dict[type[Var[Any]], jax.Array],
        ordering: VarTypeOrdering,
    ) -> VarValues:
        """Get a new VarValues of object with only a subset of the variables.
        Assumes that the input IDs are all sorted."""
        vals_from_type = dict[type[Var[Any]], Any]()
        ids_from_type = dict[type[Var[Any]], jax.Array]()
        for var_type, indices in ordering.ordered_dict_items(indices_from_type):
            vals_from_type[var_type] = jax.tree_map(
                lambda x: x[indices], self.vals_from_type[var_type]
            )
            ids_from_type[var_type] = self.ids_from_type[var_type][indices]
        return VarValues(vals_from_type=vals_from_type, ids_from_type=ids_from_type)

    def _get_tangent_dim(self) -> int:
        """Sum of tangent dimensions of all variables in this structure.

        Batch axes: assumed to be leading, not accounted for in the tangent
        dimension computation."""
        total = 0
        for var_type, ids in self.ids_from_type.items():  # Order doesn't matter here.
            total += ids.shape[-1] * var_type.tangent_dim
        return total

    def _get_batch_axes(self) -> tuple[int, ...]:
        return next(iter(self.ids_from_type.values())).shape[:-1]

    def _retract(self, tangent: jax.Array, ordering: VarTypeOrdering) -> VarValues:
        vals_from_type = dict[type[Var[Any]], Any]()
        tangent_slice_start = 0
        for var_type, ids in (
            # Respect variable ordering in tangent layout.
            ordering.ordered_dict_items(self.ids_from_type)
        ):
            assert len(ids.shape) == 1

            # Get slice of the tangent vector that will be used for this
            # variable type.
            tangent_slice_dim = var_type.tangent_dim * ids.shape[0]
            tangent_slice = tangent[
                tangent_slice_start : tangent_slice_start + tangent_slice_dim
            ]
            tangent_slice_start += tangent_slice_dim

            # Apply retraction for this variable type.
            vals_from_type[var_type] = jax.vmap(var_type.retract_fn)(
                self.vals_from_type[var_type],
                tangent_slice.reshape((ids.shape[0], var_type.tangent_dim)),
            )

        return VarValues(
            vals_from_type=vals_from_type, ids_from_type=self.ids_from_type
        )


@overload
def sort_and_stack_vars(
    variables: tuple[Var, ...], values: None = None
) -> dict[type[Var[Any]], jax.Array]: ...


@overload
def sort_and_stack_vars(
    variables: tuple[Var, ...], values: tuple[Any, ...]
) -> tuple[dict[type[Var[Any]], jax.Array], dict[type[Var[Any]], Any]]: ...


def sort_and_stack_vars(
    variables: tuple[Var, ...], values: tuple[Any, ...] | None = None
) -> (
    dict[type[Var[Any]], jax.Array]
    | tuple[dict[type[Var[Any]], jax.Array], dict[type[Var[Any]], Any]]
):
    """Sort variables by ID, ascending. If `values` is specified, returns a
    (sorted ID mapping, value mapping) tuple. Otherwise, only returns the ID
    mapping."""
    # First, organize variables and values by type.
    vars_from_type = dict[type[Var[Any]], list[Var]]()
    vals_from_type = dict[type[Var[Any]], list[Any]]() if values is not None else None
    for i in range(len(variables)):
        var = variables[i]
        val = values[i] if values is not None else None

        # Variables should either be single or stacked.
        if isinstance(var.id, int) or len(var.id.shape) == 0:
            var = jax.tree.map(lambda leaf: jnp.array(leaf)[None], var)
            if val is not None:
                val = jax.tree.map(lambda leaf: jnp.array(leaf)[None], val)
        else:
            # We could easily support more, but this feels like an unlikely use case.
            assert len(var.id.shape) == 1, "Variable IDs must be 0D or 1D."

        # Put variables and values into dictionary.
        var_type = type(var)
        vars_from_type.setdefault(var_type, [])
        if vals_from_type is not None:
            vals_from_type.setdefault(var_type, [])

        vars_from_type[var_type].append(var)
        if vals_from_type is not None:
            vals_from_type[var_type].append(val)

    # Concatenate variable IDs and values along axis=0.
    # We then re-order variables by ascending ID.
    stacked_var_from_type = {
        var_type: jax.tree.map(lambda *leafs: jnp.concatenate(leafs, axis=0), *vars)
        for var_type, vars in vars_from_type.items()
    }
    ids_argsort_from_type = {
        var_type: jnp.argsort(stacked_var.id)
        for var_type, stacked_var in stacked_var_from_type.items()
    }
    ids_sorted_from_type = {
        var_type: stacked_var.id[ids_argsort_from_type[var_type]]
        for var_type, stacked_var in stacked_var_from_type.items()
    }

    if vals_from_type is None:
        return ids_sorted_from_type
    else:
        stacked_vals_from_type = {
            var_type: jax.tree_map(lambda *leafs: jnp.concatenate(leafs, axis=0), *vals)
            for var_type, vals in vals_from_type.items()
        }
        sorted_vals_from_type = {
            var_type: jax.tree.map(
                lambda x: x[ids_argsort_from_type[var_type]], stacked_val
            )
            for var_type, stacked_val in stacked_vals_from_type.items()
        }
        return ids_sorted_from_type, sorted_vals_from_type
