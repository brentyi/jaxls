from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering
from typing import Any, ClassVar

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import flatten_util
from jax import numpy as jnp


@total_ordering
class _HashableSortableMeta(type):
    def __hash__(cls):
        return object.__hash__(cls)

    def __lt__(cls, other):
        if cls.__name__ == other.__name__:
            return id(cls) < id(other)
        else:
            return cls.__name__ < other.__name__


@dataclass(frozen=True)
class VarTypeOrdering:
    order_from_type: Any

    def ordered_dict_items(
        self,
        var_type_mapping: Any,
    ) -> Any:
        return sorted(
            var_type_mapping.items(), key=lambda x: self.order_from_type[x[0]]
        )


@jdc.pytree_dataclass
class VarWithValue:
    @classmethod
    def __class_getitem__(cls, params):
        return cls

    variable: Any
    value: Any


@jdc.pytree_dataclass
class Var(metaclass=_HashableSortableMeta):
    @classmethod
    def __class_getitem__(cls, params):
        return cls

    id: Any

    _default_factory: ClassVar[Any]
    tangent_dim: ClassVar[Any]
    retract_fn: ClassVar[Any]

    @classmethod
    def default_factory(cls) -> Any:
        return cls._default_factory()

    def with_value(self, value: Any) -> Any:
        return VarWithValue(self, value)

    def __getitem__(self, index_or_slice: Any) -> Any:
        assert not isinstance(self.id, int)
        return self.__class__(self.id[index_or_slice])

    def __init_subclass__(
        cls,
        *,
        default_factory: Any = None,
        default: Any = None,
        retract_fn: Any = None,
        tangent_dim: Any = None,
    ) -> Any:
        if default_factory is None:
            assert default is not None
            import warnings

            warnings.warn(
                "Defining 'default' for variables is deprecated. Use 'default_factory' instead.",
                stacklevel=2,
            )
            default_factory = lambda: default

        cls._default_factory = staticmethod(default_factory)
        if retract_fn is not None:
            assert tangent_dim is not None
            cls.tangent_dim = tangent_dim
            cls.retract_fn = retract_fn
        else:
            assert tangent_dim is None
            parameter_dim = int(
                sum(
                    [
                        onp.prod(leaf.shape)
                        for leaf in jax.tree.leaves(jax.eval_shape(default_factory))
                    ]
                )
            )
            cls.tangent_dim = parameter_dim
            cls.retract_fn = cls._euclidean_retract

        super().__init_subclass__()

        jdc.pytree_dataclass(cls)

    @staticmethod
    def _euclidean_retract(pytree: Any, delta: Any) -> Any:
        flat, unravel = flatten_util.ravel_pytree(pytree)
        del flat
        return jax.tree.map(jnp.add, pytree, unravel(delta))


@jdc.pytree_dataclass
class VarValues:
    vals_from_type: Any

    ids_from_type: Any

    def get_value(self, var: Any) -> Any:
        if not isinstance(var.id, int) and var.id.ndim > 0:
            return jax.vmap(self.get_value)(var)

        assert getattr(var.id, "shape", None) == () or isinstance(var.id, int)
        var_type = type(var)
        index = jnp.searchsorted(self.ids_from_type[var_type], var.id)
        return jax.tree.map(lambda x: x[index], self.vals_from_type[var_type])

    def get_stacked_value(self, var_type: Any) -> Any:
        return self.vals_from_type[var_type]

    def __getitem__(self, var_or_type: Any) -> Any:
        if isinstance(var_or_type, type):
            return self.get_stacked_value(var_or_type)
        else:
            assert isinstance(var_or_type, Var)
            return self.get_value(var_or_type)

    def __repr__(self) -> Any:
        out_lines = list()

        for var_type, ids in self.ids_from_type.items():
            for i in range(ids.shape[-1]):
                batch_axes = ids.shape[:-1]
                val = jax.tree.map(
                    lambda x: x.take(indices=i, axis=len(batch_axes)),
                    self.vals_from_type[var_type],
                )
                out_lines.append(
                    f"  {var_type.__name__}(" + f"{ids[..., i]}): ".ljust(8) + f"{val},"
                )

        props = "\n".join(out_lines)
        return f"VarValues(\n{props}\n)"

    @staticmethod
    def make(variables: Any) -> Any:
        vars = list()
        vals = list()

        cached_default_from_type = dict()

        for v in variables:
            if isinstance(v, Var):
                if type(v) not in cached_default_from_type:
                    cached_default_from_type[type(v)] = v.default_factory()

                ids = v.id
                assert isinstance(ids, int) or len(ids.shape) in (0, 1)
                vars.append(v)
                vals.append(
                    jax.tree.map(
                        (lambda x: x)
                        if isinstance(ids, int) or len(ids.shape) == 0
                        else (
                            lambda x: jnp.broadcast_to(
                                x[None, ...],
                                (len(ids.shape), *x.shape),
                            )
                        ),
                        cached_default_from_type[type(v)],
                    )
                )
            else:
                vars.append(v.variable)
                vals.append(v.value)

        ids_from_type, vals_from_type = sort_and_stack_vars(tuple(vars), tuple(vals))
        return VarValues(vals_from_type=vals_from_type, ids_from_type=ids_from_type)

    def _get_subset(
        self,
        indices_from_type: Any,
        ordering: Any,
    ) -> Any:
        vals_from_type = dict()
        ids_from_type = dict()
        for var_type, indices in ordering.ordered_dict_items(indices_from_type):
            vals_from_type[var_type] = jax.tree.map(
                lambda x: x[indices], self.vals_from_type[var_type]
            )
            ids_from_type[var_type] = self.ids_from_type[var_type][indices]
        return VarValues(vals_from_type=vals_from_type, ids_from_type=ids_from_type)

    def _get_tangent_dim(self) -> Any:
        total = 0
        for var_type, ids in self.ids_from_type.items():
            total += ids.shape[-1] * var_type.tangent_dim
        return total

    def _get_batch_axes(self) -> Any:
        return next(iter(self.ids_from_type.values())).shape[:-1]

    def _retract(self, tangent: Any, ordering: Any) -> Any:
        vals_from_type = dict()
        tangent_slice_start = 0
        for var_type, ids in ordering.ordered_dict_items(self.ids_from_type):
            assert len(ids.shape) == 1

            tangent_slice_dim = var_type.tangent_dim * ids.shape[0]
            tangent_slice = tangent[
                tangent_slice_start : tangent_slice_start + tangent_slice_dim
            ]
            tangent_slice_start += tangent_slice_dim

            vals_from_type[var_type] = jax.vmap(var_type.retract_fn)(
                self.vals_from_type[var_type],
                tangent_slice.reshape((ids.shape[0], var_type.tangent_dim)),
            )

        return VarValues(
            vals_from_type=vals_from_type, ids_from_type=self.ids_from_type
        )


def sort_and_stack_vars(variables: Any, values: Any = None) -> Any:
    vars_from_type = dict()
    vals_from_type = dict() if values is not None else None
    for i in range(len(variables)):
        var = variables[i]
        val = values[i] if values is not None else None

        if isinstance(var.id, int) or len(var.id.shape) == 0:
            var = jax.tree.map(lambda leaf: jnp.array(leaf)[None], var)
            if val is not None:
                val = jax.tree.map(lambda leaf: jnp.array(leaf)[None], val)
        else:
            assert len(var.id.shape) == 1, "Variable IDs must be 0D or 1D."

        var_type = type(var)
        vars_from_type.setdefault(var_type, [])
        if vals_from_type is not None:
            vals_from_type.setdefault(var_type, [])

        vars_from_type[var_type].append(var)
        if vals_from_type is not None:
            vals_from_type[var_type].append(val)

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
            var_type: jax.tree.map(lambda *leafs: jnp.concatenate(leafs, axis=0), *vals)
            for var_type, vals in vals_from_type.items()
        }
        sorted_vals_from_type = {
            var_type: jax.tree.map(
                lambda x: x[ids_argsort_from_type[var_type]], stacked_val
            )
            for var_type, stacked_val in stacked_vals_from_type.items()
        }
        return ids_sorted_from_type, sorted_vals_from_type
