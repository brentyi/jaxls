from __future__ import annotations

from typing import Any

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from jax.tree_util import default_registry
from typing_extensions import deprecated


@jdc.pytree_dataclass
class Cost:
    @classmethod
    def __class_getitem__(cls, params):
        return cls

    compute_residual: jdc.Static[Any]

    args: Any

    kind: jdc.Static[Any] = "l2_squared"

    jac_mode: jdc.Static[Any] = "auto"

    jac_batch_size: jdc.Static[Any] = None

    jac_custom_fn: jdc.Static[Any] = None

    jac_custom_with_cache_fn: jdc.Static[Any] = None

    name: jdc.Static[Any] = None

    def _get_name(self) -> Any:
        if self.name is None:
            return self.compute_residual.__name__
        return self.name

    def _get_variables(self) -> Any:
        from ._variables import Var

        def get_variables_recursive(current: Any) -> Any:
            children_and_meta = default_registry.flatten_one_level(current)
            if children_and_meta is None:
                return []

            variables = []
            for child in children_and_meta[0]:
                if isinstance(child, Var):
                    variables.append(child)
                else:
                    variables.extend(get_variables_recursive(child))
            return variables

        return tuple(get_variables_recursive(self.args))

    def _get_batch_axes(self) -> Any:
        variables = self._get_variables()
        assert len(variables) != 0, f"No variables found in {type(self).__name__}!"
        return jnp.broadcast_shapes(
            *[() if isinstance(v.id, int) else v.id.shape for v in variables]
        )

    def _broadcast_batch_axes(self) -> Any:
        batch_axes = self._get_batch_axes()
        if batch_axes is None:
            return self
        leaves, treedef = jax.tree.flatten(self)
        broadcasted_leaves = []
        for leaf in leaves:
            if isinstance(leaf, (int, float)):
                leaf = jnp.array(leaf)
            try:
                broadcasted_leaf = jnp.broadcast_to(
                    leaf, batch_axes + leaf.shape[len(batch_axes) :]
                )
            except ValueError as e:
                error_msg = (
                    f"{str(e)}\n"
                    f"{type(self).__name__} name: '{self._get_name()}'\n"
                    f"Detected batch axes: {batch_axes}\n"
                    f"Flattened argument shapes: {[getattr(x, 'shape', ()) for x in leaves]}\n"
                    f"All shapes should either have the same batch axis or have dimension (1,) for broadcasting."
                )
                raise ValueError(error_msg) from e
            broadcasted_leaves.append(broadcasted_leaf)
        return jax.tree.unflatten(treedef, broadcasted_leaves)

    @staticmethod
    def factory(
        compute_residual: Any = None,
        *,
        kind: Any = "l2_squared",
        jac_mode: Any = "auto",
        jac_batch_size: Any = None,
        jac_custom_fn: Any = None,
        jac_custom_with_cache_fn: Any = None,
        name: Any = None,
    ) -> Any:
        def decorator(
            compute_residual: Any,
        ) -> Any:
            def inner(*args: Any, **kwargs: Any) -> Any:
                return Cost(
                    compute_residual=lambda values, args, kwargs: compute_residual(
                        values, *args, **kwargs
                    ),
                    args=(args, kwargs),
                    kind=kind,
                    jac_mode=jac_mode,
                    jac_batch_size=jac_batch_size,
                    jac_custom_fn=(
                        lambda values, args, kwargs: jac_custom_fn(
                            values, *args, **kwargs
                        )
                    )
                    if jac_custom_fn is not None
                    else None,
                    jac_custom_with_cache_fn=(
                        lambda values, cache, args, kwargs: jac_custom_with_cache_fn(
                            values, cache, *args, **kwargs
                        )
                    )
                    if jac_custom_with_cache_fn is not None
                    else None,
                    name=name if name is not None else compute_residual.__name__,
                )

            return inner

        if compute_residual is None:
            return decorator
        return decorator(compute_residual)

    @staticmethod
    @deprecated("Use Cost.factory instead of Cost.create_factory")
    def create_factory(
        compute_residual: Any = None,
        *,
        kind: Any = "l2_squared",
        jac_mode: Any = "auto",
        jac_batch_size: Any = None,
        jac_custom_fn: Any = None,
        jac_custom_with_cache_fn: Any = None,
        name: Any = None,
    ) -> Any:
        import warnings

        warnings.warn(
            "Cost.create_factory is deprecated, use Cost.factory instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return Cost.factory(
            compute_residual,
            kind=kind,
            jac_mode=jac_mode,
            jac_batch_size=jac_batch_size,
            jac_custom_fn=jac_custom_fn,
            jac_custom_with_cache_fn=jac_custom_with_cache_fn,
            name=name,
        )

    if True:

        @staticmethod
        def make(
            compute_residual: jdc.Static[Any],
            args: Any,
            jac_mode: jdc.Static[Any] = "auto",
            jac_custom_fn: jdc.Static[Any] = None,
        ) -> Any:
            import warnings

            warnings.warn(
                "Use Cost() directly instead of Cost.make()", DeprecationWarning
            )
            return Cost(
                compute_residual=compute_residual,
                args=args,
                jac_mode=jac_mode,
                jac_batch_size=None,
                jac_custom_fn=jac_custom_fn,
            )
