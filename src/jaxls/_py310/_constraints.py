from __future__ import annotations

from typing import Any

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp
from jax.tree_util import default_registry

from ._variables import Var, VarValues, sort_and_stack_vars


@jdc.pytree_dataclass
class AugmentedLagrangianParams:
    @classmethod
    def __class_getitem__(cls, params):
        return cls

    lagrange_multipliers: Any

    penalty_params: Any

    original_args: Any

    constraint_index: jdc.Static[Any]


@jdc.pytree_dataclass
class Constraint:
    @classmethod
    def __class_getitem__(cls, params):
        return cls

    compute_residual: jdc.Static[Any]

    args: Any

    constraint_type: jdc.Static[Any] = "eq_zero"

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
        def get_variables(current: Any) -> Any:
            children_and_meta = default_registry.flatten_one_level(current)
            if children_and_meta is None:
                return []

            variables = []
            for child in children_and_meta[0]:
                if isinstance(child, Var):
                    variables.append(child)
                else:
                    variables.extend(get_variables(child))
            return variables

        return tuple(get_variables(self.args))

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
    def create_factory(
        compute_residual: Any = None,
        *,
        constraint_type: Any = "eq_zero",
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
                return Constraint(
                    compute_residual=lambda values, args, kwargs: compute_residual(
                        values, *args, **kwargs
                    ),
                    args=(args, kwargs),
                    constraint_type=constraint_type,
                    jac_mode=jac_mode,
                    jac_batch_size=jac_batch_size,
                    jac_custom_fn=(
                        (
                            lambda jac_fn: lambda values, args, kwargs: jac_fn(
                                values, *args, **kwargs
                            )
                        )(jac_custom_fn)
                    )
                    if jac_custom_fn is not None
                    else None,
                    jac_custom_with_cache_fn=(
                        (
                            lambda jac_fn: lambda values, cache, args, kwargs: jac_fn(
                                values, cache, *args, **kwargs
                            )
                        )(jac_custom_with_cache_fn)
                    )
                    if jac_custom_with_cache_fn is not None
                    else None,
                    name=name if name is not None else compute_residual.__name__,
                )

            return inner

        if compute_residual is None:
            return decorator
        return decorator(compute_residual)


def analyze_constraint(constraint: Any, constraint_index: Any = 0) -> Any:
    from ._core import _AnalyzedCost

    variables = constraint._get_variables()
    assert len(variables) > 0

    if not isinstance(variables[0].id, int):
        batch_axes = variables[0].id.shape
        assert len(batch_axes) in (0, 1)
        for var in variables[1:]:
            assert (() if isinstance(var.id, int) else var.id.shape) == batch_axes, (
                "Batch axes of variables do not match."
            )
        if len(batch_axes) == 1:
            return jax.vmap(lambda c: analyze_constraint(c, constraint_index))(
                constraint
            )

    def _constraint_no_cache(*args) -> Any:
        constraint_out = constraint.compute_residual(*args)
        if isinstance(constraint_out, tuple):
            assert len(constraint_out) == 2
            return constraint_out[0]
        else:
            return constraint_out

    dummy_vals = jax.eval_shape(VarValues.make, variables)
    constraint_dim = onp.prod(
        jax.eval_shape(_constraint_no_cache, dummy_vals, *constraint.args).shape
    )

    al_params = AugmentedLagrangianParams(
        lagrange_multipliers=jnp.zeros(constraint_dim),
        penalty_params=jnp.ones(constraint_dim),
        original_args=constraint.args,
        constraint_index=constraint_index,
    )

    orig_compute_residual = constraint.compute_residual
    orig_constraint_type = constraint.constraint_type

    def augmented_residual_fn(
        vals: Any,
        al_params_inner: Any,
    ) -> Any:
        constraint_out = orig_compute_residual(vals, *al_params_inner.original_args)

        if isinstance(constraint_out, tuple):
            assert len(constraint_out) == 2
            constraint_val = constraint_out[0].flatten()
            jac_cache = constraint_out[1]
            has_cache = True
        else:
            constraint_val = constraint_out.flatten()
            jac_cache = None
            has_cache = False

        lambdas = al_params_inner.lagrange_multipliers
        rho = al_params_inner.penalty_params
        if orig_constraint_type == "leq_zero":
            residual = jnp.sqrt(rho) * jnp.maximum(0.0, constraint_val + lambdas / rho)
        else:
            residual = jnp.sqrt(rho) * (constraint_val + lambdas / rho)

        if has_cache:
            return residual, jac_cache
        return residual

    wrapped_jac_custom_fn = None
    wrapped_jac_custom_with_cache_fn = None

    if constraint.jac_custom_fn is not None:
        orig_jac_fn = constraint.jac_custom_fn

        def _wrapped_jac_custom_fn(
            vals: Any,
            al_params_inner: Any,
        ) -> Any:
            original_jac = orig_jac_fn(vals, *al_params_inner.original_args)

            rho = al_params_inner.penalty_params
            lambdas = al_params_inner.lagrange_multipliers

            if orig_constraint_type == "leq_zero":
                constraint_out = orig_compute_residual(
                    vals, *al_params_inner.original_args
                )
                if isinstance(constraint_out, tuple):
                    constraint_val = constraint_out[0]
                else:
                    constraint_val = constraint_out
                constraint_val = constraint_val.flatten()
                active = (constraint_val + lambdas / rho) > 0
                return jnp.sqrt(rho)[:, None] * original_jac * active[:, None]
            else:
                return jnp.sqrt(rho)[:, None] * original_jac

        wrapped_jac_custom_fn = _wrapped_jac_custom_fn

    if constraint.jac_custom_with_cache_fn is not None:
        orig_jac_with_cache_fn = constraint.jac_custom_with_cache_fn

        def _wrapped_jac_custom_with_cache_fn(
            vals: Any,
            jac_cache: Any,
            al_params_inner: Any,
        ) -> Any:
            original_jac = orig_jac_with_cache_fn(
                vals, jac_cache, *al_params_inner.original_args
            )

            rho = al_params_inner.penalty_params
            lambdas = al_params_inner.lagrange_multipliers

            if orig_constraint_type == "leq_zero":
                constraint_out = orig_compute_residual(
                    vals, *al_params_inner.original_args
                )
                if isinstance(constraint_out, tuple):
                    constraint_val = constraint_out[0]
                else:
                    constraint_val = constraint_out
                constraint_val = constraint_val.flatten()
                active = (constraint_val + lambdas / rho) > 0
                return jnp.sqrt(rho)[:, None] * original_jac * active[:, None]
            else:
                return jnp.sqrt(rho)[:, None] * original_jac

        wrapped_jac_custom_with_cache_fn = _wrapped_jac_custom_with_cache_fn

    def compute_residual_original_fn(
        vals: Any,
        al_params_inner: Any,
    ) -> Any:
        constraint_out = orig_compute_residual(vals, *al_params_inner.original_args)
        if isinstance(constraint_out, tuple):
            return constraint_out[0].flatten()
        return constraint_out.flatten()

    return _AnalyzedCost(
        compute_residual=augmented_residual_fn,
        args=(al_params,),
        jac_mode=constraint.jac_mode,
        jac_batch_size=constraint.jac_batch_size,
        jac_custom_fn=wrapped_jac_custom_fn,
        jac_custom_with_cache_fn=wrapped_jac_custom_with_cache_fn,
        name=f"augmented_{constraint._get_name()}",
        num_variables=len(variables),
        sorted_ids_from_var_type=sort_and_stack_vars(variables),
        residual_flat_dim=constraint_dim,
        constraint_type=constraint.constraint_type,
        compute_residual_original=compute_residual_original_fn,
    )
