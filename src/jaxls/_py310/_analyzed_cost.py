from __future__ import annotations

from typing import Any

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp
from jax.nn import relu

from ._cost import Cost
from ._variables import VarValues, sort_and_stack_vars


@jdc.pytree_dataclass
class AugmentedLagrangianParams:
    @classmethod
    def __class_getitem__(cls, params):
        return cls

    lagrange_multipliers: Any

    penalty_params: Any

    original_args: Any

    constraint_index: jdc.Static[Any]


@jdc.pytree_dataclass(kw_only=True)
class _AnalyzedCost(Cost[Any]):
    @classmethod
    def __class_getitem__(cls, params):
        return cls

    num_variables: jdc.Static[Any]
    sorted_ids_from_var_type: Any
    residual_flat_dim: jdc.Static[Any] = 0

    compute_residual_original: jdc.Static[Any] = None

    def compute_residual_flat(self, vals: Any, *args: Any) -> Any:
        out = self.compute_residual(vals, *args)

        if isinstance(out, tuple):
            assert len(out) == 2
            out = (out[0].flatten(), out[1])
        else:
            out = out.flatten()

        return out

    @staticmethod
    @jdc.jit
    def _make(
        cost: Any,
    ) -> Any:
        if cost.kind != "l2_squared":
            return _augment_constraint_cost(cost)

        variables = cost._get_variables()
        assert len(variables) > 0

        if not isinstance(variables[0].id, int):
            batch_axes = variables[0].id.shape
            assert len(batch_axes) in (0, 1)
            for var in variables[1:]:
                assert (
                    () if isinstance(var.id, int) else var.id.shape
                ) == batch_axes, "Batch axes of variables do not match."
            if len(batch_axes) == 1:
                return jax.vmap(_AnalyzedCost._make)(cost)

        def _residual_no_cache(*args) -> Any:
            residual_out = cost.compute_residual(*args)
            if isinstance(residual_out, tuple):
                assert len(residual_out) == 2
                return residual_out[0]
            else:
                return residual_out

        dummy_vals = jax.eval_shape(VarValues.make, variables)
        residual_dim = int(
            onp.prod(jax.eval_shape(_residual_no_cache, dummy_vals, *cost.args).shape)
        )

        return _AnalyzedCost(
            compute_residual=cost.compute_residual,
            args=cost.args,
            kind=cost.kind,
            jac_mode=cost.jac_mode,
            jac_batch_size=cost.jac_batch_size,
            jac_custom_fn=cost.jac_custom_fn,
            jac_custom_with_cache_fn=cost.jac_custom_with_cache_fn,
            name=cost.name,
            num_variables=len(variables),
            sorted_ids_from_var_type=sort_and_stack_vars(variables),
            residual_flat_dim=residual_dim,
        )

    def _compute_block_sparse_jac_indices(
        self,
        tangent_ordering: Any,
        sorted_ids_from_var_type: Any,
        tangent_start_from_var_type: Any,
    ) -> Any:
        col_indices = list()
        for var_type, ids in tangent_ordering.ordered_dict_items(
            self.sorted_ids_from_var_type
        ):
            var_indices = jnp.searchsorted(sorted_ids_from_var_type[var_type], ids)
            tangent_start = tangent_start_from_var_type[var_type]
            tangent_indices = (
                onp.arange(tangent_start, tangent_start + var_type.tangent_dim)[None, :]
                + var_indices[:, None] * var_type.tangent_dim
            )
            assert tangent_indices.shape == (
                var_indices.shape[0],
                var_type.tangent_dim,
            )
            col_indices.append(tangent_indices.flatten())
        rows, cols = jnp.meshgrid(
            jnp.arange(self.residual_flat_dim),
            jnp.concatenate(col_indices, axis=0),
            indexing="ij",
        )
        return rows, cols


def _augment_constraint_cost(cost: Any, constraint_index: Any = 0) -> Any:
    assert cost.kind != "l2_squared", (
        "Only constraint-mode costs should be augmented here"
    )

    variables = cost._get_variables()
    assert len(variables) > 0

    if not isinstance(variables[0].id, int):
        batch_axes = variables[0].id.shape
        assert len(batch_axes) in (0, 1)
        for var in variables[1:]:
            assert (() if isinstance(var.id, int) else var.id.shape) == batch_axes, (
                "Batch axes of variables do not match."
            )
        if len(batch_axes) == 1:
            return jax.vmap(lambda c: _augment_constraint_cost(c, constraint_index))(
                cost
            )

    def _constraint_no_cache(*args) -> Any:
        constraint_out = cost.compute_residual(*args)
        if isinstance(constraint_out, tuple):
            assert len(constraint_out) == 2
            return constraint_out[0]
        else:
            return constraint_out

    dummy_vals = jax.eval_shape(VarValues.make, variables)
    constraint_dim = int(
        onp.prod(jax.eval_shape(_constraint_no_cache, dummy_vals, *cost.args).shape)
    )

    al_params = AugmentedLagrangianParams(
        lagrange_multipliers=jnp.zeros(constraint_dim),
        penalty_params=jnp.array(1.0),
        original_args=cost.args,
        constraint_index=constraint_index,
    )

    orig_compute_residual = cost.compute_residual
    orig_kind = cost.kind

    is_leq = orig_kind == "constraint_leq_zero"
    is_geq = orig_kind == "constraint_geq_zero"
    is_inequality = is_leq or is_geq

    needs_active_mask_cache = is_inequality and (
        cost.jac_custom_fn is not None or cost.jac_custom_with_cache_fn is not None
    )

    def augmented_residual_fn(
        vals: Any,
        al_params_inner: Any,
    ) -> Any:
        constraint_out = orig_compute_residual(vals, *al_params_inner.original_args)

        if isinstance(constraint_out, tuple):
            assert len(constraint_out) == 2
            constraint_val = constraint_out[0].flatten()
            orig_jac_cache = constraint_out[1]
        else:
            constraint_val = constraint_out.flatten()
            orig_jac_cache = None

        if is_geq:
            constraint_val = -constraint_val

        lambdas = al_params_inner.lagrange_multipliers
        rho = al_params_inner.penalty_params
        if is_inequality:
            active = (constraint_val + lambdas / rho) > 0
            residual = jnp.sqrt(rho) * relu(constraint_val + lambdas / rho)
        else:
            active = None
            residual = jnp.sqrt(rho) * (constraint_val + lambdas / rho)

        if orig_jac_cache is not None or needs_active_mask_cache:
            return residual, (orig_jac_cache, active)
        return residual

    wrapped_jac_custom_fn = None
    wrapped_jac_custom_with_cache_fn = None

    if cost.jac_custom_fn is not None:
        orig_jac_fn = cost.jac_custom_fn

        if is_inequality:

            def _wrapped_jac_with_cache_from_custom_fn(
                vals: Any,
                jac_cache: Any,
                al_params_inner: Any,
            ) -> Any:
                original_jac = orig_jac_fn(vals, *al_params_inner.original_args)
                if is_geq:
                    original_jac = -original_jac
                rho = al_params_inner.penalty_params
                _, active = jac_cache
                return jnp.sqrt(rho) * original_jac * active[:, None]

            wrapped_jac_custom_with_cache_fn = _wrapped_jac_with_cache_from_custom_fn
        else:

            def _wrapped_jac_custom_fn(
                vals: Any,
                al_params_inner: Any,
            ) -> Any:
                original_jac = orig_jac_fn(vals, *al_params_inner.original_args)
                rho = al_params_inner.penalty_params
                return jnp.sqrt(rho) * original_jac

            wrapped_jac_custom_fn = _wrapped_jac_custom_fn

    if cost.jac_custom_with_cache_fn is not None:
        orig_jac_with_cache_fn = cost.jac_custom_with_cache_fn

        def _wrapped_jac_custom_with_cache_fn(
            vals: Any,
            jac_cache: Any,
            al_params_inner: Any,
        ) -> Any:
            orig_cache, active = jac_cache
            original_jac = orig_jac_with_cache_fn(
                vals, orig_cache, *al_params_inner.original_args
            )

            if is_geq:
                original_jac = -original_jac

            rho = al_params_inner.penalty_params
            if is_inequality:
                assert active is not None
                return jnp.sqrt(rho) * original_jac * active[:, None]
            else:
                return jnp.sqrt(rho) * original_jac

        wrapped_jac_custom_with_cache_fn = _wrapped_jac_custom_with_cache_fn

    def compute_residual_original_fn(
        vals: Any,
        al_params_inner: Any,
    ) -> Any:
        constraint_out = orig_compute_residual(vals, *al_params_inner.original_args)
        if isinstance(constraint_out, tuple):
            constraint_val = constraint_out[0].flatten()
        else:
            constraint_val = constraint_out.flatten()

        if is_geq:
            constraint_val = -constraint_val
        return constraint_val

    return _AnalyzedCost(
        compute_residual=augmented_residual_fn,
        args=(al_params,),
        kind=cost.kind,
        jac_mode=cost.jac_mode,
        jac_batch_size=cost.jac_batch_size,
        jac_custom_fn=wrapped_jac_custom_fn,
        jac_custom_with_cache_fn=wrapped_jac_custom_with_cache_fn,
        name=f"augmented_{cost._get_name()}",
        num_variables=len(variables),
        sorted_ids_from_var_type=sort_and_stack_vars(variables),
        residual_flat_dim=constraint_dim,
        compute_residual_original=compute_residual_original_fn,
    )
