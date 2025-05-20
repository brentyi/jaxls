import functools
from typing import assert_never

import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from ._core import AnalyzedCost, AnalyzedLeastSquaresProblem, CustomJacobianCache
from ._sparse_matrices import BlockRowSparseMatrix, SparseBlockRow
from ._variables import VarTypeOrdering, VarValues


def compute_problem_jacobian(
    problem: AnalyzedLeastSquaresProblem,
    vals: VarValues,
    jac_cache: tuple[CustomJacobianCache, ...],
) -> BlockRowSparseMatrix:
    """Compute the Jacobian of a least squares problem's residual vector wrt
    the tangent space of it variables."""
    block_rows = list[SparseBlockRow]()
    residual_offset = 0

    compute_jac = functools.partial(
        _compute_cost_jacobian, vals=vals, tangent_ordering=problem.tangent_ordering
    )
    for i, cost in enumerate(problem.stacked_costs):
        # Compute Jacobian for each cost term.
        if cost.jac_batch_size is None:
            stacked_jac = jax.vmap(compute_jac)((cost, jac_cache[i]))
        else:
            # When `batch_size` is `None`, jax.lax.map reduces to a scan
            # (similar to `batch_size=1`).
            stacked_jac = jax.lax.map(
                compute_jac,
                (
                    cost,
                    jac_cache[i],
                ),
                batch_size=cost.jac_batch_size,
            )
        (num_costs,) = cost._get_batch_axes()
        assert stacked_jac.shape == (
            num_costs,
            cost.residual_flat_dim,
            stacked_jac.shape[-1],  # Tangent dimension.
        )
        # Compute block-row representation for sparse Jacobian.
        stacked_jac_start_col = 0
        start_cols = list[jax.Array]()
        block_widths = list[int]()
        for var_type, ids in problem.tangent_ordering.ordered_dict_items(
            # This ordering shouldn't actually matter!
            cost.sorted_ids_from_var_type
        ):
            (num_costs_, num_vars) = ids.shape
            assert num_costs == num_costs_

            # Get one block for each variable.
            for var_idx in range(ids.shape[-1]):
                start_cols.append(
                    jnp.searchsorted(
                        problem.sorted_ids_from_var_type[var_type], ids[..., var_idx]
                    )
                    * var_type.tangent_dim
                    + problem.tangent_start_from_var_type[var_type]
                )
                block_widths.append(var_type.tangent_dim)
                assert start_cols[-1].shape == (num_costs_,)

            stacked_jac_start_col = (
                stacked_jac_start_col + num_vars * var_type.tangent_dim
            )
        assert stacked_jac.shape[-1] == stacked_jac_start_col

        block_rows.append(
            SparseBlockRow(
                num_cols=problem.tangent_dim,
                start_cols=tuple(start_cols),
                block_num_cols=tuple(block_widths),
                blocks_concat=stacked_jac,
            )
        )

        residual_offset += cost.residual_flat_dim * num_costs
    assert residual_offset == problem.residual_dim

    bsparse_jacobian = BlockRowSparseMatrix(
        block_rows=tuple(block_rows),
        shape=(problem.residual_dim, problem.tangent_dim),
    )
    return bsparse_jacobian


def _compute_cost_jacobian(
    inputs: tuple[AnalyzedCost, CustomJacobianCache | None],
    *,
    vals: VarValues,
    tangent_ordering: VarTypeOrdering,
) -> jax.Array:
    """Compute Jacobian for some cost `i`. Designed to be vmapped over the
    first argument for parallelization.

    Shape should be: (single_residual_dim, sum_of_tangent_dims_of_variables)."""
    cost_i, jac_cache_i = inputs
    val_subset = vals._get_subset(
        {
            var_type: jnp.searchsorted(vals.ids_from_type[var_type], ids)
            for var_type, ids in cost_i.sorted_ids_from_var_type.items()
        },
        tangent_ordering,
    )

    # Shape should be: (residual_dim, sum_of_tangent_dims_of_variables).
    if cost_i.jac_custom_fn is not None:
        assert jac_cache_i is None, (
            "`jac_custom_with_cache_fn` should be used if a Jacobian cache is used, not `jac_custom_fn`!"
        )
        return cost_i.jac_custom_fn(vals, *cost_i.args)
    if cost_i.jac_custom_with_cache_fn is not None:
        assert jac_cache_i is not None, (
            "`jac_custom_with_cache_fn` was specified, but no cache was returned by `compute_residual`!"
        )
        return cost_i.jac_custom_with_cache_fn(vals, jac_cache_i, *cost_i.args)

    # Compute Jacobian using autodiff.
    assert jac_cache_i is None, (
        "`jac_cache` should be None if no custom Jacobian is used!"
    )
    match cost_i.jac_mode:
        case "auto":
            jacfwd_or_jacrev = (
                jax.jacrev
                if cost_i.residual_flat_dim < val_subset._get_tangent_dim()
                else jax.jacfwd
            )
        case "forward":
            jacfwd_or_jacrev = jax.jacfwd
        case "reverse":
            jacfwd_or_jacrev = jax.jacrev
        case "central_difference":
            return _compute_numerical_jacobian(cost_i, val_subset, tangent_ordering)
        case _:
            assert_never(cost_i.jac_mode)

    return jacfwd_or_jacrev(
        # We flatten the output of compute_residual before
        # computing Jacobian. The Jacobian is computed with respect
        # to the flattened residual.
        lambda tangent: cost_i.compute_residual_flat(
            val_subset._retract(tangent, tangent_ordering),
            *cost_i.args,
        )
    )(jnp.zeros((val_subset._get_tangent_dim(),)))


def _compute_numerical_jacobian(
    cost_i: AnalyzedCost,
    val_subset: VarValues,
    tangent_ordering: VarTypeOrdering,
):
    """Compute numerical Jacobian."""
    tangent_eps = list[jax.Array]()
    for (
        var_type,
        stacked_vals,
    ) in tangent_ordering.ordered_dict_items(val_subset.vals_from_type):
        (num_vars,) = val_subset.ids_from_type[var_type].shape
        flat_vals = jax.vmap(lambda x: ravel_pytree(x)[0])(stacked_vals)
        assert flat_vals.shape[0] == num_vars
        var_norms = jnp.linalg.norm(flat_vals, axis=-1, keepdims=True)
        tangent_eps.append(
            jnp.broadcast_to(
                var_norms * 1e-6 + 1e-6,
                (num_vars, var_type.tangent_dim),
            )
        )
    tangent_eps, _ = ravel_pytree(tangent_eps)
    assert tangent_eps.shape == (val_subset._get_tangent_dim(),)
    tangent_eps_pair = jnp.stack([tangent_eps, -tangent_eps], axis=0)
    assert tangent_eps_pair.shape == (2, val_subset._get_tangent_dim())
    samples = jax.vmap(
        lambda eps_dense: jax.vmap(
            lambda eps_dim: cost_i.compute_residual_flat(
                # We perturb along a single dimension.
                val_subset._retract(eps_dim, tangent_ordering),
                *cost_i.args,
            ),
            out_axes=1,
        )(jnp.diagflat(eps_dense))  # (n,) => diagonal (n, n)
    )(tangent_eps_pair)
    assert isinstance(samples, jax.Array)
    assert samples.shape == (
        2,
        cost_i.residual_flat_dim,
        val_subset._get_tangent_dim(),
    )
    return (samples[0] - samples[1]) / (2 * tangent_eps[None, :])
