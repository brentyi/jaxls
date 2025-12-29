from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp
from jax.nn import relu

from ._cost import Cost, CustomJacobianCache
from ._variables import Var, VarTypeOrdering, VarValues, sort_and_stack_vars

if TYPE_CHECKING:
    pass


@jdc.pytree_dataclass
class AugmentedLagrangianParams[*Args]:
    """Parameters for a single augmented constraint cost.

    Each augmented cost gets its own params with arrays for just that constraint.
    The original cost args are bundled here for type-safe access.
    """

    lagrange_multipliers: jax.Array
    """Lagrange multipliers for this constraint. Shape: (constraint_flat_dim,)."""

    penalty_params: jax.Array
    """Penalty parameter for this constraint instance. Shape: () (scalar)."""

    original_args: tuple[*Args]
    """The original cost args to pass to compute_residual."""

    constraint_index: jdc.Static[int]
    """Index used to group constraints with the same structure for vectorized updates.

    Constraints with the same structure (pytree shape, array shapes) are assigned the
    same constraint_index so the AL solver can update their parameters together.
    """


@jdc.pytree_dataclass(kw_only=True)
class _AnalyzedCost[*Args](Cost[*Args]):
    """Analyzed cost ready for optimization.

    Extends Cost with precomputed metadata needed for efficient Jacobian
    computation and sparse matrix assembly. Used for both regular costs
    and constraint-mode costs internally.
    """

    num_variables: jdc.Static[int]
    sorted_ids_from_var_type: dict[type[Var[Any]], jax.Array]
    residual_flat_dim: jdc.Static[int] = 0

    # For constraint-mode costs, stores the original constraint function
    # (before augmented Lagrangian transformation). None for regular costs.
    compute_residual_original: jdc.Static[Callable[..., jax.Array] | None] = None
    """For constraint-mode costs, computes the raw constraint value (not augmented).
    Takes (vals, al_params) and returns the original constraint value."""

    def compute_residual_flat(
        self, vals: VarValues, *args: *Args
    ) -> jax.Array | tuple[jax.Array, CustomJacobianCache]:
        """Compute residual and flatten to 1D array.

        Calls the underlying compute_residual function and flattens the output.
        If compute_residual returns a tuple (residual, cache), the flattened
        residual and cache are returned together.

        Args:
            vals: Variable values to evaluate the residual at.
            *args: Additional arguments passed to compute_residual.

        Returns:
            Either a flattened residual array, or a tuple of (flattened_residual, cache)
            if the compute_residual function returns a Jacobian cache.
        """
        out = self.compute_residual(vals, *args)

        # Flatten residual vector.
        if isinstance(out, tuple):
            assert len(out) == 2
            out = (out[0].flatten(), out[1])
        else:
            out = out.flatten()

        return out

    @staticmethod
    @jdc.jit
    def _make[*Args_](
        cost: Cost[*Args_],
    ) -> (
        _AnalyzedCost[*Args_] | _AnalyzedCost[tuple[AugmentedLagrangianParams[*Args_]]]
    ):
        """Construct an analyzed cost from Cost.

        For constraint-mode costs (constraint is not None), this converts to augmented
        Lagrangian form with placeholder AL params.
        """

        # Handle constraint modes -> augmented _AnalyzedCost conversion
        if cost.kind != "l2_squared":
            return _augment_constraint_cost(cost)

        # Handle regular cost -> _AnalyzedCost conversion
        variables = cost._get_variables()
        assert len(variables) > 0

        # Support batch axis.
        if not isinstance(variables[0].id, int):
            batch_axes = variables[0].id.shape
            assert len(batch_axes) in (0, 1)
            for var in variables[1:]:
                assert (
                    () if isinstance(var.id, int) else var.id.shape
                ) == batch_axes, "Batch axes of variables do not match."
            if len(batch_axes) == 1:
                return jax.vmap(_AnalyzedCost._make)(cost)

        # Same as `compute_residual`, but removes Jacobian cache if present.
        def _residual_no_cache(*args) -> jax.Array:
            residual_out = cost.compute_residual(*args)  # type: ignore
            if isinstance(residual_out, tuple):
                assert len(residual_out) == 2
                return residual_out[0]
            else:
                return residual_out

        # Cache the residual dimension for this cost.
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
        tangent_ordering: VarTypeOrdering,
        sorted_ids_from_var_type: dict[type[Var[Any]], jax.Array],
        tangent_start_from_var_type: dict[type[Var[Any]], int],
    ) -> tuple[jax.Array, jax.Array]:
        """Compute row and column indices for block-sparse Jacobian.

        Computes the sparsity pattern for a Jacobian of shape
        (residual_dim, total_tangent_dim). The indices map each residual
        element to its corresponding tangent space elements.

        Args:
            tangent_ordering: Ordering of variable types for consistent indexing.
            sorted_ids_from_var_type: Sorted variable IDs for each type.
            tangent_start_from_var_type: Starting column index for each variable type.

        Returns:
            Tuple of (row_indices, col_indices) for the sparse Jacobian.
            Row indices start at 0 for this cost's residual block.
        """
        col_indices = list[jax.Array]()
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
            col_indices.append(cast(jax.Array, tangent_indices).flatten())
        rows, cols = jnp.meshgrid(
            jnp.arange(self.residual_flat_dim),
            jnp.concatenate(col_indices, axis=0),
            indexing="ij",
        )
        return rows, cols


def _augment_constraint_cost[*Args](
    cost: Cost[*Args], constraint_index: int = 0
) -> _AnalyzedCost[tuple[AugmentedLagrangianParams[*Args]]]:
    """Augment a constraint-mode cost and convert it to augmented _AnalyzedCost form.

    This creates an _AnalyzedCost that wraps the cost with the augmented
    Lagrangian formulation, with placeholder AL params that will be updated
    during optimization.

    For equality constraints (kind="constraint_eq_zero"): h(x) = 0
        r = sqrt(rho) * (h(x) + lambda/rho)

    For inequality constraints (kind="constraint_leq_zero"): g(x) <= 0
        r = sqrt(rho) * max(0, g(x) + lambda/rho)

    For inequality constraints (kind="constraint_geq_zero"): g(x) >= 0
        Internally converted to -g(x) <= 0, then treated as leq_zero.

    Args:
        cost: The constraint-mode cost to analyze.
        constraint_index: Index for this constraint (used by AL solver).

    Returns:
        An _AnalyzedCost with augmented residual and placeholder AL params.
    """
    assert cost.kind != "l2_squared", (
        "Only constraint-mode costs should be augmented here"
    )

    variables = cost._get_variables()
    assert len(variables) > 0

    # Support batch axis.
    if not isinstance(variables[0].id, int):
        batch_axes = variables[0].id.shape
        assert len(batch_axes) in (0, 1)
        for var in variables[1:]:
            assert (() if isinstance(var.id, int) else var.id.shape) == batch_axes, (
                "Batch axes of variables do not match."
            )
        if len(batch_axes) == 1:
            return cast(
                _AnalyzedCost[Any],
                jax.vmap(lambda c: _augment_constraint_cost(c, constraint_index))(cost),
            )

    # Compute constraint dimension.
    def _constraint_no_cache(*args) -> jax.Array:
        constraint_out = cost.compute_residual(*args)  # type: ignore
        if isinstance(constraint_out, tuple):
            assert len(constraint_out) == 2
            return constraint_out[0]
        else:
            return constraint_out

    dummy_vals = jax.eval_shape(VarValues.make, variables)
    constraint_dim = int(
        onp.prod(jax.eval_shape(_constraint_no_cache, dummy_vals, *cost.args).shape)
    )

    # Create placeholder AL params.
    # Note: penalty_params is scalar (per-instance), lagrange_multipliers is per-element.
    al_params = AugmentedLagrangianParams(
        lagrange_multipliers=jnp.zeros(constraint_dim),
        penalty_params=jnp.array(1.0),  # Scalar per-instance penalty.
        original_args=cost.args,
        constraint_index=constraint_index,
    )

    # Capture cost for closures.
    orig_compute_residual = cost.compute_residual
    orig_kind = cost.kind

    # Determine if this is an inequality constraint
    is_leq = orig_kind == "constraint_leq_zero"
    is_geq = orig_kind == "constraint_geq_zero"
    is_inequality = is_leq or is_geq

    # Determine if we need to return a cache for the active mask.
    # This is needed when we have inequality constraints with custom Jacobians.
    needs_active_mask_cache = is_inequality and (
        cost.jac_custom_fn is not None or cost.jac_custom_with_cache_fn is not None
    )

    def augmented_residual_fn(
        vals: VarValues,
        al_params_inner: AugmentedLagrangianParams[*Args],
    ) -> jax.Array | tuple[jax.Array, Any]:
        """Compute augmented constraint residual with per-constraint penalty.

        Applies the augmented Lagrangian transformation to convert constraint
        satisfaction into a squared penalty term suitable for least-squares.
        """
        constraint_out = orig_compute_residual(vals, *al_params_inner.original_args)

        # Handle Jacobian cache if present.
        if isinstance(constraint_out, tuple):
            assert len(constraint_out) == 2
            constraint_val = constraint_out[0].flatten()
            orig_jac_cache = constraint_out[1]
        else:
            constraint_val = constraint_out.flatten()
            orig_jac_cache = None

        # For geq_zero, negate to convert to leq_zero form
        if is_geq:
            constraint_val = -constraint_val

        # For inequality constraints: only penalize when violated (max formulation)
        # For equality constraints: always penalize deviation
        lambdas = al_params_inner.lagrange_multipliers
        rho = al_params_inner.penalty_params
        if is_inequality:
            # g(x) <= 0: penalize only when g(x) + lambda/rho > 0
            active = (constraint_val + lambdas / rho) > 0
            residual = jnp.sqrt(rho) * relu(constraint_val + lambdas / rho)
        else:
            # h(x) = 0: always penalize
            active = None
            residual = jnp.sqrt(rho) * (constraint_val + lambdas / rho)

        # Return cache if needed (either original cache or active mask for custom Jacobians).
        if orig_jac_cache is not None or needs_active_mask_cache:
            # Bundle original cache with active mask for inequality constraints.
            return residual, (orig_jac_cache, active)
        return residual

    # Create wrapper Jacobian functions if original cost has custom Jacobians.
    # When we have inequality constraints with custom Jacobians, we cache the active
    # mask in the residual computation to avoid redundant constraint evaluation.
    wrapped_jac_custom_fn = None
    wrapped_jac_custom_with_cache_fn = None

    if cost.jac_custom_fn is not None:
        orig_jac_fn = cost.jac_custom_fn

        if is_inequality:
            # For inequality constraints, use cache-based wrapper to get active mask.
            def _wrapped_jac_with_cache_from_custom_fn(
                vals: VarValues,
                jac_cache: tuple[None, jax.Array],
                al_params_inner: AugmentedLagrangianParams[*Args],
            ) -> jax.Array:
                """Wrapper using cached active mask for inequality constraint Jacobian."""
                original_jac = orig_jac_fn(vals, *al_params_inner.original_args)
                if is_geq:
                    original_jac = -original_jac
                rho = al_params_inner.penalty_params  # Scalar per-instance penalty.
                _, active = jac_cache  # Extract cached active mask.
                return jnp.sqrt(rho) * original_jac * active[:, None]

            wrapped_jac_custom_with_cache_fn = _wrapped_jac_with_cache_from_custom_fn
        else:
            # Equality constraints don't need active mask.
            def _wrapped_jac_custom_fn(
                vals: VarValues,
                al_params_inner: AugmentedLagrangianParams[*Args],
            ) -> jax.Array:
                """Wrapper Jacobian for equality constraints."""
                original_jac = orig_jac_fn(vals, *al_params_inner.original_args)
                rho = al_params_inner.penalty_params  # Scalar per-instance penalty.
                return jnp.sqrt(rho) * original_jac

            wrapped_jac_custom_fn = _wrapped_jac_custom_fn

    if cost.jac_custom_with_cache_fn is not None:
        orig_jac_with_cache_fn = cost.jac_custom_with_cache_fn

        def _wrapped_jac_custom_with_cache_fn(
            vals: VarValues,
            jac_cache: tuple[CustomJacobianCache, jax.Array | None],
            al_params_inner: AugmentedLagrangianParams[*Args],
        ) -> jax.Array:
            """Wrapper Jacobian with cache that applies chain rule for augmentation."""
            orig_cache, active = jac_cache  # Unpack bundled cache.
            original_jac = orig_jac_with_cache_fn(
                vals, orig_cache, *al_params_inner.original_args
            )

            # For geq_zero, negate the Jacobian.
            if is_geq:
                original_jac = -original_jac

            rho = al_params_inner.penalty_params  # Scalar per-instance penalty.
            if is_inequality:
                assert active is not None
                return jnp.sqrt(rho) * original_jac * active[:, None]
            else:
                return jnp.sqrt(rho) * original_jac

        wrapped_jac_custom_with_cache_fn = _wrapped_jac_custom_with_cache_fn

    # Create function to compute original constraint value (without augmentation).
    def compute_residual_original_fn(
        vals: VarValues,
        al_params_inner: AugmentedLagrangianParams[*Args],
    ) -> jax.Array:
        """Compute original constraint value (not augmented).

        Used by the AL solver to track constraint violations and update
        Lagrange multipliers.
        """
        constraint_out = orig_compute_residual(vals, *al_params_inner.original_args)
        if isinstance(constraint_out, tuple):
            constraint_val = constraint_out[0].flatten()
        else:
            constraint_val = constraint_out.flatten()
        # For geq_zero, negate to get the leq_zero equivalent
        if is_geq:
            constraint_val = -constraint_val
        return constraint_val

    return _AnalyzedCost(
        compute_residual=augmented_residual_fn,  # type: ignore[arg-type]
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
