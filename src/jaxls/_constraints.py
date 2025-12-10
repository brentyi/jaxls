from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Concatenate, Literal, cast, overload

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp
from jax.tree_util import default_registry
from typing_extensions import Self

from ._variables import Var, VarValues, sort_and_stack_vars

if TYPE_CHECKING:
    from ._core import _AnalyzedCost


# Type aliases for constraints.
type ConstraintFunc[**Args] = Callable[
    Concatenate[VarValues, Args],
    jax.Array,
]
type ConstraintFuncWithJacCache[**Args, TJacobianCache] = Callable[
    Concatenate[VarValues, Args],
    tuple[jax.Array, TJacobianCache],
]

type ConstraintJacobianFunc[**Args] = Callable[
    Concatenate[VarValues, Args],
    jax.Array,
]
type ConstraintJacobianFuncWithCache[**Args, TJacobianCache] = Callable[
    Concatenate[VarValues, TJacobianCache, Args],
    jax.Array,
]

type ConstraintFactory[**Args] = Callable[
    Args,
    Constraint[tuple[Any, ...], dict[str, Any]],
]


@jdc.pytree_dataclass
class AugmentedLagrangianParams[*Args]:
    """Parameters for a single augmented constraint cost.

    Each augmented cost gets its own params with arrays for just that constraint.
    The original constraint args are bundled here for type-safe access.
    """

    lagrange_multipliers: jax.Array
    """Lagrange multipliers for this constraint. Shape: (constraint_flat_dim,)."""

    penalty_params: jax.Array
    """Penalty parameters for this constraint. Shape: (constraint_flat_dim,)."""

    original_args: tuple[*Args]
    """The original constraint args to pass to compute_residual."""

    constraint_index: jdc.Static[int]
    """Index of this constraint in the original problem's stacked_constraints.

    This is needed because augmented costs may be reordered during re-analysis.
    """


@jdc.pytree_dataclass
class Constraint[*Args]:
    """A constraint in our optimization problem.

    Supports two types of constraints:
    - Equality constraints: `h(x) = 0` with `constraint_type="eq_zero"` (default)
    - Inequality constraints: `g(x) <= 0` with `constraint_type="leq_zero"`

    The recommended way to create a constraint is to use the `create_factory`
    decorator on a function that computes the constraint value.

    ```python
    # Equality constraint: h(x) = 0
    @jaxls.Constraint.create_factory(constraint_type="eq_zero")
    def equality_constraint(values: VarValues, [...args]) -> jax.Array:
        return values[var] - target  # Should equal zero

    # Inequality constraint: g(x) <= 0
    @jaxls.Constraint.create_factory(constraint_type="leq_zero")
    def inequality_constraint(values: VarValues, [...args]) -> jax.Array:
        return values[var] - upper_bound  # Should be <= 0

    problem = jaxls.LeastSquaresProblem(
        costs=[...],
        variables=[...],
        constraints=[equality_constraint(...), inequality_constraint(...), ...],
    )
    ```

    Each `Constraint.compute_residual` should take at least one argument that inherits
    from the symbolic variable `jaxls.Var(id)`, where `id` must be a scalar
    integer.

    To create a batch of constraints, a leading batch axis can be added to the
    arguments passed to `Constraint.args`:
    - The batch axis must be the same for all arguments. Leading axes of shape
      `(1,)` are broadcasted.
    - The `id` field of each `jaxls.Var` instance must have shape of either
      `()` (unbatched) or `(batch_size,)` (batched).
    """

    compute_residual: jdc.Static[
        Callable[[VarValues, *Args], jax.Array]
        | Callable[[VarValues, *Args], tuple[jax.Array, Any]]
    ]
    """Constraint computation function. Can either return:
        1. A constraint vector, or
        2. A tuple, where the tuple values should be (constraint, jacobian_cache).

    The second option is useful when custom Jacobian computation benefits from
    intermediate values computed during the constraint computation.

    `jac_custom_with_cache_fn` should be specified in the second case,
    and will be expected to take arguments in the form `(values,
    jacobian_cache, *args)`."""

    args: tuple[*Args]
    """Arguments to the constraint function. This should include at least one
    `jaxls.Var` object, which can either be in the root of the tuple or nested
    within a PyTree structure arbitrarily."""

    constraint_type: jdc.Static[Literal["eq_zero", "leq_zero"]] = "eq_zero"
    """Type of constraint. Supported types:
    - 'eq_zero': h(x) = 0 (equality constraint)
    - 'leq_zero': g(x) <= 0 (inequality constraint)
    """

    jac_mode: jdc.Static[Literal["auto", "forward", "reverse"]] = "auto"
    """Depending on the function being differentiated, it may be faster to use
    forward-mode or reverse-mode autodiff. Ignored if `jac_custom_fn` is
    specified."""

    jac_batch_size: jdc.Static[int | None] = None
    """Batch size for computing Jacobians that can be parallelized. Can be set
    to make tradeoffs between runtime and memory usage.

    If None, we compute all Jacobians in parallel. If 1, we compute Jacobians
    one at a time."""

    jac_custom_fn: jdc.Static[Callable[[VarValues, *Args], jax.Array] | None] = None
    """Optional custom Jacobian function. If None, we use autodiff. Inputs are
    the same as `compute_residual`. Output is a single 2D Jacobian matrix with
    shape (constraint_dim, sum_of_tangent_dims_of_variables)."""

    jac_custom_with_cache_fn: jdc.Static[
        Callable[[VarValues, Any, *Args], jax.Array] | None
    ] = None
    """Optional custom Jacobian function. The same as `jac_custom_fn`, but
    should be used when `compute_residual` returns a tuple with cache."""

    name: jdc.Static[str | None] = None
    """Custom name for debugging and logging."""

    # Methods duplicated from _CostBase to avoid circular import.
    # These provide shared functionality for variable extraction, batch axis handling,
    # and broadcasting.

    def _get_name(self) -> str:
        """Get the name. If not set, falls back to the function name."""
        if self.name is None:
            return self.compute_residual.__name__
        return self.name

    def _get_variables(self) -> tuple[Var, ...]:
        """Extract all Var objects from args (walks the pytree)."""

        def get_variables(current: Any) -> list[Var]:
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

    def _get_batch_axes(self) -> tuple[int, ...]:
        """Get batch axes from variables in args."""
        variables = self._get_variables()
        assert len(variables) != 0, f"No variables found in {type(self).__name__}!"
        return jnp.broadcast_shapes(
            *[() if isinstance(v.id, int) else v.id.shape for v in variables]
        )

    def _broadcast_batch_axes(self) -> Self:
        """Broadcast all args to consistent batch axes."""
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
                # Create a more informative error message
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

    # Simple decorator.
    @overload
    @staticmethod
    def create_factory[**Args_](
        compute_residual: ConstraintFunc[Args_],
    ) -> ConstraintFactory[Args_]: ...

    # Decorator factory with keyword arguments.
    @overload
    @staticmethod
    def create_factory[**Args_](
        *,
        constraint_type: Literal["eq_zero", "leq_zero"] = "eq_zero",
        jac_mode: Literal["auto", "forward", "reverse"] = "auto",
        jac_batch_size: int | None = None,
        name: str | None = None,
    ) -> Callable[[ConstraintFunc[Args_]], ConstraintFactory[Args_]]: ...

    # Decorator factory with custom Jacobian.
    @overload
    @staticmethod
    def create_factory[**Args_](
        *,
        constraint_type: Literal["eq_zero", "leq_zero"] = "eq_zero",
        jac_custom_fn: ConstraintJacobianFunc[Args_],
        jac_batch_size: int | None = None,
        name: str | None = None,
    ) -> Callable[[ConstraintFunc[Args_]], ConstraintFactory[Args_]]: ...

    # Decorator factory with custom Jacobian with cache.
    @overload
    @staticmethod
    def create_factory[**Args_, TJacobianCache](
        *,
        constraint_type: Literal["eq_zero", "leq_zero"] = "eq_zero",
        jac_custom_with_cache_fn: ConstraintJacobianFuncWithCache[
            Args_, TJacobianCache
        ],
        jac_batch_size: int | None = None,
        name: str | None = None,
    ) -> Callable[
        [ConstraintFuncWithJacCache[Args_, TJacobianCache]], ConstraintFactory[Args_]
    ]: ...

    @staticmethod
    def create_factory[**Args_](
        compute_residual: ConstraintFunc[Args_] | None = None,
        *,
        constraint_type: Literal["eq_zero", "leq_zero"] = "eq_zero",
        jac_mode: Literal["auto", "forward", "reverse"] = "auto",
        jac_batch_size: int | None = None,
        jac_custom_fn: ConstraintJacobianFunc[Args_] | None = None,
        jac_custom_with_cache_fn: ConstraintJacobianFuncWithCache[Args_, Any]
        | None = None,
        name: str | None = None,
    ) -> (
        Callable[[ConstraintFunc[Args_]], ConstraintFactory[Args_]]
        | Callable[[ConstraintFuncWithJacCache[Args_, Any]], ConstraintFactory[Args_]]
        | ConstraintFactory[Args_]
    ):
        """Decorator for creating constraints from a constraint function.

        Examples:
            # Equality constraint: h(x) = 0
            @jaxls.Constraint.create_factory(constraint_type="eq_zero")
            def equality_constraint(values: VarValues, var1: SE2Var, target: float) -> jax.Array:
                # Fix variable to target value.
                return values[var1].translation()[0] - target

            # Inequality constraint: g(x) <= 0
            @jaxls.Constraint.create_factory(constraint_type="leq_zero")
            def inequality_constraint(values: VarValues, var1: ScalarVar, max_val: float) -> jax.Array:
                # Variable must be less than or equal to max_val
                return values[var1] - max_val

            # With custom Jacobian:
            @jaxls.Constraint.create_factory(
                constraint_type="eq_zero",
                jac_custom_fn=my_jacobian_fn,
            )
            def constraint_with_custom_jac(values: VarValues, var1: Var) -> jax.Array:
                ...

            # Factory will have the same input signature as the wrapped
            # constraint function, but without the `VarValues` argument. The
            # return type will be `Constraint` instead of `jax.Array`.
            constraint = equality_constraint(var1=SE2Var(0), target=5.0)
            assert isinstance(constraint, jaxls.Constraint)
        """

        def decorator(
            compute_residual: Callable[Concatenate[VarValues, Args_], jax.Array],
        ) -> ConstraintFactory[Args_]:
            def inner(
                *args: Args_.args, **kwargs: Args_.kwargs
            ) -> Constraint[tuple[Any, ...], dict[str, Any]]:
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


def analyze_constraint[*Args](
    constraint: Constraint[*Args], constraint_index: int = 0
) -> _AnalyzedCost[tuple[AugmentedLagrangianParams[*Args]]]:
    """Analyze a constraint and convert it directly to augmented _AnalyzedCost form.

    This creates an _AnalyzedCost that wraps the constraint with the augmented
    Lagrangian formulation, with placeholder AL params that will be updated
    during optimization.

    For equality constraints h(x) = 0:
        r = sqrt(rho) * (h(x) + lambda/rho)

    For inequality constraints g(x) <= 0:
        r = sqrt(rho) * max(0, g(x) + lambda/rho)

    Args:
        constraint: The constraint to analyze.
        constraint_index: Index for this constraint (used by AL solver).

    Returns:
        An _AnalyzedCost with augmented residual and placeholder AL params.
    """
    from ._core import _AnalyzedCost

    variables = constraint._get_variables()
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
                jax.vmap(lambda c: analyze_constraint(c, constraint_index))(constraint),
            )

    # Compute constraint dimension.
    def _constraint_no_cache(*args) -> jax.Array:
        constraint_out = constraint.compute_residual(*args)  # type: ignore
        if isinstance(constraint_out, tuple):
            assert len(constraint_out) == 2
            return constraint_out[0]
        else:
            return constraint_out

    dummy_vals = jax.eval_shape(VarValues.make, variables)
    constraint_dim = onp.prod(
        jax.eval_shape(_constraint_no_cache, dummy_vals, *constraint.args).shape
    )

    # Create placeholder AL params.
    al_params = AugmentedLagrangianParams(
        lagrange_multipliers=jnp.zeros(constraint_dim),
        penalty_params=jnp.ones(constraint_dim),
        original_args=constraint.args,
        constraint_index=constraint_index,
    )

    # Capture constraint for closures.
    orig_compute_residual = constraint.compute_residual
    orig_constraint_type = constraint.constraint_type

    def augmented_residual_fn(
        vals: VarValues,
        al_params_inner: AugmentedLagrangianParams[*Args],
    ) -> jax.Array | tuple[jax.Array, Any]:
        """Compute augmented constraint residual with per-constraint penalty."""
        constraint_out = orig_compute_residual(vals, *al_params_inner.original_args)

        # Handle Jacobian cache if present.
        if isinstance(constraint_out, tuple):
            assert len(constraint_out) == 2
            constraint_val = constraint_out[0].flatten()
            jac_cache = constraint_out[1]
            has_cache = True
        else:
            constraint_val = constraint_out.flatten()
            jac_cache = None
            has_cache = False

        # For inequality constraints: only penalize when violated (max formulation)
        # For equality constraints: always penalize deviation
        lambdas = al_params_inner.lagrange_multipliers
        rho = al_params_inner.penalty_params
        if orig_constraint_type == "leq_zero":
            # g(x) <= 0: penalize only when g(x) + lambda/rho > 0
            residual = jnp.sqrt(rho) * jnp.maximum(0.0, constraint_val + lambdas / rho)
        else:
            # h(x) = 0: always penalize
            residual = jnp.sqrt(rho) * (constraint_val + lambdas / rho)

        if has_cache:
            return residual, jac_cache
        return residual

    # Create wrapper Jacobian functions if original constraint has custom Jacobians.
    wrapped_jac_custom_fn = None
    wrapped_jac_custom_with_cache_fn = None

    if constraint.jac_custom_fn is not None:
        orig_jac_fn = constraint.jac_custom_fn

        def _wrapped_jac_custom_fn(
            vals: VarValues,
            al_params_inner: AugmentedLagrangianParams[*Args],
        ) -> jax.Array:
            """Wrapper Jacobian that applies chain rule for augmented residual."""
            original_jac = orig_jac_fn(vals, *al_params_inner.original_args)

            rho = al_params_inner.penalty_params
            lambdas = al_params_inner.lagrange_multipliers

            if orig_constraint_type == "leq_zero":
                # g(x) <= 0: zero Jacobian when inactive
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
            vals: VarValues,
            jac_cache: Any,
            al_params_inner: AugmentedLagrangianParams[*Args],
        ) -> jax.Array:
            """Wrapper Jacobian with cache that applies chain rule."""
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

    # Create function to compute original constraint value (without augmentation).
    def compute_residual_original_fn(
        vals: VarValues,
        al_params_inner: AugmentedLagrangianParams[*Args],
    ) -> jax.Array:
        """Compute original constraint value (not augmented)."""
        constraint_out = orig_compute_residual(vals, *al_params_inner.original_args)
        if isinstance(constraint_out, tuple):
            return constraint_out[0].flatten()
        return constraint_out.flatten()

    return _AnalyzedCost(
        compute_residual=augmented_residual_fn,  # type: ignore[arg-type]
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
