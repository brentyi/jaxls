from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Literal,
    Self,
    cast,
    overload,
)

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from jax.tree_util import default_registry
from typing_extensions import deprecated

if TYPE_CHECKING:
    from ._variables import Var, VarValues


# Type alias for Jacobian cache returned by compute_residual.
CustomJacobianCache = Any


type ResidualFunc[**Args] = Callable[
    Concatenate[VarValues, Args],
    jax.Array,
]
type ResidualFuncWithJacCache[**Args, TJacobianCache: CustomJacobianCache] = Callable[
    Concatenate[VarValues, Args],
    tuple[jax.Array, TJacobianCache],
]

type JacobianFunc[**Args] = Callable[
    Concatenate[VarValues, Args],
    jax.Array,
]
type JacobianFuncWithCache[**Args, TJacobianCache: CustomJacobianCache] = Callable[
    Concatenate[VarValues, TJacobianCache, Args],
    jax.Array,
]

type CostFactory[**Args] = Callable[
    Args,
    Cost[tuple[Any, ...], dict[str, Any]],
]


type CostKind = Literal[
    "l2_squared", "constraint_eq_zero", "constraint_leq_zero", "constraint_geq_zero"
]


@jdc.pytree_dataclass
class Cost[*Args]:
    """A cost or constraint term in our optimization problem.

    The ``kind`` field determines how the residual function is interpreted:

    - ``"l2_squared"`` (default): Minimize squared L2 norm: ``||r(x)||^2``
    - ``"constraint_eq_zero"``: Equality constraint: ``r(x) = 0``
    - ``"constraint_leq_zero"``: Inequality constraint: ``r(x) <= 0``
    - ``"constraint_geq_zero"``: Inequality constraint: ``r(x) >= 0``

    Use the :meth:`~jaxls.Cost.factory` decorator to create costs from a
    residual function.

    Each ``Cost.compute_residual`` must include at least one ``jaxls.Var(id)``
    in its inputs, where ``id`` is a scalar integer. Variables can appear
    anywhere in the input structure, including nested within pytrees (lists,
    dicts, dataclasses, etc.).

    To create a batch of costs, a leading batch axis can be added to the
    arguments passed to ``Cost.args``:

    - The batch axis must be the same for all arguments. Leading axes of shape
      ``(1,)`` are broadcasted.
    - The ``id`` field of each ``jaxls.Var`` instance must have shape of either
      ``()`` (unbatched) or ``(batch_size,)`` (batched).
    """

    compute_residual: jdc.Static[
        Callable[[VarValues, *Args], jax.Array]
        | Callable[[VarValues, *Args], tuple[jax.Array, Any]]
    ]
    """Residual/constraint computation function. Can either return:
        1. A residual/constraint vector, or
        2. A tuple of (residual/constraint, jacobian_cache).

    The second option is useful when custom Jacobian computation benefits from
    intermediate values computed during the residual computation."""

    args: tuple[*Args]
    """Arguments to the residual function. This should include at least one
    `jaxls.Var` object, which can either be in the root of the tuple or nested
    within a PyTree structure arbitrarily."""

    kind: jdc.Static[CostKind] = "l2_squared"
    """How the residual function is interpreted:
    - 'l2_squared': Minimize squared L2 norm ||r(x)||^2
    - 'constraint_eq_zero': Equality constraint r(x) = 0
    - 'constraint_leq_zero': Inequality constraint r(x) <= 0
    - 'constraint_geq_zero': Inequality constraint r(x) >= 0
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
    shape (residual_dim, sum_of_tangent_dims_of_variables)."""

    jac_custom_with_cache_fn: jdc.Static[
        Callable[[VarValues, Any, *Args], jax.Array] | None
    ] = None
    """Optional custom Jacobian function. The same as `jac_custom_fn`, but
    should be used when `compute_residual` returns a tuple with cache."""

    name: jdc.Static[str | None] = None
    """Custom name for debugging and logging."""

    def _get_name(self) -> str:
        """Get the name. If not set, falls back to the function name."""
        if self.name is None:
            return self.compute_residual.__name__
        return self.name

    def _get_variables(self) -> tuple[Var, ...]:
        """Extract all Var objects from args by walking the pytree structure.

        Returns:
            Tuple of all Var instances found in the cost arguments.
        """
        from ._variables import Var

        def get_variables_recursive(current: Any) -> list[Var]:
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

    def _get_batch_axes(self) -> tuple[int, ...]:
        """Get batch axes from variables in args.

        Returns:
            Tuple of batch dimensions, or empty tuple if unbatched.
        """
        variables = self._get_variables()
        assert len(variables) != 0, f"No variables found in {type(self).__name__}!"
        return jnp.broadcast_shapes(
            *[() if isinstance(v.id, int) else v.id.shape for v in variables]
        )

    def _broadcast_batch_axes(self) -> Self:
        """Broadcast all args to consistent batch axes.

        Returns:
            A new Cost with all arguments broadcasted to matching batch dimensions.
        """
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
    def factory[**Args_](
        compute_residual: ResidualFunc[Args_],
    ) -> CostFactory[Args_]: ...

    # Decorator factory with keyword arguments.
    @overload
    @staticmethod
    def factory[**Args_](
        *,
        kind: CostKind = "l2_squared",
        jac_mode: Literal["auto", "forward", "reverse"] = "auto",
        jac_batch_size: int | None = None,
        name: str | None = None,
    ) -> Callable[[ResidualFunc[Args_]], CostFactory[Args_]]: ...

    # Decorator factory with keyword arguments + custom Jacobian.
    # `jac_mode` is ignored in this case.
    @overload
    @staticmethod
    def factory[**Args_](
        *,
        kind: CostKind = "l2_squared",
        jac_custom_fn: JacobianFunc[Args_],
        jac_batch_size: int | None = None,
        name: str | None = None,
    ) -> Callable[[ResidualFunc[Args_]], CostFactory[Args_]]: ...

    # Decorator factory with keyword arguments + custom Jacobian with cache.
    # `jac_mode` is ignored in this case.
    @overload
    @staticmethod
    def factory[**Args_, TJacobianCache](
        *,
        kind: CostKind = "l2_squared",
        jac_custom_with_cache_fn: JacobianFuncWithCache[Args_, TJacobianCache],
        jac_batch_size: int | None = None,
        name: str | None = None,
    ) -> Callable[
        [ResidualFuncWithJacCache[Args_, TJacobianCache]], CostFactory[Args_]
    ]: ...

    @staticmethod
    def factory[**Args_](
        compute_residual: ResidualFunc[Args_] | None = None,
        *,
        kind: CostKind = "l2_squared",
        jac_mode: Literal["auto", "forward", "reverse"] = "auto",
        jac_batch_size: int | None = None,
        jac_custom_fn: JacobianFunc[Args_] | None = None,
        jac_custom_with_cache_fn: JacobianFuncWithCache[Args_, Any] | None = None,
        name: str | None = None,
    ) -> (
        Callable[[ResidualFunc[Args_]], CostFactory[Args_]]
        | Callable[[ResidualFuncWithJacCache[Args_, Any]], CostFactory[Args_]]
        | CostFactory[Args_]
    ):
        """Decorator for creating costs from a residual function.

        The decorated function should take ``VarValues`` as its first argument
        and return a residual array. The resulting factory will have the same
        signature but without the ``VarValues`` argument.

        Args:
            kind: How to interpret the residual (default: ``"l2_squared"``).
            jac_mode: Autodiff mode for Jacobians (``"auto"``, ``"forward"``,
                or ``"reverse"``).
            jac_batch_size: Batch size for Jacobian computation. Set to 1 to
                reduce memory usage.
        """

        def decorator(
            compute_residual: Callable[Concatenate[VarValues, Args_], jax.Array],
        ) -> CostFactory[Args_]:
            def inner(
                *args: Args_.args, **kwargs: Args_.kwargs
            ) -> Cost[tuple[Any, ...], dict[str, Any]]:
                return Cost(
                    compute_residual=lambda values, args, kwargs: compute_residual(
                        values, *args, **kwargs
                    ),
                    args=(args, kwargs),
                    kind=kind,
                    jac_mode=jac_mode,
                    jac_batch_size=jac_batch_size,
                    jac_custom_fn=(
                        lambda values, args, kwargs: cast(
                            JacobianFunc[Args_], jac_custom_fn
                        )(values, *args, **kwargs)
                    )
                    if jac_custom_fn is not None
                    else None,
                    jac_custom_with_cache_fn=(
                        lambda values, cache, args, kwargs: cast(
                            JacobianFuncWithCache[Args_, Any],
                            jac_custom_with_cache_fn,
                        )(values, cache, *args, **kwargs)
                    )
                    if jac_custom_with_cache_fn is not None
                    else None,
                    name=name if name is not None else compute_residual.__name__,
                )

            return inner

        if compute_residual is None:
            return decorator
        return decorator(compute_residual)

    if not TYPE_CHECKING:

        @staticmethod
        @deprecated("Use Cost.factory instead of Cost.create_factory")
        def create_factory[**Args_](
            compute_residual: ResidualFunc[Args_] | None = None,
            *,
            kind: CostKind = "l2_squared",
            jac_mode: Literal["auto", "forward", "reverse"] = "auto",
            jac_batch_size: int | None = None,
            jac_custom_fn: JacobianFunc[Args_] | None = None,
            jac_custom_with_cache_fn: JacobianFuncWithCache[Args_, Any] | None = None,
            name: str | None = None,
        ) -> (
            Callable[[ResidualFunc[Args_]], CostFactory[Args_]]
            | Callable[[ResidualFuncWithJacCache[Args_, Any]], CostFactory[Args_]]
            | CostFactory[Args_]
        ):
            """Deprecated: Use Cost.factory instead."""
            import warnings

            warnings.warn(
                "Cost.create_factory is deprecated, use Cost.factory instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return Cost.factory(  # type: ignore
                compute_residual,
                kind=kind,
                jac_mode=jac_mode,
                jac_batch_size=jac_batch_size,
                jac_custom_fn=jac_custom_fn,
                jac_custom_with_cache_fn=jac_custom_with_cache_fn,
                name=name,
            )

        @staticmethod
        def make[*Args_](
            compute_residual: jdc.Static[Callable[[VarValues, *Args_], jax.Array]],
            args: tuple[*Args_],
            jac_mode: jdc.Static[Literal["auto", "forward", "reverse"]] = "auto",
            jac_custom_fn: jdc.Static[
                Callable[[VarValues, *Args_], jax.Array] | None
            ] = None,
        ) -> Cost[*Args_]:
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
