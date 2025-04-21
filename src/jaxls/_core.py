from __future__ import annotations

import dis
import functools
from typing import (
    Any,
    Callable,
    Concatenate,
    Hashable,
    Iterable,
    Literal,
    cast,
    overload,
)

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp
from jax.tree_util import default_registry
from loguru import logger
from typing_extensions import deprecated

from ._solvers import (
    ConjugateGradientConfig,
    NonlinearSolver,
    TerminationConfig,
    TrustRegionConfig,
)
from ._sparse_matrices import (
    BlockRowSparseMatrix,
    SparseBlockRow,
    SparseCooCoordinates,
    SparseCsrCoordinates,
)
from ._variables import Var, VarTypeOrdering, VarValues, sort_and_stack_vars


def _get_function_signature(func: Callable) -> Hashable:
    """Returns a hashable value, which should be equal for equivalent input functions."""
    closure = getattr(func, "__closure__", None)
    if closure is not None:
        closure_vars = tuple(sorted((str(cell.cell_contents) for cell in closure)))
    else:
        closure_vars = ()

    instance = getattr(func, "__self__", None)
    if instance is not None:
        instance_id = id(instance)
    else:
        instance_id = None

    bytecode = dis.Bytecode(func)
    bytecode_tuple = tuple((instr.opname, instr.argrepr) for instr in bytecode)
    return bytecode_tuple, closure_vars, instance_id


@jdc.pytree_dataclass
class LeastSquaresProblem:
    """We define least squares problem as bipartite graphs, which have two types of nodes:

    - `jaxls.Cost`. These are the terms we want to minimize.
    - `jaxls.Var`. These are the parameters we want to optimize.
    """

    costs: Iterable[Cost]
    variables: Iterable[Var]

    def analyze(self, use_onp: bool = False) -> AnalyzedLeastSquaresProblem:
        """Analyze sparsity pattern of least squares problem. Needed before solving."""

        # Operations using vanilla numpy can be faster.
        if use_onp:
            jnp = onp
        else:
            from jax import numpy as jnp

        variables = tuple(self.variables)
        compute_residual_from_hash = dict[Hashable, Callable]()
        costs = tuple(
            jdc.replace(
                cost,
                compute_residual=compute_residual_from_hash.setdefault(
                    _get_function_signature(cost.compute_residual),
                    cost.compute_residual,
                ),
            )
            for cost in self.costs
        )

        # We're assuming no more than 1 batch axis.
        num_costs = 0
        for f in costs:
            assert len(f._get_batch_axes()) in (0, 1)
            num_costs += 1 if len(f._get_batch_axes()) == 0 else f._get_batch_axes()[0]

        num_variables = 0
        for v in variables:
            assert isinstance(v.id, int) or len(v.id.shape) in (0, 1)
            num_variables += (
                1 if isinstance(v.id, int) or v.id.shape == () else v.id.shape[0]
            )
        logger.info(
            "Building optimization problem with {} costs and {} variables.",
            num_costs,
            num_variables,
        )

        # Create storage layout: this describes which parts of our tangent
        # vector is allocated to each variable.
        tangent_start_from_var_type = dict[type[Var[Any]], int]()

        def _sort_key(x: Any) -> str:
            """We're going to sort variable / cost types by name. This should
            prevent re-compiling when costs or variables are reordered."""
            return str(x)

        # Count variables of each type.
        count_from_var_type = dict[type[Var[Any]], int]()
        for var in variables:
            if isinstance(var.id, int) or var.id.shape == ():
                increment = 1
            else:
                (increment,) = var.id.shape
            count_from_var_type[type(var)] = (
                count_from_var_type.get(type(var), 0) + increment
            )
        tangent_dim_sum = 0
        for var_type in sorted(count_from_var_type.keys(), key=_sort_key):
            tangent_start_from_var_type[var_type] = tangent_dim_sum
            tangent_dim_sum += var_type.tangent_dim * count_from_var_type[var_type]

        # Create ordering helper.
        tangent_ordering = VarTypeOrdering(
            {
                var_type: i
                for i, var_type in enumerate(tangent_start_from_var_type.keys())
            }
        )

        # Start by grouping our costs and grabbing a list of (ordered!) variables
        costs_from_group = dict[Any, list[Cost]]()
        count_from_group = dict[Any, int]()
        for cost in costs:
            # Support broadcasting for cost arguments.
            cost = cost._broadcast_batch_axes()

            # Each cost is ultimately just a pytree node; in order for a set of
            # costs to be batchable, they must share the same:
            group_key: Hashable = (
                # (1) Treedef. Structure of inputs must match.
                jax.tree.structure(cost),
                # (2) Leaf shapes: contained array shapes must match
                tuple(
                    leaf.shape if hasattr(leaf, "shape") else ()
                    for leaf in jax.tree.leaves(cost)
                ),
            )
            costs_from_group.setdefault(group_key, [])
            count_from_group.setdefault(group_key, 0)

            batch_axes = cost._get_batch_axes()
            if len(batch_axes) == 0:
                cost = jax.tree.map(lambda x: jnp.asarray(x)[None], cost)
                count_from_group[group_key] += 1
            else:
                assert len(batch_axes) == 1
                count_from_group[group_key] += batch_axes[0]

            # Record costs and variables.
            costs_from_group[group_key].append(cost)

        # Fields we want to populate.
        stacked_costs = list[_AnalyzedCost]()
        cost_counts = list[int]()
        jac_coords = list[tuple[jax.Array, jax.Array]]()

        # Sort variable IDs.
        sorted_ids_from_var_type = sort_and_stack_vars(variables)
        del variables

        # Prepare each cost group. We put groups in a consistent order.
        residual_dim_sum = 0
        for group_key in sorted(costs_from_group.keys(), key=_sort_key):
            group = costs_from_group[group_key]

            # Stack cost parameters.
            stacked_cost: Cost = jax.tree.map(
                lambda *args: jnp.concatenate(args, axis=0), *group
            )
            stacked_cost_expanded: _AnalyzedCost = jax.vmap(_AnalyzedCost._make)(
                stacked_cost
            )
            stacked_costs.append(stacked_cost_expanded)
            cost_counts.append(count_from_group[group_key])

            logger.info(
                "Vectorizing group with {} costs, {} variables each: {}",
                count_from_group[group_key],
                stacked_costs[-1].num_variables,
                stacked_costs[-1]._get_name(),
            )

            # Compute Jacobian coordinates.
            #
            # These should be N pairs of (row, col) indices, where rows correspond to
            # residual indices and columns correspond to tangent vector indices.
            rows, cols = jax.vmap(
                functools.partial(
                    _AnalyzedCost._compute_block_sparse_jac_indices,
                    tangent_ordering=tangent_ordering,
                    sorted_ids_from_var_type=sorted_ids_from_var_type,
                    tangent_start_from_var_type=tangent_start_from_var_type,
                )
            )(stacked_cost_expanded)
            assert (
                rows.shape
                == cols.shape
                == (
                    count_from_group[group_key],
                    stacked_cost_expanded.residual_flat_dim,
                    rows.shape[-1],
                )
            )
            rows = rows + (
                jnp.arange(count_from_group[group_key])[:, None, None]
                * stacked_cost_expanded.residual_flat_dim
            )
            rows = rows + residual_dim_sum
            jac_coords.append((rows.flatten(), cols.flatten()))
            residual_dim_sum += (
                stacked_cost_expanded.residual_flat_dim * count_from_group[group_key]
            )

        jac_coords_coo: SparseCooCoordinates = SparseCooCoordinates(
            *jax.tree.map(lambda *arrays: jnp.concatenate(arrays, axis=0), *jac_coords),
            shape=(residual_dim_sum, tangent_dim_sum),
        )
        csr_indptr = jnp.searchsorted(
            jac_coords_coo.rows, jnp.arange(residual_dim_sum + 1)
        )
        jac_coords_csr = SparseCsrCoordinates(
            indices=jac_coords_coo.cols,
            indptr=cast(jax.Array, csr_indptr),
            shape=(residual_dim_sum, tangent_dim_sum),
        )
        return AnalyzedLeastSquaresProblem(
            stacked_costs=tuple(stacked_costs),
            cost_counts=tuple(cost_counts),
            sorted_ids_from_var_type=sorted_ids_from_var_type,
            jac_coords_coo=jac_coords_coo,
            jac_coords_csr=jac_coords_csr,
            tangent_ordering=tangent_ordering,
            tangent_start_from_var_type=tangent_start_from_var_type,
            tangent_dim=tangent_dim_sum,
            residual_dim=residual_dim_sum,
        )


@jdc.pytree_dataclass
class AnalyzedLeastSquaresProblem:
    stacked_costs: tuple[_AnalyzedCost, ...]
    cost_counts: jdc.Static[tuple[int, ...]]
    sorted_ids_from_var_type: dict[type[Var], jax.Array]
    jac_coords_coo: SparseCooCoordinates
    jac_coords_csr: SparseCsrCoordinates
    tangent_ordering: jdc.Static[VarTypeOrdering]
    tangent_start_from_var_type: jdc.Static[dict[type[Var[Any]], int]]
    tangent_dim: jdc.Static[int]
    residual_dim: jdc.Static[int]

    def solve(
        self,
        initial_vals: VarValues | None = None,
        *,
        linear_solver: Literal["conjugate_gradient", "cholmod", "dense_cholesky"]
        | ConjugateGradientConfig = "conjugate_gradient",
        trust_region: TrustRegionConfig | None = TrustRegionConfig(),
        termination: TerminationConfig = TerminationConfig(),
        sparse_mode: Literal["blockrow", "coo", "csr"] = "blockrow",
        verbose: bool = True,
    ) -> VarValues:
        """Solve the nonlinear least squares problem using either Gauss-Newton
        or Levenberg-Marquardt.

        Args:
            initial_vals: Initial values for the variables. If None, default values will be used.
            linear_solver: The linear solver to use.
            trust_region: Configuration for Levenberg-Marquardt trust region.
            termination: Configuration for termination criteria.
            sparse_mode: The representation to use for sparse matrix
                multiplication. Can be "blockrow", "coo", or "csr".
            verbose: Whether to print verbose output during optimization.

        Returns:
            Optimized variable values.
        """
        if initial_vals is None:
            initial_vals = VarValues.make(
                var_type(ids) for var_type, ids in self.sorted_ids_from_var_type.items()
            )

        # In our internal API, linear_solver needs to always be a string. The
        # conjugate gradient config is a separate field. This is more
        # convenient to implement, because then the former can be static while
        # the latter is a pytree.
        conjugate_gradient_config = None
        if isinstance(linear_solver, ConjugateGradientConfig):
            conjugate_gradient_config = linear_solver
            linear_solver = "conjugate_gradient"

        solver = NonlinearSolver(
            linear_solver,
            trust_region,
            termination,
            conjugate_gradient_config,
            sparse_mode,
            verbose,
        )
        return solver.solve(problem=self, initial_vals=initial_vals)

    def compute_residual_vector(self, vals: VarValues) -> jax.Array:
        """Compute the residual vector. The cost we are optimizing is defined
        as the sum of squared terms within this vector."""
        residual_slices = list[jax.Array]()
        for stacked_cost in self.stacked_costs:
            # Flatten the output of the user-provided compute_residual.
            stacked_residual_slice = jax.vmap(
                lambda args: stacked_cost.compute_residual_flat(vals, *args)
            )(stacked_cost.args)
            assert len(stacked_residual_slice.shape) == 2
            residual_slices.append(stacked_residual_slice.reshape((-1,)))
        return jnp.concatenate(residual_slices, axis=0)

    def _compute_jac_values(self, vals: VarValues) -> BlockRowSparseMatrix:
        block_rows = list[SparseBlockRow]()
        residual_offset = 0

        for cost in self.stacked_costs:
            # Shape should be: (count_from_group[group_key], single_residual_dim, sum_of_tangent_dims_of_variables).
            def compute_jac_with_perturb(cost: _AnalyzedCost) -> jax.Array:
                val_subset = vals._get_subset(
                    {
                        var_type: jnp.searchsorted(vals.ids_from_type[var_type], ids)
                        for var_type, ids in cost.sorted_ids_from_var_type.items()
                    },
                    self.tangent_ordering,
                )

                # Shape should be: (residual_dim, sum_of_tangent_dims_of_variables).
                if cost.jac_custom_fn is not None:
                    return cost.jac_custom_fn(vals, *cost.args)

                jacfunc = {
                    "forward": jax.jacfwd,
                    "reverse": jax.jacrev,
                    "auto": jax.jacrev
                    if cost.residual_flat_dim < val_subset._get_tangent_dim()
                    else jax.jacfwd,
                }[cost.jac_mode]
                return jacfunc(
                    # We flatten the output of compute_residual before
                    # computing Jacobian. The Jacobian is computed with respect
                    # to the flattened residual.
                    lambda tangent: cost.compute_residual_flat(
                        val_subset._retract(tangent, self.tangent_ordering),
                        *cost.args,
                    )
                )(jnp.zeros((val_subset._get_tangent_dim(),)))

            # Compute Jacobian for each cost term.
            if cost.jac_batch_size is None:
                stacked_jac = jax.vmap(compute_jac_with_perturb)(cost)
            else:
                # When `batch_size` is `None`, jax.lax.map reduces to a scan
                # (similar to `batch_size=1`).
                stacked_jac = jax.lax.map(
                    compute_jac_with_perturb, cost, batch_size=cost.jac_batch_size
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
            for var_type, ids in self.tangent_ordering.ordered_dict_items(
                # This ordering shouldn't actually matter!
                cost.sorted_ids_from_var_type
            ):
                (num_costs_, num_vars) = ids.shape
                assert num_costs == num_costs_

                # Get one block for each variable.
                for var_idx in range(ids.shape[-1]):
                    start_cols.append(
                        jnp.searchsorted(
                            self.sorted_ids_from_var_type[var_type], ids[..., var_idx]
                        )
                        * var_type.tangent_dim
                        + self.tangent_start_from_var_type[var_type]
                    )
                    block_widths.append(var_type.tangent_dim)
                    assert start_cols[-1].shape == (num_costs_,)

                stacked_jac_start_col = (
                    stacked_jac_start_col + num_vars * var_type.tangent_dim
                )
            assert stacked_jac.shape[-1] == stacked_jac_start_col

            block_rows.append(
                SparseBlockRow(
                    num_cols=self.tangent_dim,
                    start_cols=tuple(start_cols),
                    block_num_cols=tuple(block_widths),
                    blocks_concat=stacked_jac,
                )
            )

            residual_offset += cost.residual_flat_dim * num_costs
        assert residual_offset == self.residual_dim

        bsparse_jacobian = BlockRowSparseMatrix(
            block_rows=tuple(block_rows),
            shape=(self.residual_dim, self.tangent_dim),
        )
        return bsparse_jacobian


type ResidualFunc[**Args] = Callable[Concatenate[VarValues, Args], jax.Array]
type JacobianFunc[**Args] = Callable[Concatenate[VarValues, Args], jax.Array]
type CostFactory[**Args] = Callable[Args, Cost[tuple[Any, ...], dict[str, Any]]]


@jdc.pytree_dataclass
class Cost[*Args]:
    """A least squares cost term in our optimization problem, defined by a
    residual function. The cost that this represents is defined as:

         || compute_residual(values, *args) ||_2^2

    where `values` is a `jaxls.VarValues` object.

    Each `Cost.compute_residual` should take at least one argument that inherits
    from the symbolic variable `jaxls.Var(id)`, where `id` must be a scalar
    integer.

    To create a batch of costs, a leading batch axis can be added to the
    arguments passed to `Cost.args`:
    - The batch axis must be the same for all arguments.
    -
    """

    compute_residual: jdc.Static[Callable[[VarValues, *Args], jax.Array]]
    args: tuple[*Args]
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

    name: jdc.Static[str | None] = None
    """Custom name for the cost. This is used for debugging and logging."""

    def _get_name(self) -> str:
        """Get the name of the cost. If not set, we use
        `cost.compute_residual.__name__`."""
        if self.name is None:
            return self.compute_residual.__name__
        return self.name

    # Simple decorator.
    @overload
    @staticmethod
    def create_factory[**Args_](
        compute_residual: ResidualFunc[Args_],
    ) -> CostFactory[Args_]: ...

    # Decorator factory with keyword arguments.
    @overload
    @staticmethod
    def create_factory[**Args_](
        *,
        jac_mode: jdc.Static[Literal["auto", "forward", "reverse"]] = "auto",
        jac_batch_size: jdc.Static[int | None] = None,
        jac_custom_fn: jdc.Static[JacobianFunc | None] = None,
        name: jdc.Static[str | None] = None,
    ) -> Callable[[ResidualFunc[Args_]], CostFactory[Args_]]: ...

    @staticmethod
    def create_factory[**Args_](
        compute_residual: ResidualFunc[Args_] | None = None,
        *,
        jac_mode: jdc.Static[Literal["auto", "forward", "reverse"]] = "auto",
        jac_batch_size: jdc.Static[int | None] = None,
        jac_custom_fn: jdc.Static[JacobianFunc | None] = None,
        name: jdc.Static[str | None] = None,
    ) -> Callable[[ResidualFunc[Args_]], CostFactory[Args_]] | CostFactory[Args_]:
        """Decorator for creating costs from a residual function.

        Examples:

            @jaxls.Cost.create_factory
            def cost1(values: VarValues, var1: SE2Var, var2: int) -> jax.Array:
                ...

            # Factory will have the same input signature as the wrapped
            # residual function, but without the `VarValues` argument. The
            # return type will be `Factor` instead of `jax.Array`.
            cost = cost1(var1=SE2Var(0), var2=5)
            assert isinstance(cost, jaxls.Cost)

        Keyword arguments can also be used for configuration. For example:

            # To enforce forward-mode autodiff for Jacobians.
            @Factor.create_factory(jac_mode="forward")
            def cost(...): ...

            # To reduce memory usage.
            @Factor.create_factory(jac_batch_size=1)
            def cost(...): ...

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
                    jac_mode=jac_mode,
                    jac_batch_size=jac_batch_size,
                    jac_custom_fn=(
                        lambda values, args, kwargs: cast(
                            JacobianFunc[Args_], jac_custom_fn
                        )(values, *args, **kwargs)
                    )
                    if jac_custom_fn is not None
                    else None,
                    name=name,
                )

            return inner

        if compute_residual is None:
            return decorator
        return decorator(compute_residual)

    @staticmethod
    @deprecated("Use Factor() directly instead of Factor.make()")
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
            "Use Factor() directly instead of Factor.make()", DeprecationWarning
        )
        return Cost(compute_residual, args, jac_mode, None, jac_custom_fn)

    def _get_variables(self) -> tuple[Var, ...]:
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
        variables = self._get_variables()
        assert len(variables) != 0, "No variables found in cost!"
        return jnp.broadcast_shapes(
            *[() if isinstance(v.id, int) else v.id.shape for v in variables]
        )

    def _broadcast_batch_axes(self) -> Cost:
        batch_axes = self._get_batch_axes()
        if batch_axes is None:
            return self
        leaves, treedef = jax.tree.flatten(self)
        broadcasted_leaves = []
        for leaf in leaves:
            if isinstance(leaf, (int, float)):
                leaf = jnp.array(leaf)
            broadcasted_leaves.append(
                jnp.broadcast_to(leaf, batch_axes + leaf.shape[len(batch_axes) :])
            )
        return jax.tree.unflatten(treedef, broadcasted_leaves)


@jdc.pytree_dataclass(kw_only=True)
class _AnalyzedCost[*Args](Cost[*Args]):
    """Same as `Factor`, but with extra fields."""

    num_variables: jdc.Static[int]
    sorted_ids_from_var_type: dict[type[Var[Any]], jax.Array]
    residual_flat_dim: jdc.Static[int] = 0

    def compute_residual_flat(self, vals: VarValues, *args: *Args) -> jax.Array:
        return self.compute_residual(vals, *args).flatten()

    @staticmethod
    @jdc.jit
    def _make[*Args_](cost: Cost[*Args_]) -> _AnalyzedCost[*Args_]:
        """Construct a cost for our optimization problem."""
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

        # Cache the residual dimension for this cost.
        dummy_vals = jax.eval_shape(VarValues.make, variables)
        residual_dim = onp.prod(
            jax.eval_shape(cost.compute_residual, dummy_vals, *cost.args).shape
        )

        return _AnalyzedCost(
            **vars(cost),
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
        """Compute row and column indices for block-sparse Jacobian of shape
        (residual dim, total tangent dim). Residual indices will start at row=0."""
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
