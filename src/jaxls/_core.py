from __future__ import annotations

import dis
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Hashable,
    Iterable,
    Literal,
    Self,
    cast,
    overload,
)

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp
from jax.nn import relu
from jax.tree_util import default_registry
from loguru import logger
from typing_extensions import deprecated

from ._solvers import (
    ConjugateGradientConfig,
    NonlinearSolver,
    SolveSummary,
    TerminationConfig,
    TrustRegionConfig,
)

if TYPE_CHECKING:
    from ._augmented_lagrangian import AugmentedLagrangianConfig

from ._sparse_matrices import (
    BlockRowSparseMatrix,
    SparseBlockRow,
    SparseCooCoordinates,
    SparseCsrCoordinates,
)
from ._variables import Var, VarTypeOrdering, VarValues, sort_and_stack_vars


@jdc.pytree_dataclass
class _ResidualInfo:
    """Bundled residual computation results."""

    residual_vectors: tuple[jax.Array, ...]
    """Per-group residual vectors (matches _stacked_costs structure)."""

    residual_vector: jax.Array
    """Concatenated residual vector for linear algebra operations."""

    cost_total: jax.Array
    """Total cost (sum of all squared residuals)."""

    cost_nonconstraint: jax.Array
    """Cost from l2_squared terms only (original objective, excludes constraint terms)."""

    jac_cache: tuple[Any, ...]
    """Jacobian cache."""


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
    """Least squares problems are bipartite graphs with two types of nodes:

    - :class:`~jaxls.Cost`: cost terms or constraints.

        - ``kind="l2_squared"`` (default): Minimize squared L2 norm ``||r(x)||^2``
        - ``kind="constraint_eq_zero"``: Equality constraint ``r(x) = 0``
        - ``kind="constraint_leq_zero"``: Inequality constraint ``r(x) <= 0``
        - ``kind="constraint_geq_zero"``: Inequality constraint ``r(x) >= 0``

    - :class:`~jaxls.Var`: the parameters we want to optimize.
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

        def _deduplicate_compute_residual(cost: Cost) -> Cost:
            with jdc.copy_and_mutate(cost) as cost_copy:
                cost_copy.compute_residual = compute_residual_from_hash.setdefault(
                    _get_function_signature(cost.compute_residual),
                    cost.compute_residual,
                )
            return cost_copy

        costs = tuple(_deduplicate_compute_residual(cost) for cost in self.costs)

        # Count costs by mode.
        count_by_kind: dict[CostKind, int] = {
            "l2_squared": 0,
            "constraint_eq_zero": 0,
            "constraint_leq_zero": 0,
            "constraint_geq_zero": 0,
        }
        for f in costs:
            assert len(f._get_batch_axes()) in (0, 1)
            increment = 1 if len(f._get_batch_axes()) == 0 else f._get_batch_axes()[0]
            count_by_kind[f.kind] += increment

        num_variables = 0
        for v in variables:
            assert isinstance(v.id, int) or len(v.id.shape) in (0, 1)
            num_variables += (
                1 if isinstance(v.id, int) or v.id.shape == () else v.id.shape[0]
            )

        # Log counts by constraint type.
        total_costs = sum(count_by_kind.values())
        logger.info(
            "Building optimization problem with {} terms and {} variables: "
            "{} costs, {} eq_zero, {} leq_zero, {} geq_zero",
            total_costs,
            num_variables,
            count_by_kind["l2_squared"],
            count_by_kind["constraint_eq_zero"],
            count_by_kind["constraint_leq_zero"],
            count_by_kind["constraint_geq_zero"],
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

        # Group all costs by structure (both regular and constraint-mode).
        # This allows us to share the grouping/stacking logic.
        costs_from_group = dict[Any, list[Cost]]()
        count_from_group = dict[Any, int]()
        constraint_index_from_group = dict[Any, int]()
        constraint_index = 0

        for cost in costs:
            cost = cost._broadcast_batch_axes()
            batch_axes = cost._get_batch_axes()

            group_key: Hashable = (
                jax.tree.structure(cost),
                tuple(
                    leaf.shape[len(batch_axes) :] if hasattr(leaf, "shape") else ()
                    for leaf in jax.tree.leaves(cost)
                ),
            )

            # Initialize group if new.
            if group_key not in costs_from_group:
                costs_from_group[group_key] = []
                count_from_group[group_key] = 0
                # Assign constraint_index for constraint-mode groups.
                if cost.kind != "l2_squared":
                    constraint_index_from_group[group_key] = constraint_index
                    constraint_index += 1

            if len(batch_axes) == 0:
                cost = jax.tree.map(lambda x: jnp.asarray(x)[None], cost)
                count_from_group[group_key] += 1
            else:
                assert len(batch_axes) == 1
                count_from_group[group_key] += batch_axes[0]

            costs_from_group[group_key].append(cost)

        # Fields we want to populate.
        stacked_costs = list[_AnalyzedCost]()
        cost_counts = list[int]()
        jac_coords = list[tuple[jax.Array, jax.Array]]()

        # Sort variable IDs.
        sorted_ids_from_var_type = sort_and_stack_vars(variables)
        del variables

        # Process all cost groups with unified logic.
        residual_dim_sum = 0
        for group_key in sorted(costs_from_group.keys(), key=_sort_key):
            group = costs_from_group[group_key]
            count = count_from_group[group_key]

            # Stack costs within this group.
            stacked_cost: Cost = jax.tree.map(
                lambda *args: jnp.concatenate(args, axis=0), *group
            )

            # Convert to _AnalyzedCost.
            # For constraint-mode costs, use _augment_constraint_cost with constraint_index.
            # For regular costs, use _AnalyzedCost._make.
            is_constraint_group = group_key in constraint_index_from_group
            if is_constraint_group:
                group_constraint_index = constraint_index_from_group[group_key]
                stacked_cost_expanded: _AnalyzedCost = jax.vmap(
                    lambda c: _augment_constraint_cost(c, group_constraint_index)
                )(stacked_cost)
            else:
                stacked_cost_expanded: _AnalyzedCost = jax.vmap(_AnalyzedCost._make)(
                    stacked_cost
                )

            stacked_costs.append(stacked_cost_expanded)
            cost_counts.append(count)

            # Log group info.
            if is_constraint_group:
                logger.info(
                    "Vectorizing constraint group with {} constraints ({}), {} variables each: {}",
                    count,
                    stacked_cost_expanded.kind,
                    stacked_cost_expanded.num_variables,
                    stacked_cost_expanded._get_name(),
                )
            else:
                logger.info(
                    "Vectorizing group with {} costs, {} variables each: {}",
                    count,
                    stacked_cost_expanded.num_variables,
                    stacked_cost_expanded._get_name(),
                )

            # Compute Jacobian coordinates.
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
                    count,
                    stacked_cost_expanded.residual_flat_dim,
                    rows.shape[-1],
                )
            )
            rows = rows + (
                jnp.arange(count)[:, None, None]
                * stacked_cost_expanded.residual_flat_dim
            )
            rows = rows + residual_dim_sum
            jac_coords.append((rows.flatten(), cols.flatten()))
            residual_dim_sum += stacked_cost_expanded.residual_flat_dim * count

        # Build sparse coordinates.
        jac_coords_coo = SparseCooCoordinates(
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
            _stacked_costs=tuple(stacked_costs),
            _cost_counts=tuple(cost_counts),
            _sorted_ids_from_var_type=sorted_ids_from_var_type,
            _jac_coords_coo=jac_coords_coo,
            _jac_coords_csr=jac_coords_csr,
            _tangent_ordering=tangent_ordering,
            _tangent_start_from_var_type=tangent_start_from_var_type,
            _tangent_dim=tangent_dim_sum,
            _residual_dim=residual_dim_sum,
        )


@jdc.pytree_dataclass
class AnalyzedLeastSquaresProblem:
    _stacked_costs: tuple[_AnalyzedCost, ...]
    """All costs including constraint-mode costs. Constraint-mode costs have
    constraint is not None and args=(AugmentedLagrangianParams,)."""
    _cost_counts: jdc.Static[tuple[int, ...]]
    _sorted_ids_from_var_type: dict[type[Var], jax.Array]
    _jac_coords_coo: SparseCooCoordinates
    _jac_coords_csr: SparseCsrCoordinates
    _tangent_ordering: jdc.Static[VarTypeOrdering]
    _tangent_start_from_var_type: jdc.Static[dict[type[Var[Any]], int]]
    _tangent_dim: jdc.Static[int]
    _residual_dim: jdc.Static[int]

    @overload
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
        augmented_lagrangian: AugmentedLagrangianConfig | None = None,
        return_summary: Literal[False] = False,
    ) -> VarValues: ...

    @overload
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
        augmented_lagrangian: AugmentedLagrangianConfig | None = None,
        return_summary: Literal[True],
    ) -> tuple[VarValues, SolveSummary]: ...

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
        augmented_lagrangian: AugmentedLagrangianConfig | None = None,
        return_summary: bool = False,
    ) -> VarValues | tuple[VarValues, SolveSummary]:
        """Solve the nonlinear least squares problem using either Gauss-Newton
        or Levenberg-Marquardt.

        For constrained problems (with equality constraints), the Augmented
        Lagrangian method will be automatically used.

        Args:
            initial_vals: Initial values for the variables. If None, default values will be used.
            linear_solver: The linear solver to use.
            trust_region: Configuration for Levenberg-Marquardt trust region.
            termination: Configuration for termination criteria.
            sparse_mode: The representation to use for sparse matrix
                multiplication. Can be "blockrow", "coo", or "csr".
            verbose: Whether to print verbose output during optimization.
            augmented_lagrangian: Configuration for Augmented Lagrangian method.
                Only used if constraints are present. If None and constraints
                exist, a default configuration will be used.
            return_summary: If `True`, return a summary of the solve.

        Returns:
            Optimized variable values.
        """
        if initial_vals is None:
            initial_vals = VarValues.make(
                var_type(ids)
                for var_type, ids in self._sorted_ids_from_var_type.items()
            )

        # Check if we have constraints (costs with constraint is not None).
        has_constraints = any(cost.kind != "l2_squared" for cost in self._stacked_costs)

        # Use default AL config if constraints present and none specified.
        if has_constraints and augmented_lagrangian is None:
            from ._augmented_lagrangian import AugmentedLagrangianConfig

            augmented_lagrangian = AugmentedLagrangianConfig()

        # In our internal API, linear_solver needs to always be a string. The
        # conjugate gradient config is a separate field. This is more
        # convenient to implement, because then the former can be static while
        # the latter is a pytree.
        conjugate_gradient_config = None
        if isinstance(linear_solver, ConjugateGradientConfig):
            conjugate_gradient_config = linear_solver
            linear_solver = "conjugate_gradient"

        # Create unified solver (handles both constrained and unconstrained).
        solver = NonlinearSolver(
            linear_solver,
            trust_region,
            termination,
            conjugate_gradient_config,
            sparse_mode,
            verbose,
            augmented_lagrangian if has_constraints else None,
        )
        return solver.solve(
            problem=self, initial_vals=initial_vals, return_summary=return_summary
        )

    def compute_residual_vector(self, vals: VarValues) -> jax.Array:
        """Compute the residual vector. The cost we are optimizing is defined
        as the sum of squared terms within this vector."""
        residual_slices = list[jax.Array]()
        jac_cache = list[CustomJacobianCache]()
        for stacked_cost in self._stacked_costs:
            # Vmap over the entire cost object (including al_params if present).
            # This ensures consistency with Jacobian computation.
            compute_residual_out = jax.vmap(
                lambda cost: cost.compute_residual_flat(vals, *cost.args)
            )(stacked_cost)

            if isinstance(compute_residual_out, tuple):
                assert len(compute_residual_out) == 2
                residual_slices.append(compute_residual_out[0].reshape((-1,)))
                jac_cache.append(compute_residual_out[1])
            else:
                assert len(compute_residual_out.shape) == 2
                residual_slices.append(compute_residual_out.reshape((-1,)))
                jac_cache.append(None)
        return jnp.concatenate(residual_slices, axis=0)

    def _compute_residual_info(self, vals: VarValues) -> _ResidualInfo:
        """Compute residuals, costs, and Jacobian cache in one pass.

        This bundles all residual-related computation into a single call,
        computing per-group residuals, concatenated residual, total cost,
        and non-constraint cost (original objective) together.

        Args:
            vals: Variable values to evaluate at.
            include_jac_cache: Whether to compute Jacobian cache.

        Returns:
            _ResidualInfo with all computed values.
        """
        residual_vectors: list[jax.Array] = []
        jac_caches: list[Any] = []
        cost_nonconstraint = jnp.array(0.0)

        for stacked_cost in self._stacked_costs:
            # Vmap over the entire cost object (including al_params if present).
            compute_residual_out = jax.vmap(
                lambda cost: cost.compute_residual_flat(vals, *cost.args)
            )(stacked_cost)

            if isinstance(compute_residual_out, tuple):
                assert len(compute_residual_out) == 2
                residual = compute_residual_out[0].reshape((-1,))
                jac_caches.append(compute_residual_out[1])
            else:
                assert len(compute_residual_out.shape) == 2
                residual = compute_residual_out.reshape((-1,))
                jac_caches.append(None)

            residual_vectors.append(residual)

            # Accumulate cost from l2_squared terms only.
            if stacked_cost.kind == "l2_squared":
                cost_nonconstraint = cost_nonconstraint + jnp.sum(residual**2)

        residual_vector = jnp.concatenate(residual_vectors, axis=0)
        cost_total = jnp.sum(residual_vector**2)

        return _ResidualInfo(
            residual_vectors=tuple(residual_vectors),
            residual_vector=residual_vector,
            cost_total=cost_total,
            cost_nonconstraint=cost_nonconstraint,
            jac_cache=tuple(jac_caches),
        )

    def _compute_constraint_values(self, vals: VarValues) -> tuple[jax.Array, ...]:
        """Compute original constraint values (not augmented), one array per constraint group.

        For constraint-mode costs, this uses compute_residual_original to get
        the raw constraint values (before augmented Lagrangian transformation).

        Returns:
            Tuple of arrays, one per constraint group. Each has shape
            (constraint_count, constraint_flat_dim).
        """
        constraint_slices = list[jax.Array]()
        for stacked_cost in self._stacked_costs:
            if stacked_cost.kind == "l2_squared":
                continue  # Skip regular costs

            # Use compute_residual_original to get raw constraint values.
            assert stacked_cost.compute_residual_original is not None
            constraint_vals = jax.vmap(
                lambda c: c.compute_residual_original(vals, *c.args)
            )(stacked_cost)
            # Keep 2D shape (constraint_count, constraint_flat_dim) for AL state.
            constraint_slices.append(constraint_vals)

        return tuple(constraint_slices)

    def compute_constraint_values(self, vals: VarValues) -> jax.Array:
        """Compute all constraint values as a flat array.

        For equality constraints, these should all be zero at a feasible solution.
        For inequality constraints (g(x) <= 0), these should be <= 0.
        """
        constraint_slices = self._compute_constraint_values(vals)
        if len(constraint_slices) == 0:
            return jnp.array([])
        # Flatten each 2D slice before concatenating.
        return jnp.concatenate([c.reshape(-1) for c in constraint_slices], axis=0)

    def _compute_jac_values(
        self, vals: VarValues, jac_cache: tuple[CustomJacobianCache, ...]
    ) -> BlockRowSparseMatrix:
        block_rows = list[SparseBlockRow]()
        residual_offset = 0

        for i, cost in enumerate(self._stacked_costs):
            # Shape should be: (count_from_group[group_key], single_residual_dim, sum_of_tangent_dims_of_variables).
            def compute_jac_with_perturb(
                cost: _AnalyzedCost, jac_cache_i: CustomJacobianCache | None = None
            ) -> jax.Array:
                val_subset = vals._get_subset(
                    {
                        var_type: jnp.searchsorted(vals.ids_from_type[var_type], ids)
                        for var_type, ids in cost.sorted_ids_from_var_type.items()
                    },
                    self._tangent_ordering,
                )

                # Shape should be: (residual_dim, sum_of_tangent_dims_of_variables).
                if cost.jac_custom_fn is not None:
                    assert jac_cache_i is None, (
                        "`jac_custom_with_cache_fn` should be used if a Jacobian cache is used, not `jac_custom_fn`!"
                    )
                    return cost.jac_custom_fn(vals, *cost.args)
                if cost.jac_custom_with_cache_fn is not None:
                    assert jac_cache_i is not None, (
                        "`jac_custom_with_cache_fn` was specified, but no cache was returned by `compute_residual`!"
                    )
                    return cost.jac_custom_with_cache_fn(vals, jac_cache_i, *cost.args)

                jacfunc = {
                    "forward": jax.jacfwd,
                    "reverse": jax.jacrev,
                    "auto": jax.jacrev
                    if cost.residual_flat_dim < val_subset._get_tangent_dim()
                    else jax.jacfwd,
                }[cost.jac_mode]

                # compute_residual_flat handles al_params internally via self.al_params
                return jacfunc(
                    lambda tangent: cost.compute_residual_flat(
                        val_subset._retract(tangent, self._tangent_ordering),
                        *cost.args,
                    )
                )(jnp.zeros((val_subset._get_tangent_dim(),)))

            optional_jac_cache_i = (jac_cache[i],) if jac_cache[i] is not None else ()

            # Compute Jacobian for each cost term.
            if cost.jac_batch_size is None:
                stacked_jac = jax.vmap(compute_jac_with_perturb)(
                    cost, *optional_jac_cache_i
                )
            else:
                # When `batch_size` is `None`, jax.lax.map reduces to a scan
                # (similar to `batch_size=1`).
                stacked_jac = jax.lax.map(
                    compute_jac_with_perturb,
                    cost,
                    *optional_jac_cache_i,
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
            for var_type, ids in self._tangent_ordering.ordered_dict_items(
                # This ordering shouldn't actually matter!
                cost.sorted_ids_from_var_type
            ):
                (num_costs_, num_vars) = ids.shape
                assert num_costs == num_costs_

                # Get one block for each variable.
                for var_idx in range(ids.shape[-1]):
                    start_cols.append(
                        jnp.searchsorted(
                            self._sorted_ids_from_var_type[var_type], ids[..., var_idx]
                        )
                        * var_type.tangent_dim
                        + self._tangent_start_from_var_type[var_type]
                    )
                    block_widths.append(var_type.tangent_dim)
                    assert start_cols[-1].shape == (num_costs_,)

                stacked_jac_start_col = (
                    stacked_jac_start_col + num_vars * var_type.tangent_dim
                )
            assert stacked_jac.shape[-1] == stacked_jac_start_col

            block_rows.append(
                SparseBlockRow(
                    num_cols=self._tangent_dim,
                    start_cols=tuple(start_cols),
                    block_num_cols=tuple(block_widths),
                    blocks_concat=stacked_jac,
                )
            )

            residual_offset += cost.residual_flat_dim * num_costs
        assert residual_offset == self._residual_dim

        bsparse_jacobian = BlockRowSparseMatrix(
            block_rows=tuple(block_rows),
            shape=(self._residual_dim, self._tangent_dim),
        )
        return bsparse_jacobian


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
        """Extract all Var objects from args (walks the pytree)."""

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

    if not TYPE_CHECKING:

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


@jdc.pytree_dataclass(kw_only=True)
class _AnalyzedCost[*Args](Cost[*Args]):
    """Analyzed cost ready for optimization.

    Used for both regular costs and constraint-mode costs internally.
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
        """Compute augmented constraint residual with per-constraint penalty."""
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
                """Wrapper using cached active mask."""
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
            """Wrapper Jacobian with cache that applies chain rule."""
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
        """Compute original constraint value (not augmented)."""
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
