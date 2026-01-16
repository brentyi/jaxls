from __future__ import annotations

import dis
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
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
from loguru import logger

from ._analyzed_cost import _AnalyzedCost, _augment_constraint_cost
from ._cost import Cost, CostKind, CustomJacobianCache
from ._solvers import (
    ConjugateGradientConfig,
    NonlinearSolver,
    SolveSummary,
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

if TYPE_CHECKING:
    from ._augmented_lagrangian import AugmentedLagrangianConfig
    from ._covariance import CovarianceEstimator, LinearSolverCovarianceEstimatorConfig


@jdc.pytree_dataclass
class _CostInfo:
    """Bundled cost computation results.

    This dataclass groups together all the outputs from evaluating cost terms
    at a given set of variable values, avoiding redundant computation.
    """

    residual_vectors: tuple[jax.Array, ...]
    """Per-group residual vectors (matches _stacked_costs structure)."""

    residual_vector: jax.Array
    """Concatenated residual vector for linear algebra operations."""

    cost_total: jax.Array
    """Total cost (sum of all squared residuals)."""

    cost_nonconstraint: jax.Array
    """Cost from l2_squared terms only (original objective, excludes constraint terms)."""

    jac_cache: tuple[Any, ...]
    """Jacobian cache from residual computation, used for custom Jacobian functions."""


def _get_function_signature(func: Callable) -> Hashable:
    """Returns a hashable value that should be equal for equivalent input functions.

    Used to deduplicate compute_residual functions that have identical bytecode
    and closure, avoiding redundant JIT compilation.

    Args:
        func: The function to compute a signature for.

    Returns:
        A hashable tuple of (bytecode, closure_vars, instance_id) that uniquely
        identifies the function's behavior.
    """
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

    def show(
        self,
        *,
        width: int = 800,
        height: int = 500,
        max_costs: int = 1000,
        max_variables: int = 500,
    ) -> None:
        """Display an interactive graph showing costs and variables.

        In Jupyter/JupyterLab/VS Code notebooks, displays inline. Otherwise,
        opens in the default web browser.

        Args:
            width: Maximum width of the visualization in pixels.
            height: Height of the visualization in pixels.
            max_costs: Maximum number of cost nodes to show. When multiple cost
                types exist, the limit is distributed proportionally across types.
            max_variables: Maximum number of variables per type to show.
                Only costs where all variables are visible are shown.
        """
        from ._visualization import problem_show

        problem_show(
            self,
            width=width,
            height=height,
            max_costs=max_costs,
            max_variables=max_variables,
        )

    def analyze(self, use_onp: bool = False) -> AnalyzedLeastSquaresProblem:
        """Analyze sparsity pattern of least squares problem. Needed before solving.

        Processes all costs and variables to compute the sparse Jacobian structure,
        group costs by structure for vectorization, and prepare for optimization.

        Args:
            use_onp: If True, use numpy instead of jax.numpy for index computations.
                Can be faster for problem setup on CPU.

        Returns:
            An AnalyzedLeastSquaresProblem ready for solving.
        """

        # Operations using vanilla numpy can be faster.
        if use_onp:
            jnp = onp
        else:
            from jax import numpy as jnp

        variables = tuple(self.variables)
        compute_residual_from_hash = dict[Hashable, Callable]()

        def _deduplicate_compute_residual(cost: Cost) -> Cost:
            """Deduplicate compute_residual functions with identical signatures."""
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

    def _compute_cost_info(self, vals: VarValues) -> _CostInfo:
        """Compute residuals, costs, and Jacobian cache in one pass.

        This bundles all cost-related computation into a single call,
        computing per-group residuals, concatenated residual, total cost,
        and non-constraint cost (original objective) together.

        Args:
            vals: Variable values to evaluate at.

        Returns:
            _CostInfo with all computed values.
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

        return _CostInfo(
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
        """Compute Jacobian values in block-row sparse format.

        Computes the Jacobian of the residual vector with respect to the tangent
        space of all variables. Uses either autodiff or custom Jacobian functions
        if provided.

        The Jacobian is stored in a block-row sparse format where each cost group
        contributes a block of rows. Within each block, the Jacobian has a
        block-sparse structure determined by which variables each cost depends on.

        Args:
            vals: Variable values at which to evaluate the Jacobian.
            jac_cache: Cached values from residual computation, used by custom
                Jacobian functions. One entry per cost group (None if no cache).

        Returns:
            BlockRowSparseMatrix containing the full Jacobian with shape
            (residual_dim, tangent_dim).
        """
        block_rows = list[SparseBlockRow]()
        residual_offset = 0

        for i, cost in enumerate(self._stacked_costs):
            # Shape should be: (count_from_group[group_key], single_residual_dim, sum_of_tangent_dims_of_variables).
            def compute_jac_with_perturb(
                cost: _AnalyzedCost, jac_cache_i: CustomJacobianCache | None = None
            ) -> jax.Array:
                """Compute Jacobian for a single cost using perturbation or custom function."""
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

    def make_covariance_estimator(
        self,
        vals: VarValues,
        method: Literal["cholmod_spinv"] | "LinearSolverCovarianceEstimatorConfig" | None = None,
        *,
        scale_by_residual_variance: bool = True,
    ) -> "CovarianceEstimator":
        """Create a covariance estimator for uncertainty quantification.

        This computes blocks of the covariance matrix (J^T J)^{-1}, which
        represents the uncertainty of estimated variables at the solution.
        The covariance is computed in the tangent space, appropriate for
        manifold variables like SE3, SO3, etc.

        Args:
            vals: Variable values at which to compute covariance (typically
                the solution from solve()).
            method: Covariance computation method. Options:
                - None (default): Use CG with block-Jacobi preconditioning.
                  GPU-friendly and adapts to problem structure.
                - LinearSolverCovarianceEstimatorConfig: Custom linear solver config.
                - "cholmod_spinv": Use CHOLMOD's sparse inverse. Fast extraction
                  but requires sksparse and only includes entries in the
                  sparsity pattern.
            scale_by_residual_variance: If True, scale by the estimated residual
                variance sigma^2 = ||r||^2 / (m - n), where m is the number of
                residuals and n is the tangent dimension.

        Returns:
            A CovarianceEstimator that can compute covariance blocks via
            estimator.covariance(var0, var1).
        """
        from ._covariance import (
            CovarianceEstimator,
            LinearSolveCovarianceEstimator,
            LinearSolverCovarianceEstimatorConfig,
            SpinvCovarianceEstimator,
        )
        from ._preconditioning import (
            make_block_jacobi_precoditioner,
            make_point_jacobi_precoditioner,
        )

        # Default to CG-based estimator.
        if method is None:
            method = LinearSolverCovarianceEstimatorConfig()

        # Compute residual variance if needed.
        if scale_by_residual_variance:
            residual = self.compute_residual_vector(vals)
            m = self._residual_dim
            n = self._tangent_dim
            residual_variance = jnp.sum(residual**2) / jnp.maximum(m - n, 1)
        else:
            residual_variance = jnp.array(1.0)

        if method == "cholmod_spinv":
            # Use CHOLMOD spinv for sparse inverse.
            import sksparse.cholmod

            # Compute the Jacobian.
            cost_info = self._compute_cost_info(vals)
            A_blocksparse = self._compute_jac_values(vals, cost_info.jac_cache)

            # Build CSR matrix for CHOLMOD.
            jac_values = jnp.concatenate(
                [
                    block_row.blocks_concat.flatten()
                    for block_row in A_blocksparse.block_rows
                ],
                axis=0,
            )
            import scipy.sparse

            # Convert to numpy arrays for scipy/CHOLMOD.
            # CHOLMOD requires float64.
            A_csr = scipy.sparse.csr_matrix(
                (
                    onp.asarray(jac_values, dtype=onp.float64),
                    onp.asarray(self._jac_coords_csr.indices),
                    onp.asarray(self._jac_coords_csr.indptr),
                ),
                shape=self._jac_coords_csr.shape,
            )
            # Compute J^T @ J explicitly (the Hessian).
            JTJ = (A_csr.T @ A_csr).tocsc()
            # Add regularization to ensure positive definiteness.
            n = JTJ.shape[0]
            JTJ = JTJ + 1e-8 * scipy.sparse.eye(n, format="csc", dtype=onp.float64)

            # Compute Cholesky factorization.
            factor = sksparse.cholmod.cholesky(JTJ)

            # Compute inverse by solving (J^T J) X = I column by column.
            cov_dense = onp.zeros((n, n), dtype=onp.float64)
            for i in range(n):
                e_i = onp.zeros(n, dtype=onp.float64)
                e_i[i] = 1.0
                cov_dense[:, i] = factor(e_i)
            sparse_cov = scipy.sparse.csr_matrix(cov_dense)

            return SpinvCovarianceEstimator(
                _sparse_cov=sparse_cov,
                _residual_variance=residual_variance,
                _tangent_start_from_var_type=self._tangent_start_from_var_type,
                _sorted_ids_from_var_type=self._sorted_ids_from_var_type,
            )

        elif isinstance(method, LinearSolverCovarianceEstimatorConfig):
            # Use linear solve-based estimator.
            cost_info = self._compute_cost_info(vals)
            A_blocksparse = self._compute_jac_values(vals, cost_info.jac_cache)

            # Build J @ v and J^T @ v functions.
            A_multiply = A_blocksparse.multiply
            AT_multiply_ = jax.linear_transpose(
                A_multiply, jnp.zeros((A_blocksparse.shape[1],))
            )
            AT_multiply = lambda vec: AT_multiply_(vec)[0]

            def ATA_multiply(vec: jax.Array) -> jax.Array:
                return AT_multiply(A_multiply(vec))

            # Build solve function based on linear solver config.
            linear_solver = method.linear_solver

            if isinstance(linear_solver, ConjugateGradientConfig) or linear_solver == "conjugate_gradient":
                cg_config = (
                    ConjugateGradientConfig()
                    if linear_solver == "conjugate_gradient"
                    else linear_solver
                )

                # Set up preconditioner.
                if cg_config.preconditioner == "block_jacobi":
                    preconditioner = make_block_jacobi_precoditioner(self, A_blocksparse)
                elif cg_config.preconditioner == "point_jacobi":
                    preconditioner = make_point_jacobi_precoditioner(A_blocksparse)
                else:
                    preconditioner = lambda x: x

                def solve_fn(b: jax.Array) -> jax.Array:
                    x, _ = jax.scipy.sparse.linalg.cg(
                        A=ATA_multiply,
                        b=b,
                        x0=jnp.zeros(self._tangent_dim),
                        maxiter=self._tangent_dim,
                        tol=cast(float, cg_config.tolerance_min),
                        M=preconditioner,
                    )
                    return x

            elif linear_solver == "dense_cholesky":
                # Dense Cholesky with cached factorization.
                A_dense = A_blocksparse.to_dense()
                ATA = A_dense.T @ A_dense
                ATA = ATA + 1e-8 * jnp.eye(self._tangent_dim)
                cho_factor = jax.scipy.linalg.cho_factor(ATA)

                def solve_fn(b: jax.Array) -> jax.Array:
                    return jax.scipy.linalg.cho_solve(cho_factor, b)

            else:
                raise ValueError(f"Unknown linear solver: {linear_solver}")

            return LinearSolveCovarianceEstimator(
                _solve_fn=solve_fn,
                _tangent_dim=self._tangent_dim,
                _residual_variance=residual_variance,
                _tangent_start_from_var_type=self._tangent_start_from_var_type,
                _sorted_ids_from_var_type=self._sorted_ids_from_var_type,
            )

        else:
            raise ValueError(f"Unknown covariance method: {method}")
