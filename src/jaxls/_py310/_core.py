from __future__ import annotations

import dis
import functools
from typing import Any

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


@jdc.pytree_dataclass
class _ResidualInfo:
    residual_vectors: Any

    residual_vector: Any

    cost_total: Any

    cost_nonconstraint: Any

    jac_cache: Any


def _get_function_signature(func: Any) -> Any:
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
    costs: Any
    variables: Any

    def analyze(self, use_onp: Any = False) -> Any:
        if use_onp:
            jnp = onp
        else:
            from jax import numpy as jnp

        variables = tuple(self.variables)
        compute_residual_from_hash = dict()

        def _deduplicate_compute_residual(cost: Any) -> Any:
            with jdc.copy_and_mutate(cost) as cost_copy:
                cost_copy.compute_residual = compute_residual_from_hash.setdefault(
                    _get_function_signature(cost.compute_residual),
                    cost.compute_residual,
                )
            return cost_copy

        costs = tuple(_deduplicate_compute_residual(cost) for cost in self.costs)

        count_by_kind: Any = {
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

        tangent_start_from_var_type = dict()

        def _sort_key(x: Any) -> Any:
            return str(x)

        count_from_var_type = dict()
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

        tangent_ordering = VarTypeOrdering(
            {
                var_type: i
                for i, var_type in enumerate(tangent_start_from_var_type.keys())
            }
        )

        costs_from_group = dict()
        count_from_group = dict()
        constraint_index_from_group = dict()
        constraint_index = 0

        for cost in costs:
            cost = cost._broadcast_batch_axes()
            batch_axes = cost._get_batch_axes()

            group_key: Any = (
                jax.tree.structure(cost),
                tuple(
                    leaf.shape[len(batch_axes) :] if hasattr(leaf, "shape") else ()
                    for leaf in jax.tree.leaves(cost)
                ),
            )

            if group_key not in costs_from_group:
                costs_from_group[group_key] = []
                count_from_group[group_key] = 0

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

        stacked_costs = list()
        cost_counts = list()
        jac_coords = list()

        sorted_ids_from_var_type = sort_and_stack_vars(variables)
        del variables

        residual_dim_sum = 0
        for group_key in sorted(costs_from_group.keys(), key=_sort_key):
            group = costs_from_group[group_key]
            count = count_from_group[group_key]

            stacked_cost: Any = jax.tree.map(
                lambda *args: jnp.concatenate(args, axis=0), *group
            )

            is_constraint_group = group_key in constraint_index_from_group
            if is_constraint_group:
                group_constraint_index = constraint_index_from_group[group_key]
                stacked_cost_expanded: Any = jax.vmap(
                    lambda c: _augment_constraint_cost(c, group_constraint_index)
                )(stacked_cost)
            else:
                stacked_cost_expanded: Any = jax.vmap(_AnalyzedCost._make)(stacked_cost)

            stacked_costs.append(stacked_cost_expanded)
            cost_counts.append(count)

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

        jac_coords_coo = SparseCooCoordinates(
            *jax.tree.map(lambda *arrays: jnp.concatenate(arrays, axis=0), *jac_coords),
            shape=(residual_dim_sum, tangent_dim_sum),
        )
        csr_indptr = jnp.searchsorted(
            jac_coords_coo.rows, jnp.arange(residual_dim_sum + 1)
        )
        jac_coords_csr = SparseCsrCoordinates(
            indices=jac_coords_coo.cols,
            indptr=csr_indptr,
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
    _stacked_costs: Any
    _cost_counts: jdc.Static[Any]
    _sorted_ids_from_var_type: Any
    _jac_coords_coo: Any
    _jac_coords_csr: Any
    _tangent_ordering: jdc.Static[Any]
    _tangent_start_from_var_type: jdc.Static[Any]
    _tangent_dim: jdc.Static[Any]
    _residual_dim: jdc.Static[Any]

    def solve(
        self,
        initial_vals: Any = None,
        *,
        linear_solver: Any = "conjugate_gradient",
        trust_region: Any = TrustRegionConfig(),
        termination: Any = TerminationConfig(),
        sparse_mode: Any = "blockrow",
        verbose: Any = True,
        augmented_lagrangian: Any = None,
        return_summary: Any = False,
    ) -> Any:
        if initial_vals is None:
            initial_vals = VarValues.make(
                var_type(ids)
                for var_type, ids in self._sorted_ids_from_var_type.items()
            )

        has_constraints = any(cost.kind != "l2_squared" for cost in self._stacked_costs)

        if has_constraints and augmented_lagrangian is None:
            from ._augmented_lagrangian import AugmentedLagrangianConfig

            augmented_lagrangian = AugmentedLagrangianConfig()

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
            augmented_lagrangian if has_constraints else None,
        )
        return solver.solve(
            problem=self, initial_vals=initial_vals, return_summary=return_summary
        )

    def compute_residual_vector(self, vals: Any) -> Any:
        residual_slices = list()
        jac_cache = list()
        for stacked_cost in self._stacked_costs:
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

    def _compute_residual_info(self, vals: Any, include_jac_cache: Any = False) -> Any:
        residual_vectors: Any = []
        jac_caches: Any = []
        cost_nonconstraint = jnp.array(0.0)

        for stacked_cost in self._stacked_costs:
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

            if stacked_cost.kind == "l2_squared":
                cost_nonconstraint = cost_nonconstraint + jnp.sum(residual**2)

        residual_vector = jnp.concatenate(residual_vectors, axis=0)
        cost_total = jnp.sum(residual_vector**2)

        return _ResidualInfo(
            residual_vectors=tuple(residual_vectors),
            residual_vector=residual_vector,
            cost_total=cost_total,
            cost_nonconstraint=cost_nonconstraint,
            jac_cache=tuple(jac_caches) if include_jac_cache else None,
        )

    def _compute_constraint_values(self, vals: Any) -> Any:
        constraint_slices = list()
        for stacked_cost in self._stacked_costs:
            if stacked_cost.kind == "l2_squared":
                continue

            assert stacked_cost.compute_residual_original is not None
            constraint_vals = jax.vmap(
                lambda c: c.compute_residual_original(vals, *c.args)
            )(stacked_cost)

            constraint_slices.append(constraint_vals)

        return tuple(constraint_slices)

    def compute_constraint_values(self, vals: Any) -> Any:
        constraint_slices = self._compute_constraint_values(vals)
        if len(constraint_slices) == 0:
            return jnp.array([])

        return jnp.concatenate([c.reshape(-1) for c in constraint_slices], axis=0)

    def _compute_jac_values(self, vals: Any, jac_cache: Any) -> Any:
        block_rows = list()
        residual_offset = 0

        for i, cost in enumerate(self._stacked_costs):

            def compute_jac_with_perturb(cost: Any, jac_cache_i: Any = None) -> Any:
                val_subset = vals._get_subset(
                    {
                        var_type: jnp.searchsorted(vals.ids_from_type[var_type], ids)
                        for var_type, ids in cost.sorted_ids_from_var_type.items()
                    },
                    self._tangent_ordering,
                )

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

                return jacfunc(
                    lambda tangent: cost.compute_residual_flat(
                        val_subset._retract(tangent, self._tangent_ordering),
                        *cost.args,
                    )
                )(jnp.zeros((val_subset._get_tangent_dim(),)))

            optional_jac_cache_i = (jac_cache[i],) if jac_cache[i] is not None else ()

            if cost.jac_batch_size is None:
                stacked_jac = jax.vmap(compute_jac_with_perturb)(
                    cost, *optional_jac_cache_i
                )
            else:
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
                stacked_jac.shape[-1],
            )

            stacked_jac_start_col = 0
            start_cols = list()
            block_widths = list()
            for var_type, ids in self._tangent_ordering.ordered_dict_items(
                cost.sorted_ids_from_var_type
            ):
                (num_costs_, num_vars) = ids.shape
                assert num_costs == num_costs_

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
