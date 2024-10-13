from __future__ import annotations

import dis
import functools
from typing import Any, Callable, Hashable, Iterable, Literal, cast

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
    closure = func.__closure__
    if closure:
        closure_vars = tuple(sorted((str(cell.cell_contents) for cell in closure)))
    else:
        closure_vars = ()

    bytecode = dis.Bytecode(func)
    bytecode_tuple = tuple((instr.opname, instr.argrepr) for instr in bytecode)
    return bytecode_tuple, closure_vars


@jdc.pytree_dataclass
class FactorGraph:
    stacked_factors: tuple[_AnalyzedFactor, ...]
    factor_counts: jdc.Static[tuple[int, ...]]
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
        linear_solver: Literal["cholmod", "conjugate_gradient", "dense_cholesky"]
        | ConjugateGradientConfig = "cholmod",
        trust_region: TrustRegionConfig | None = TrustRegionConfig(),
        termination: TerminationConfig = TerminationConfig(),
        verbose: bool = True,
    ) -> VarValues:
        """Solve the nonlinear least squares problem using either Gauss-Newton
        or Levenberg-Marquardt."""
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
            linear_solver, trust_region, termination, conjugate_gradient_config, verbose
        )
        return solver.solve(graph=self, initial_vals=initial_vals)

    def compute_residual_vector(self, vals: VarValues) -> jax.Array:
        """Compute the residual vector. The cost we are optimizing is defined
        as the sum of squared terms within this vector."""
        residual_slices = list[jax.Array]()
        for stacked_factor in self.stacked_factors:
            stacked_residual_slice = jax.vmap(
                lambda args: stacked_factor.compute_residual(vals, *args)
            )(stacked_factor.args)
            assert len(stacked_residual_slice.shape) == 2
            residual_slices.append(stacked_residual_slice.reshape((-1,)))
        return jnp.concatenate(residual_slices, axis=0)

    def _compute_jac_values(self, vals: VarValues) -> BlockRowSparseMatrix:
        block_rows = list[SparseBlockRow]()
        residual_offset = 0

        for factor in self.stacked_factors:
            # Shape should be: (num_variables, count_from_group[group_key], single_residual_dim, var.tangent_dim).
            def compute_jac_with_perturb(factor: _AnalyzedFactor) -> jax.Array:
                val_subset = vals._get_subset(
                    {
                        var_type: jnp.searchsorted(vals.ids_from_type[var_type], ids)
                        for var_type, ids in factor.sorted_ids_from_var_type.items()
                    },
                    self.tangent_ordering,
                )
                # Shape should be: (single_residual_dim, vars * tangent_dim).
                jacfunc = {
                    "forward": jax.jacfwd,
                    "reverse": jax.jacrev,
                    "auto": jax.jacrev
                    if factor.residual_dim < val_subset._get_tangent_dim()
                    else jax.jacrev,
                }[factor.jac_mode]
                return jacfunc(
                    # The residual function, with respect to to some local delta.
                    lambda tangent: factor.compute_residual(
                        val_subset._retract(tangent, self.tangent_ordering),
                        *factor.args,
                    )
                )(jnp.zeros((val_subset._get_tangent_dim(),)))

            # Compute Jacobian for each factor.
            stacked_jac = jax.vmap(compute_jac_with_perturb)(factor)
            (num_factor,) = factor._get_batch_axes()
            assert stacked_jac.shape == (
                num_factor,
                factor.residual_dim,
                stacked_jac.shape[-1],  # Tangent dimension.
            )

            # Compute block-row representation for sparse Jacobian.
            stacked_jac_start_col = 0
            start_cols = list[jax.Array]()
            block_widths = list[int]()
            for var_type, ids in self.tangent_ordering.ordered_dict_items(
                # This ordering shouldn't actually matter!
                factor.sorted_ids_from_var_type
            ):
                (num_factor_, num_vars) = ids.shape
                assert num_factor == num_factor_

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
                    assert start_cols[-1].shape == (num_factor_,)

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

            residual_offset += factor.residual_dim * num_factor
        assert residual_offset == self.residual_dim

        bsparse_jacobian = BlockRowSparseMatrix(
            block_rows=tuple(block_rows),
            shape=(self.residual_dim, self.tangent_dim),
        )
        return bsparse_jacobian

    @staticmethod
    def make(
        factors: Iterable[Factor],
        variables: Iterable[Var],
        use_onp: bool = True,
    ) -> FactorGraph:
        """Create a factor graph from a set of factors and a set of variables."""

        # Operations using vanilla numpy can be faster.
        if use_onp:
            jnp = onp
        else:
            from jax import numpy as jnp

        variables = tuple(variables)
        compute_residual_from_hash = dict[Hashable, Callable]()
        factors = tuple(
            jdc.replace(
                factor,
                compute_residual=compute_residual_from_hash.setdefault(
                    _get_function_signature(factor.compute_residual),
                    factor.compute_residual,
                ),
            )
            for factor in factors
        )

        # We're assuming no more than 1 batch axis.
        num_factors = 0
        for f in factors:
            assert len(f._get_batch_axes()) in (0, 1)
            num_factors += (
                1 if len(f._get_batch_axes()) == 0 else f._get_batch_axes()[0]
            )

        num_variables = 0
        for v in variables:
            assert isinstance(v.id, int) or len(v.id.shape) in (0, 1)
            num_variables += (
                1 if isinstance(v.id, int) or v.id.shape == () else v.id.shape[0]
            )
        logger.info(
            "Building graph with {} factors and {} variables.",
            num_factors,
            num_variables,
        )

        # Create storage layout: this describes which parts of our tangent
        # vector is allocated to each variable.
        tangent_start_from_var_type = dict[type[Var[Any]], int]()

        def _sort_key(x: Any) -> str:
            """We're going to sort variable / factor types by name. This should
            prevent re-compiling when factors or variables are reordered."""
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

        # Start by grouping our factors and grabbing a list of (ordered!) variables
        factors_from_group = dict[Any, list[Factor]]()
        count_from_group = dict[Any, int]()
        for i, factor in enumerate(factors):
            # Each factor is ultimately just a pytree node; in order for a set of
            # factors to be batchable, they must share the same:
            group_key: Hashable = (
                # (1) Treedef. Structure of inputs must match.
                jax.tree_structure(factor),
                # (2) Leaf shapes: contained array shapes must match
                tuple(
                    leaf.shape if hasattr(leaf, "shape") else ()
                    for leaf in jax.tree_leaves(factor)
                ),
            )
            factors_from_group.setdefault(group_key, [])
            count_from_group.setdefault(group_key, 0)

            batch_axes = factor._get_batch_axes()
            if len(batch_axes) == 0:
                factor = jax.tree.map(lambda x: jnp.asarray(x)[None], factor)
                count_from_group[group_key] += 1
            else:
                assert len(batch_axes) == 1
                count_from_group[group_key] += batch_axes[0]

            # Record factor and variables.
            factors_from_group[group_key].append(factor)

        # Fields we want to populate.
        stacked_factors = list[_AnalyzedFactor]()
        factor_counts = list[int]()
        jac_coords = list[tuple[jax.Array, jax.Array]]()

        # Sort variable IDs.
        sorted_ids_from_var_type = sort_and_stack_vars(variables)
        del variables

        # Prepare each factor group. We put groups in a consistent order.
        residual_dim_sum = 0
        for group_key in sorted(factors_from_group.keys(), key=_sort_key):
            group = factors_from_group[group_key]

            # Stack factor parameters.
            stacked_factor: Factor = jax.tree.map(
                lambda *args: jnp.concatenate(args, axis=0), *group
            )
            stacked_factor_expanded: _AnalyzedFactor = jax.vmap(_AnalyzedFactor._make)(
                stacked_factor
            )
            stacked_factors.append(stacked_factor_expanded)
            factor_counts.append(count_from_group[group_key])

            logger.info(
                "Vectorizing group with {} factors, {} variables each: {}",
                count_from_group[group_key],
                stacked_factors[-1].num_variables,
                stacked_factors[-1].compute_residual.__name__,
            )

            # Compute Jacobian coordinates.
            #
            # These should be N pairs of (row, col) indices, where rows correspond to
            # residual indices and columns correspond to tangent vector indices.
            rows, cols = jax.vmap(
                functools.partial(
                    _AnalyzedFactor._compute_block_sparse_jac_indices,
                    tangent_ordering=tangent_ordering,
                    sorted_ids_from_var_type=sorted_ids_from_var_type,
                    tangent_start_from_var_type=tangent_start_from_var_type,
                )
            )(stacked_factor_expanded)
            assert (
                rows.shape
                == cols.shape
                == (
                    count_from_group[group_key],
                    stacked_factor_expanded.residual_dim,
                    rows.shape[-1],
                )
            )
            rows = rows + (
                jnp.arange(count_from_group[group_key])[:, None, None]
                * stacked_factor_expanded.residual_dim
            )
            rows = rows + residual_dim_sum
            jac_coords.append((rows.flatten(), cols.flatten()))
            residual_dim_sum += (
                stacked_factor_expanded.residual_dim * count_from_group[group_key]
            )

        jac_coords_coo: SparseCooCoordinates = SparseCooCoordinates(
            *jax.tree_map(lambda *arrays: jnp.concatenate(arrays, axis=0), *jac_coords),
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
        return FactorGraph(
            stacked_factors=tuple(stacked_factors),
            factor_counts=tuple(factor_counts),
            sorted_ids_from_var_type=sorted_ids_from_var_type,
            jac_coords_coo=jac_coords_coo,
            jac_coords_csr=jac_coords_csr,
            tangent_ordering=tangent_ordering,
            tangent_start_from_var_type=tangent_start_from_var_type,
            tangent_dim=tangent_dim_sum,
            residual_dim=residual_dim_sum,
        )


@jdc.pytree_dataclass
class Factor[*Args]:
    """A single cost in our factor graph."""

    compute_residual: jdc.Static[Callable[[VarValues, *Args], jax.Array]]
    args: tuple[*Args]
    jac_mode: jdc.Static[Literal["auto", "forward", "reverse"]] = "auto"

    @staticmethod
    @deprecated("Use Factor() directly instead of Factor.make()")
    def make[*Args_](
        compute_residual: jdc.Static[Callable[[VarValues, *Args_], jax.Array]],
        args: tuple[*Args_],
        jac_mode: jdc.Static[Literal["auto", "forward", "reverse"]] = "auto",
    ) -> Factor[*Args_]:
        import warnings

        warnings.warn(
            "Use Factor() directly instead of Factor.make()", DeprecationWarning
        )
        return Factor(compute_residual, args, jac_mode)

    def _get_batch_axes(self) -> tuple[int, ...]:
        def traverse_args(current: Any) -> tuple[int, ...]:
            children_and_meta = default_registry.flatten_one_level(current)
            assert children_and_meta is not None
            for child in children_and_meta[0]:
                if isinstance(child, Var):
                    return () if isinstance(child.id, int) else child.id.shape
                else:
                    return traverse_args(child)
            assert False, "No variables found in factor!"

        return traverse_args(self.args)


@jdc.pytree_dataclass
class _AnalyzedFactor[*Args](Factor[*Args]):
    """Same as `Factor`, but with extra fields."""

    # These need defaults because `jac_mode` has a default.
    num_variables: jdc.Static[int] = 0
    sorted_ids_from_var_type: dict[type[Var[Any]], jax.Array] = jdc.field(
        default_factory=dict
    )
    residual_dim: jdc.Static[int] = 0

    @staticmethod
    @jdc.jit
    def _make[*Args_](factor: Factor[*Args_]) -> _AnalyzedFactor[*Args_]:
        """Construct a factor for our factor graph."""

        compute_residual = factor.compute_residual
        args = factor.args
        jac_mode = factor.jac_mode

        # Get all variables in the PyTree structure.
        def traverse_args(current: Any, variables: list[Var]) -> list[Var]:
            children_and_meta = default_registry.flatten_one_level(current)
            if children_and_meta is None:
                return variables
            for child in children_and_meta[0]:
                if isinstance(child, Var):
                    variables.append(child)
                else:
                    traverse_args(child, variables)
            return variables

        variables = tuple(traverse_args(args, []))
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
                return jax.vmap(_AnalyzedFactor._make)(factor)

        # Cache the residual dimension for this factor.
        dummy_vals = jax.eval_shape(VarValues.make, variables)
        residual_shape = jax.eval_shape(compute_residual, dummy_vals, *args).shape
        assert len(residual_shape) == 1, "Residual must be a 1D array."
        (residual_dim,) = residual_shape

        return _AnalyzedFactor(
            compute_residual,
            args=args,
            num_variables=len(variables),
            sorted_ids_from_var_type=sort_and_stack_vars(variables),
            residual_dim=residual_dim,
            jac_mode=jac_mode,
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
            jnp.arange(self.residual_dim),
            jnp.concatenate(col_indices, axis=0),
            indexing="ij",
        )
        return rows, cols
