from __future__ import annotations

import dataclasses
import dis
import inspect
import linecache
from typing import Callable, Hashable, Iterable, Mapping, Self, cast

import jax
import jax_dataclasses as jdc
import numpy as onp
from frozendict import frozendict
from jax import numpy as jnp
from loguru import logger

from ._sparse_matrices import SparseCooCoordinates, SparseCooMatrix
from ._variables import Var, VarTypeOrdering, VarValues, sort_and_stack_vars


@dataclasses.dataclass(frozen=True)
class TangentStorageLayout:
    """Our tangent vector will be represented as a single flattened 1D vector.
    How should it be laid out?"""

    counts: Mapping[type[Var], int]
    start_indices: Mapping[type[Var], int]
    ordering: VarTypeOrdering

    @staticmethod
    def make(vars: Iterable[Var]) -> TangentStorageLayout:
        counts: dict[type[Var], int] = {}
        for var in vars:
            var_type = type(var)
            if isinstance(var.id, int) or len(var.id.shape) == 0:
                counts[var_type] = counts.get(var_type, 0) + 1
            else:
                assert len(var.id.shape) == 1
                counts[var_type] = counts.get(var_type, 0) + var.id.shape[0]

        i = 0
        start_indices: dict[type[Var], int] = {}
        for var_type, count in counts.items():
            start_indices[var_type] = i
            i += var_type.parameter_dim * count

        return TangentStorageLayout(
            counts=frozendict(counts),
            start_indices=frozendict(start_indices),
            ordering=VarTypeOrdering(tuple(start_indices.keys())),
        )


@jdc.pytree_dataclass
class StackedFactorGraph:
    stacked_factors: tuple[Factor, ...]
    stacked_factors_var_indices: tuple[dict[type[Var], jax.Array], ...]
    jacobian_coords: SparseCooCoordinates
    tangent_layout: jdc.Static[TangentStorageLayout]
    residual_dim: jdc.Static[int]

    def compute_residual_vector(self, vals: VarValues) -> jax.Array:
        residual_slices = list[jax.Array]()
        for stacked_factor in self.stacked_factors:
            stacked_residual_slice = jax.vmap(
                lambda args: stacked_factor.compute_residual(vals, *args)
            )(stacked_factor.args)
            assert len(stacked_residual_slice.shape) == 2
            residual_slices.append(stacked_residual_slice.reshape((-1,)))
        return jnp.concatenate(residual_slices, axis=0)

    def _compute_jacobian_wrt_tangent(self, vals: VarValues) -> SparseCooMatrix:
        jac_vals = []
        for factor, subset_indices in zip(
            self.stacked_factors, self.stacked_factors_var_indices
        ):
            # Shape should be: (num_variables, len(group), single_residual_dim, var.tangent_dim).
            def compute_jac_with_perturb(
                factor: Factor, indices_from_type: dict[type[Var], jax.Array]
            ) -> jax.Array:
                val_subset = vals._get_subset(
                    indices_from_type, self.tangent_layout.ordering
                )
                # Shape should be: (single_residual_dim, vars * tangent_dim).
                return (
                    # Getting the Jacobian of...
                    jax.jacrev
                    if factor.residual_dim < val_subset._get_tangent_dim()
                    else jax.jacfwd
                )(
                    # The residual function, with respect to to some local delta.
                    lambda tangent: factor.compute_residual(
                        val_subset._retract(tangent, self.tangent_layout.ordering),
                        *factor.args,
                    )
                )(jnp.zeros((val_subset._get_tangent_dim(),)))

            stacked_jac = jax.vmap(compute_jac_with_perturb)(factor, subset_indices)
            (num_factor,) = factor._get_batch_axes()
            assert stacked_jac.shape == (
                num_factor,
                factor.residual_dim,
                stacked_jac.shape[-1],  # Tangent dimension.
            )
            jac_vals.append(stacked_jac.flatten())
        jac_vals = jnp.concatenate(jac_vals, axis=0)
        assert jac_vals.shape == (self.jacobian_coords.rows.shape[0],)
        return SparseCooMatrix(
            values=jac_vals,
            coords=self.jacobian_coords,
            shape=(self.residual_dim, vals._get_tangent_dim()),
        )

    @staticmethod
    def make(
        factors: Iterable[Factor],
        vars: Iterable[Var],
        use_onp: bool = True,
    ) -> StackedFactorGraph:
        """Create a factor graph from a set of factors and a set of variables."""

        # Operations using vanilla numpy can be faster.
        if use_onp:
            jnp = onp
        else:
            from jax import numpy as jnp

        factors = tuple(factors)
        vars = tuple(vars)
        logger.info(
            "Building graph with {} factors and {} variables.", len(factors), len(vars)
        )

        # Start by grouping our factors and grabbing a list of (ordered!) variables
        factors_from_group = dict[Hashable, list[Factor]]()
        for factor in factors:
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

            # Record factor and variables.
            factors_from_group.setdefault(group_key, [])
            factors_from_group[group_key].append(factor)

        # Fields we want to populate.
        stacked_factors: list[Factor] = []
        stacked_factor_var_indices: list[dict[type[Var], jax.Array]] = []
        jacobian_coords: list[SparseCooCoordinates] = []

        # Create storage layout: this describes which parts of our tangent
        # vector is allocated to each variable.
        tangent_layout = TangentStorageLayout.make(vars)

        # Sort variable IDs.
        sorted_ids_from_var_type = sort_and_stack_vars(vars)
        del vars

        # Prepare each factor group.
        residual_offset = 0
        for group_key, group in factors_from_group.items():
            logger.info(
                "Group with factors={}, variables={}: {}",
                len(group),
                group[0].num_vars,
                group[0].compute_residual.__name__,
            )

            # Stack factor parameters.
            stacked_factor: Factor = jax.tree.map(
                lambda *args: jnp.stack(args, axis=0), *group
            )
            stacked_factors.append(stacked_factor)

            # Compute Jacobian coordinates.
            #
            # These should be N pairs of (row, col) indices, where rows correspond to
            # residual indices and columns correspond to tangent vector indices.
            single_residual_dim = stacked_factor.residual_dim
            stacked_residual_dim = single_residual_dim * len(group)

            subset_indices_from_type = dict[type[Var], jax.Array]()

            for var_type, ids in tangent_layout.ordering.ordered_dict_items(
                stacked_factor.sorted_ids_from_var_type
            ):
                logger.info("Making initial grid")
                # Jacobian of a single residual vector with respect to a
                # single variable. Upper-left corner at (0, 0).
                jac_coords = onp.mgrid[:single_residual_dim, : var_type.tangent_dim]
                assert jac_coords.shape == (
                    2,
                    single_residual_dim,
                    var_type.tangent_dim,
                )

                logger.info("Getting indices")

                # Get index of each variable, which is based on the sorted IDs.
                var_indices = jnp.searchsorted(sorted_ids_from_var_type[var_type], ids)
                subset_indices_from_type[var_type] = cast(jax.Array, var_indices)
                assert var_indices.shape == (len(group), ids.shape[-1])

                logger.info("Computing tangent")
                # Variable index => indices into the tangent vector.
                tangent_start_indices = (
                    tangent_layout.start_indices[var_type]
                    + var_indices * var_type.tangent_dim
                )
                assert tangent_start_indices.shape == (len(group), ids.shape[-1])

                logger.info("Broadcasting")
                jac_coords = jnp.broadcast_to(
                    jac_coords[:, None, :, None, :],
                    (
                        2,
                        len(group),
                        single_residual_dim,
                        ids.shape[-1],
                        var_type.tangent_dim,
                    ),
                )
                logger.info(
                    "Computed indices for Jacobian block with shape {}",
                    jac_coords.shape,
                )
                jacobian_coords.append(
                    SparseCooCoordinates(
                        rows=(
                            jac_coords[0]
                            + (
                                onp.arange(len(group)) * single_residual_dim
                                + residual_offset
                            )[:, None, None, None]
                        ).flatten(),
                        # Offset the column indices by the start index within the
                        # flattened tangent vector.
                        cols=(
                            jac_coords[1] + tangent_start_indices[:, None, :, None]
                        ).flatten(),
                    )
                )
            stacked_factor_var_indices.append(subset_indices_from_type)
            residual_offset += stacked_residual_dim

        jacobian_coords_concat: SparseCooCoordinates = jax.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0), *jacobian_coords
        )
        logger.info("Done!")
        return StackedFactorGraph(
            stacked_factors=tuple(stacked_factors),
            stacked_factors_var_indices=tuple(stacked_factor_var_indices),
            jacobian_coords=jacobian_coords_concat,
            tangent_layout=tangent_layout,
            residual_dim=residual_offset,
        )


_residual_dim_cache = dict[Hashable, int]()
_function_cache = dict[Hashable, Callable]()


@jdc.pytree_dataclass
class Factor[*Args]:
    """A single cost in our factor graph."""

    compute_residual: jdc.Static[Callable[[VarValues, *Args], jax.Array]]
    args: tuple[*Args]
    num_vars: jdc.Static[int]
    sorted_ids_from_var_type: dict[type[Var], jax.Array]
    residual_dim: jdc.Static[int]

    @staticmethod
    def make[*Args_](
        compute_residual: Callable[[VarValues, *Args_], jax.Array],
        args: tuple[*Args_],
    ) -> Factor[*Args_]:
        """Construct a factor for our factor graph."""
        # If we see two functions with the same signature, we always use the
        # first one. This helps with vectorization.
        compute_residual = cast(
            Callable[[VarValues, *Args_], jax.Array],
            _function_cache.setdefault(
                Factor._get_function_signature(compute_residual), compute_residual
            ),
        )
        return Factor._make_impl(compute_residual, args)

    @staticmethod
    @jdc.jit
    def _make_impl[*Args_](
        compute_residual: jdc.Static[Callable[[VarValues, *Args_], jax.Array]],
        args: tuple[*Args_],
    ) -> Factor[*Args_]:
        """Construct a factor for our factor graph."""

        # TODO: ideally we should get the variables by traversing as a pytree.
        variable_indices = tuple(
            i for i, arg in enumerate(args) if isinstance(arg, Var)
        )

        # Cache the residual dimension for this factor.
        residual_dim_cache_key = (
            compute_residual,
            variable_indices,
            tuple(type(args[i]) for i in variable_indices),
        )
        if residual_dim_cache_key not in _residual_dim_cache:
            dummy = VarValues.from_defaults(
                tuple(cast(Var, args[i]) for i in variable_indices)
            )
            residual_shape = jax.eval_shape(compute_residual, dummy, *args).shape
            assert len(residual_shape) == 1, "Residual must be a 1D array."
            _residual_dim_cache[residual_dim_cache_key] = residual_shape[0]

        # Let's not leak too much memory...
        MAX_CACHE_SIZE = 512
        if len(_function_cache) > MAX_CACHE_SIZE:
            _function_cache.pop(next(iter(_function_cache.keys())))
        if len(_residual_dim_cache) > MAX_CACHE_SIZE:
            _residual_dim_cache.pop(next(iter(_residual_dim_cache.keys())))

        return Factor(
            compute_residual,
            args=args,
            num_vars=len(variable_indices),
            sorted_ids_from_var_type=sort_and_stack_vars(
                tuple(cast(Var, args[i]) for i in variable_indices)
            ),
            residual_dim=_residual_dim_cache[residual_dim_cache_key],
        )

    @staticmethod
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

    def _get_batch_axes(self) -> tuple[int, ...]:
        return next(iter(self.sorted_ids_from_var_type.values())).shape[:-1]
