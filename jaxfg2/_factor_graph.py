from __future__ import annotations

import dataclasses
import dis
import functools
import inspect
import linecache
from typing import Callable, Hashable, Iterable, Literal, Mapping, Self, cast

import jax
import jax_dataclasses as jdc
import numpy as onp
from frozendict import frozendict
from jax import numpy as jnp
from loguru import logger

from ._sparse_matrices import (
    SparseCooCoordinates,
    SparseCooMatrix,
    SparseCsrCoordinates,
)
from ._variables import Var, VarTypeOrdering, VarValues, sort_and_stack_vars


@jdc.pytree_dataclass
class StackedFactorGraph:
    _stacked_factors: tuple[Factor, ...]
    _jacobian_coords_coo: SparseCooCoordinates
    _jacobian_coords_csr: SparseCsrCoordinates
    _tangent_ordering: jdc.Static[VarTypeOrdering]
    _residual_dim: jdc.Static[int]

    def compute_residual_vector(self, vals: VarValues) -> jax.Array:
        residual_slices = list[jax.Array]()
        for stacked_factor in self._stacked_factors:
            stacked_residual_slice = jax.vmap(
                lambda args: stacked_factor.compute_residual(vals, *args)
            )(stacked_factor.args)
            assert len(stacked_residual_slice.shape) == 2
            residual_slices.append(stacked_residual_slice.reshape((-1,)))
        return jnp.concatenate(residual_slices, axis=0)

    def _compute_jacobian_values(self, vals: VarValues) -> jax.Array:
        jac_vals = []
        for factor in self._stacked_factors:
            # Shape should be: (num_variables, len(group), single_residual_dim, var.tangent_dim).
            def compute_jac_with_perturb(factor: Factor) -> jax.Array:
                val_subset = vals._get_subset(
                    {
                        var_type: jnp.searchsorted(vals.ids_from_type[var_type], ids)
                        for var_type, ids in factor.sorted_ids_from_var_type.items()
                    },
                    self._tangent_ordering,
                )
                # Shape should be: (single_residual_dim, vars * tangent_dim).
                return (
                    # Getting the Jacobian of...
                    jax.jacrev
                    if (
                        factor.jacobian_mode == "auto"
                        and factor.residual_dim < val_subset._get_tangent_dim()
                        or factor.jacobian_mode == "reverse"
                    )
                    else jax.jacfwd
                )(
                    # The residual function, with respect to to some local delta.
                    lambda tangent: factor.compute_residual(
                        val_subset._retract(tangent, self._tangent_ordering),
                        *factor.args,
                    )
                )(jnp.zeros((val_subset._get_tangent_dim(),)))

            stacked_jac = jax.vmap(compute_jac_with_perturb)(factor)
            (num_factor,) = factor._get_batch_axes()
            assert stacked_jac.shape == (
                num_factor,
                factor.residual_dim,
                stacked_jac.shape[-1],  # Tangent dimension.
            )
            jac_vals.append(stacked_jac.flatten())
        jac_vals = jnp.concatenate(jac_vals, axis=0)
        assert jac_vals.shape == (self._jacobian_coords_coo.rows.shape[0],)
        return jac_vals

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
            factors_from_group.setdefault(group_key, []).append(factor)

        # Fields we want to populate.
        stacked_factors: list[Factor] = []
        jacobian_coords: list[tuple[jax.Array, jax.Array]] = []

        # Create storage layout: this describes which parts of our tangent
        # vector is allocated to each variable.
        tangent_start_from_var_type = dict[type[Var], int]()

        vars_from_var_type = dict[type[Var], list[Var]]()
        for var in vars:
            vars_from_var_type.setdefault(type(var), []).append(var)
        tangent_offset = 0
        for var_type, vars_one_type in vars_from_var_type.items():
            tangent_start_from_var_type[var_type] = tangent_offset
            tangent_offset += var_type.tangent_dim * len(vars_one_type)

        # Create ordering helper.
        tangent_ordering = VarTypeOrdering(
            {
                var_type: i
                for i, var_type in enumerate(tangent_start_from_var_type.keys())
            }
        )

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
            rows, cols = jax.vmap(
                functools.partial(
                    Factor._compute_block_sparse_jacobian_indices,
                    tangent_ordering=tangent_ordering,
                    sorted_ids_from_var_type=sorted_ids_from_var_type,
                    tangent_start_from_var_type=tangent_start_from_var_type,
                )
            )(stacked_factor)
            assert (
                rows.shape
                == cols.shape
                == (len(group), stacked_factor.residual_dim, rows.shape[-1])
            )
            rows = rows + (
                jnp.arange(len(group))[:, None, None] * stacked_factor.residual_dim
            )
            rows = rows + residual_offset
            jacobian_coords.append((rows.flatten(), cols.flatten()))
            residual_offset += stacked_factor.residual_dim * len(group)

        jacobian_coords_concat: SparseCooCoordinates = SparseCooCoordinates(
            *jax.tree_map(
                lambda *arrays: jnp.concatenate(arrays, axis=0), *jacobian_coords
            ),
            shape=(residual_offset, tangent_offset),
        )
        logger.info("Done!")
        return StackedFactorGraph(
            _stacked_factors=tuple(stacked_factors),
            _jacobian_coords_coo=jacobian_coords_concat,
            _jacobian_coords_csr=SparseCsrCoordinates(
                indices=jacobian_coords_concat.cols,
                indptr=cast(
                    jax.Array,
                    jnp.searchsorted(
                        jacobian_coords_concat.rows, jnp.arange(residual_offset + 1)
                    ),
                ),
                shape=(residual_offset, tangent_offset),
            ),
            _tangent_ordering=tangent_ordering,
            _residual_dim=residual_offset,
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
    jacobian_mode: jdc.Static[Literal["auto", "forward", "reverse"]]

    @staticmethod
    def make[*Args_](
        compute_residual: Callable[[VarValues, *Args_], jax.Array],
        args: tuple[*Args_],
        jacobian_mode: Literal["auto", "forward", "reverse"] = "auto",
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
        return Factor._make_impl(compute_residual, args, jacobian_mode)

    @staticmethod
    @jdc.jit
    def _make_impl[*Args_](
        compute_residual: jdc.Static[Callable[[VarValues, *Args_], jax.Array]],
        args: tuple[*Args_],
        jacobian_mode: jdc.Static[Literal["auto", "forward", "reverse"]],
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
            jacobian_mode=jacobian_mode,
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

    def _compute_block_sparse_jacobian_indices(
        self: Factor,
        tangent_ordering: VarTypeOrdering,
        sorted_ids_from_var_type: dict[type[Var], jax.Array],
        tangent_start_from_var_type: dict[type[Var], int],
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
