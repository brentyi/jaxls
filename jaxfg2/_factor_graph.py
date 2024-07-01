from __future__ import annotations

import dataclasses
from typing import Any, Callable, Hashable, Iterable, Mapping, Self, cast

import jax
import jax_dataclasses as jdc
import numpy as onp
from frozendict import frozendict
from jax import numpy as jnp

from ._sparse_matrices import SparseCooCoordinates
from ._variables import Var, VarValues, _VarOrderInfo


@dataclasses.dataclass(frozen=True)
class _TangentStorageLayout:
    """Our tangent vector will be represented as a single flattened 1D vector.
    How should it be laid out?"""

    counts: Mapping[type[Var], int] = {}
    start_indices: Mapping[type[Var], int] = {}

    @staticmethod
    def make(vars: Iterable[Var]) -> _TangentStorageLayout:
        counts: dict[type[Var], int] = {}
        for var in vars:
            var_type = type(var)
            counts[var_type] = counts.get(var_type, 0) + 1

        i = 0
        start_indices: dict[type[Var], int] = {}
        for var_type, count in counts.items():
            start_indices[var_type] = i
            i += var_type.parameter_dim * count

        return _TangentStorageLayout(
            counts=frozendict(counts),
            start_indices=frozendict(start_indices),
        )


@jdc.pytree_dataclass
class StackedFactorGraph:
    stacked_factors: list[Factor]
    jacobian_coords: SparseCooCoordinates
    tangent_layout: jdc.Static[_TangentStorageLayout]
    residual_dim: jdc.Static[int]

    @staticmethod
    def make(
        factors: Iterable[Factor],
        vars: Iterable[Var],
    ) -> StackedFactorGraph:
        """Create a factor graph from a set of factors and a set of variables."""

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

            # Record factor and variables
            factors_from_group[group_key].append(factor)

        # Fields we want to populate.
        stacked_factors: list[Factor] = []
        jacobian_coords: list[SparseCooCoordinates] = []

        # Create storage layout: this describes which parts of our tangent
        # vector is allocated to each variable.
        tangent_layout = _TangentStorageLayout.make(vars)
        order_from_var_type = _VarOrderInfo.make(vars)

        # Prepare each factor group.
        residual_offset = 0
        for group_key, group in factors_from_group.items():
            # Make factor stack
            stacked_factor: Factor = jax.tree.map(
                lambda *args: jnp.stack(args, axis=0), group
            )
            stacked_factors.append(stacked_factor)

            # Compute Jacobian coordinates
            #
            # These should be N pairs of (row, col) indices, where rows correspond to
            # residual indices and columns correspond to tangent vector indices.
            stacked_residual_dim = stacked_factor.residual_dim * len(group)
            for var in stacked_factor._get_variables():
                # Jacobian of residual slice with respect to a single, zero-indexed variable.
                jac_coords = jnp.mgrid[
                    residual_offset : residual_offset + stacked_residual_dim,
                    : var.tangent_dim,
                ]
                assert jac_coords.shape == (2, stacked_residual_dim, var.tangent_dim)

                # Get index of each variable, which is based on the sorted IDs.
                var_indices = jax.vmap(order_from_var_type[type(var)].index_from_id)(
                    var.id
                )
                assert var_indices.shape == (len(group),)

                # Variable index => indices into the tangent vector.
                tangent_start_indices = (
                    tangent_layout.start_indices[type(var)]
                    + var_indices * var.tangent_dim
                )
                assert tangent_start_indices.shape == (len(group),)

                jac_coords = (
                    jnp.broadcast_to(
                        jac_coords[:, None, :, :],
                        (2, len(group), stacked_residual_dim, var.tangent_dim),
                    )
                    # Offset the column indices by the start index within the
                    # flattened tangent vector.
                    .at[1, :, :, :]
                    .add(tangent_start_indices[:, None, None])
                )

                # tangent_layout.start_indices[var]
                jacobian_coords.append(
                    SparseCooCoordinates(
                        rows=jac_coords[0].flatten(),
                        cols=jac_coords[1].flatten(),
                    )
                )
            residual_offset += stacked_residual_dim

        jacobian_coords_concat: SparseCooCoordinates = jax.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0), *jacobian_coords
        )

        return StackedFactorGraph(
            stacked_factors=stacked_factors,
            jacobian_coords=jacobian_coords_concat,
            tangent_layout=tangent_layout,
            residual_dim=residual_offset,
        )


@jdc.pytree_dataclass
class Factor[*Args]:
    """A single cost in our factor graph."""

    compute_residual: jdc.Static[Callable[[VarValues, *Args], jax.Array]]
    args: tuple[*Args]
    variable_indices: tuple[int, ...]
    residual_dim: jdc.Static[int]

    @staticmethod
    def make(
        compute_residual: Callable[[VarValues, *Args], jax.Array],
        args: tuple[*Args],
    ) -> Factor:
        """Construct a factor for our factor graph."""
        (residual_dim,) = jax.eval_shape(compute_residual, args).shape
        variable_indices = tuple(
            i for i, arg in enumerate(args) if isinstance(arg, Var)
        )
        return Factor(
            compute_residual,
            args=args,
            variable_indices=variable_indices,
            residual_dim=residual_dim,
        )

    def _get_variables(self) -> tuple[Var, ...]:
        """Returns a tuple of variables, which are arguments for the residual function."""
        return tuple(cast(Var, self.args[i]) for i in self.variable_indices)
