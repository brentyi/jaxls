from __future__ import annotations

import dis
import functools
from typing import Any, Callable, Hashable, Iterable, Literal, cast

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp
from loguru import logger

from ._solvers import (
    CholmodLinearSolver,
    ConjugateGradientLinearSolver,
    NonlinearSolver,
    TerminationConfig,
    TrustRegionConfig,
)
from ._sparse_matrices import SparseCooCoordinates, SparseCsrCoordinates
from ._variables import Var, VarTypeOrdering, VarValues, sort_and_stack_vars


@jdc.pytree_dataclass
class FactorGraph:
    stacked_factors: tuple[Factor, ...]
    sorted_ids_from_var_type: dict[type[Var], jax.Array]
    jac_coords_coo: SparseCooCoordinates
    jac_coords_csr: SparseCsrCoordinates
    tangent_ordering: jdc.Static[VarTypeOrdering]
    residual_dim: jdc.Static[int]

    def solve(
        self,
        initial_vals: VarValues | None = None,
        linear_solver: CholmodLinearSolver
        | ConjugateGradientLinearSolver = CholmodLinearSolver(),
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
        solver = NonlinearSolver(linear_solver, trust_region, termination, verbose)
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

    def _compute_jac_values(self, vals: VarValues) -> jax.Array:
        jac_vals = []
        for factor in self.stacked_factors:
            # Shape should be: (num_variables, len(group), single_residual_dim, var.tangent_dim).
            def compute_jac_with_perturb(factor: Factor) -> jax.Array:
                val_subset = vals._get_subset(
                    {
                        var_type: jnp.searchsorted(vals.ids_from_type[var_type], ids)
                        for var_type, ids in factor.sorted_ids_from_var_type.items()
                    },
                    self.tangent_ordering,
                )
                # Shape should be: (single_residual_dim, vars * tangent_dim).
                return (
                    # Getting the Jacobian of...
                    jax.jacrev
                    if (
                        factor.jac_mode == "auto"
                        and factor.residual_dim < val_subset._get_tangent_dim()
                        or factor.jac_mode == "reverse"
                    )
                    else jax.jacfwd
                )(
                    # The residual function, with respect to to some local delta.
                    lambda tangent: factor.compute_residual(
                        val_subset._retract(tangent, self.tangent_ordering),
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
        assert jac_vals.shape == (self.jac_coords_coo.rows.shape[0],)
        return jac_vals

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

        factors = tuple(factors)
        variables = tuple(variables)
        logger.info(
            "Building graph with {} factors and {} variables.",
            len(factors),
            len(variables),
        )

        # Start by grouping our factors and grabbing a list of (ordered!) variables
        factors_from_group = dict[Any, list[Factor]]()
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
        jac_coords: list[tuple[jax.Array, jax.Array]] = []

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
            count_from_var_type[type(var)] = count_from_var_type.get(type(var), 0) + 1
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

        # Sort variable IDs.
        sorted_ids_from_var_type = sort_and_stack_vars(variables)
        del variables

        # Prepare each factor group. We put groups in a consistent order.
        residual_dim_sum = 0
        for group_key in sorted(factors_from_group.keys(), key=_sort_key):
            group = factors_from_group[group_key]
            logger.info(
                "Group with factors={}, variables={}: {}",
                len(group),
                group[0].num_variables,
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
                    Factor._compute_block_sparse_jac_indices,
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
            rows = rows + residual_dim_sum
            jac_coords.append((rows.flatten(), cols.flatten()))
            residual_dim_sum += stacked_factor.residual_dim * len(group)

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
        logger.info("Done!")
        return FactorGraph(
            stacked_factors=tuple(stacked_factors),
            sorted_ids_from_var_type=sorted_ids_from_var_type,
            jac_coords_coo=jac_coords_coo,
            jac_coords_csr=jac_coords_csr,
            tangent_ordering=tangent_ordering,
            residual_dim=residual_dim_sum,
        )


_residual_dim_cache = dict[Hashable, int]()
_function_cache = dict[Hashable, Callable]()


@jdc.pytree_dataclass
class Factor[*Args]:
    """A single cost in our factor graph."""

    compute_residual: jdc.Static[Callable[[VarValues, *Args], jax.Array]]
    args: tuple[*Args]
    num_variables: jdc.Static[int]
    sorted_ids_from_var_type: dict[type[Var[Any]], jax.Array]
    residual_dim: jdc.Static[int]
    jac_mode: jdc.Static[Literal["auto", "forward", "reverse"]]

    @staticmethod
    def make[*Args_](
        compute_residual: Callable[[VarValues, *Args_], jax.Array],
        args: tuple[*Args_],
        jac_mode: Literal["auto", "forward", "reverse"] = "auto",
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
        return Factor._make_impl(compute_residual, args, jac_mode)

    @staticmethod
    @jdc.jit
    def _make_impl[*Args_](
        compute_residual: jdc.Static[Callable[[VarValues, *Args_], jax.Array]],
        args: tuple[*Args_],
        jac_mode: jdc.Static[Literal["auto", "forward", "reverse"]],
    ) -> Factor[*Args_]:
        """Construct a factor for our factor graph."""

        # TODO: ideally we should get the variables by traversing as a pytree.
        variables = tuple(arg for arg in args if isinstance(arg, Var))

        # Cache the residual dimension for this factor.
        residual_dim_cache_key = (
            compute_residual,
            jax.tree.structure(args),
        )
        if residual_dim_cache_key not in _residual_dim_cache:
            dummy = VarValues.make(variables)
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
            num_variables=len(variables),
            sorted_ids_from_var_type=sort_and_stack_vars(variables),
            residual_dim=_residual_dim_cache[residual_dim_cache_key],
            jac_mode=jac_mode,
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

    def _compute_block_sparse_jac_indices(
        self: Factor,
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
