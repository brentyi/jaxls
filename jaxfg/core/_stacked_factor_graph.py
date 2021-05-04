import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict, Hashable, Iterable, List, Tuple

import jax
import numpy as onp
from jax import numpy as jnp

from .. import sparse, utils
from ..solvers import GaussNewtonSolver, NonlinearSolverBase
from ._factor_base import FactorBase
from ._factor_stack import FactorStack
from ._variable_assignments import StorageMetadata, VariableAssignments
from ._variables import VariableBase

# Key for determining which factors are grouped for stacking
GroupKey = Hashable


@utils.register_dataclass_pytree(
    static_fields=("local_storage_metadata", "residual_dim")
)
@dataclasses.dataclass
class StackedFactorGraph:
    """Dataclass for vectorized factor graph computations.

    Improves runtime by stacking factors based on their group key.
    """

    factor_stacks: List[FactorStack]
    jacobian_coords: sparse.SparseCooCoordinates
    local_storage_metadata: StorageMetadata
    residual_dim: int

    def __post_init__(self):
        """Check that inputs make sense!"""
        for stacked_factor in self.factor_stacks:
            N = stacked_factor.factor.scale_tril_inv.shape[0]
            for value_indices, variable in zip(
                stacked_factor.value_indices, stacked_factor.factor.variables
            ):
                assert value_indices.shape == (N, variable.get_parameter_dim())

    def get_variables(self) -> Iterable[VariableBase]:
        return self.local_storage_metadata.get_variables()

    @staticmethod
    def make(
        factors: Iterable[FactorBase],
        use_onp: bool = True,
    ) -> "StackedFactorGraph":
        """Create a factor graph from a set of factors."""

        # For one-off operations, onp has much less overhead than jnp
        jnp = onp if use_onp else globals()["jnp"]

        # Start by grouping our factors and grabbing a list of (ordered!) variables
        factors_from_group: DefaultDict[GroupKey, List[FactorBase]] = defaultdict(list)
        variables_ordered_set: Dict[VariableBase, None] = {}
        for factor in factors:
            # Each factor is ultimately just a PyTree node; in order for a set of
            # factors to be batchable, they must share the same:
            group_key: GroupKey = (
                # (1) Tree structure: this encompasses the factor class, variable types
                jax.tree_structure(factor),
                # (2) Leaf shapes: array shapes must match in order to be stacked
                tuple(leaf.shape for leaf in jax.tree_leaves(factor)),
            )

            # Record factor and variables
            factors_from_group[group_key].append(factor)
            for v in factor.variables:
                variables_ordered_set[v] = None
        variables = list(variables_ordered_set.keys())

        # Make sure factors are unique
        for factors in factors_from_group.values():
            assert len(factors) == len(set(factors))

        # Fields we want to populate
        stacked_factors: List[FactorStack] = []
        jacobian_coords: List[sparse.SparseCooCoordinates] = []

        # Create storage metadata: this determines which parts of our storage object is
        # allocated to each variable type
        storage_metadata = StorageMetadata.make(variables, local=False)
        local_storage_metadata = StorageMetadata.make(variables, local=True)

        # Prepare each factor group
        residual_offset = 0
        for group_key, group in factors_from_group.items():
            # Make factor stack
            stacked_factors.append(
                FactorStack.make(
                    group,
                    storage_metadata,
                    use_onp=use_onp,
                )
            )

            # Compute Jacobian coordinates
            #
            # These should be N pairs of (row, col) indices, where rows correspond to
            # residual indices and columns correspond to local parameter indices
            jacobian_coords.extend(
                FactorStack.compute_jacobian_coords(
                    factors=group,
                    local_storage_metadata=local_storage_metadata,
                    row_offset=residual_offset,
                )
            )
            residual_offset += stacked_factors[-1].get_residual_dim()

        jacobian_coords_concat: sparse.SparseCooCoordinates = jax.tree_multimap(
            lambda *arrays: onp.concatenate(arrays, axis=0), *jacobian_coords
        )

        return StackedFactorGraph(
            factor_stacks=stacked_factors,
            jacobian_coords=jacobian_coords_concat,
            local_storage_metadata=local_storage_metadata,
            residual_dim=residual_offset,
        )

    @jax.jit
    def compute_residual_vector(self, assignments: VariableAssignments) -> jnp.ndarray:
        """Computes flattened residual vector associated with our factor graph.

        Args:
            assignments (VariableAssignments): Variable assignments.

        Returns:
            jnp.ndarray: Residual vector.
        """

        # Flatten and concatenate residuals from all groups
        residual_vector = jnp.concatenate(
            [
                stacked_factor.compute_residual_vector(assignments).flatten()
                for stacked_factor in self.factor_stacks
            ],
            axis=0,
        )
        assert residual_vector.shape == (self.residual_dim,)
        return residual_vector

    @jax.jit
    def compute_cost(
        self, assignments: VariableAssignments
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the sum of squared residuals associated with a factor graph. Also
        returns intermediate residual vector.

        Args:
            assignments (VariableAssignments): Variable assignments.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Scalar cost, residual vector.
        """
        residual_vector = self.compute_residual_vector(assignments)
        cost = jnp.sum(residual_vector ** 2)
        return cost, residual_vector

    @jax.jit
    def compute_joint_nll(
        self,
        assignments: VariableAssignments,
        include_constants: bool = False,
    ) -> jnp.ndarray:
        """Compute the joint negative log-likelihood density associated with a set of
        variable assignments.

        Args:
            assignments (VariableAssignments): Variable assignments.
            include_constants (bool): Whether or not to include constant terms.

        Returns:
            jnp.ndarray: Scalar cost negative log-likelihood.
        """

        if include_constants:
            raise NotImplementedError()

        # Add Mahalanobis distance terms to NLL
        joint_nll = jnp.sum(self.compute_residual_vector(assignments) ** 2)

        # Add log-determinant terms
        for stacked_factor in self.factor_stacks:
            N, dim, _dim = stacked_factor.factor.scale_tril_inv.shape

            cov_determinants = -2.0 * jnp.log(
                jnp.linalg.det(stacked_factor.factor.scale_tril_inv)
            )
            assert cov_determinants.shape == (N,)

            joint_nll = joint_nll + jnp.sum(cov_determinants)

        return joint_nll

    @jax.jit
    def compute_residual_jacobian(
        self, assignments: VariableAssignments
    ) -> sparse.SparseCooMatrix:
        """Compute the Jacobian of a graph's residual vector with respect to the stacked
        local delta vectors. Shape should be `(residual_dim, local_delta_storage_dim)`."""

        # Linearize factors by group
        A_values_list: List[jnp.ndarray] = []
        for stacked_factor in self.factor_stacks:
            A_values_list.extend(stacked_factor.compute_residual_jacobian(assignments))

        # Build Jacobian
        A = sparse.SparseCooMatrix(
            values=jnp.concatenate([A.flatten() for A in A_values_list]),
            coords=self.jacobian_coords,
            shape=(self.residual_dim, self.local_storage_metadata.dim),
        )
        return A

    @jax.partial(jax.jit, static_argnums=2)
    def _compute_variable_hessian_block(
        self, assignments: VariableAssignments, variable: VariableBase
    ) -> jnp.ndarray:
        """Extract a Hessian block associated with a specific variable, given a set of
        assignments. Should be equivalent to an inverse covariance conditioned on the
        rest of the variables.

        Possible precursor to implementing proper variable elimination, etc...

        TODO: hack for debugging, should revisit. This is currently inefficient for many
        reasons, including that we build out a dense slice of the (sparse) Jacobian and
        square it to compute a much smaller block from the Hessian.
        """
        local_dim = variable.get_local_parameter_dim()
        start_col_index = self.local_storage_metadata.index_from_variable[variable]
        end_col_index = start_col_index + local_dim

        # Construct the full Jacobian, then grab only the columns that we care about
        #
        # Unfortunately it's not possible to JIT compile normal boolean masking because
        # it results in dynamic shapes, so we resort to zeroing out the terms that we
        # don't care about...
        A_all = self.compute_residual_jacobian(assignments)
        mask = jnp.logical_and(
            A_all.coords.cols >= start_col_index, A_all.coords.cols < end_col_index
        )
        A_sliced = sparse.SparseCooMatrix(
            values=jnp.where(
                mask,
                A_all.values,
                jnp.zeros_like(A_all.values),
            ),
            coords=jnp.where(
                mask[:, None],
                A_all.coords - jnp.array([[0, start_col_index]]),
                jnp.zeros_like(A_all.coords),
            ),
            shape=(A_all.shape[0], local_dim),
        )

        # Build out a dense matrix, yikes
        # Note that we add instead of setting to make sure the zero terms don't impact
        # our results
        A_sliced_dense = (
            jnp.zeros(A_sliced.shape)
            .at[A_sliced.coords.rows, A_sliced.coords.cols]
            .add(A_sliced.values)
        )
        hessian_block = A_sliced_dense.T @ A_sliced_dense
        assert hessian_block.shape == (local_dim, local_dim)

        return hessian_block

    def solve(
        self,
        initial_assignments: VariableAssignments,
        solver: NonlinearSolverBase = GaussNewtonSolver(),
    ) -> VariableAssignments:
        """Solve MAP inference problem."""
        return solver.solve(graph=self, initial_assignments=initial_assignments)
