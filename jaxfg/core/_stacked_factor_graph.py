import dataclasses
from typing import (
    Dict,
    Generic,
    Hashable,
    Iterable,
    List,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import jax
import numpy as onp
from jax import numpy as jnp

from .. import types as types
from .. import utils
from ..solvers import GaussNewtonSolver, NonlinearSolverBase
from ._factors import FactorBase
from ._variable_assignments import StorageMetadata, VariableAssignments
from ._variables import VariableBase

FactorType = TypeVar("FactorType", bound=FactorBase)

GroupKey = Hashable


@jax.partial(utils.register_dataclass_pytree, static_fields=("num_factors",))
@dataclasses.dataclass
class FactorStack(Generic[FactorType]):
    """A set of factors, with their parameters stacked."""

    num_factors: int
    factor: FactorType
    value_indices: Tuple[types.Array, ...]

    def __post_init__(self):
        # There should be on set of indices for each variable type
        assert len(self.value_indices) == len(self.factor.variables)

        # Check that shapes make sense
        for variable, indices in zip(self.factor.variables, self.value_indices):
            N, res_dim0, res_dim1 = self.factor.scale_tril_inv.shape
            assert N == self.num_factors
            assert indices.shape == (
                N,
                variable.get_parameter_dim(),
            )
            assert res_dim0 == res_dim1 == self.factor.get_residual_dim()

    @staticmethod
    def make(
        factors: Sequence[FactorType],
        storage_metadata: StorageMetadata,
    ) -> "FactorStack[FactorType]":
        """Make a stacked factor."""
        # Stack factors in our group
        stacked_factor: FactorType = jax.tree_multimap(
            lambda *arrays: jnp.stack(arrays, axis=0), *factors
        )

        # Get indices for each variable of each factor
        value_indices_list: Tuple[List[onp.ndarray], ...] = tuple(
            [] for _ in range(len(stacked_factor.variables))
        )
        for factor in factors:
            for i, variable in enumerate(factor.variables):
                storage_pos = storage_metadata.index_from_variable[variable]
                value_indices_list[i].append(
                    onp.arange(storage_pos, storage_pos + variable.get_parameter_dim())
                )

        # Stack: end result should be Tuple[array of shape (N, parameter_dim), ...]
        value_indices_stacked: Tuple[onp.ndarray, ...] = tuple(
            onp.array(indices) for indices in value_indices_list
        )

        # Record values
        return FactorStack(
            num_factors=len(factors),
            factor=stacked_factor,
            value_indices=value_indices_stacked,
        )

    @staticmethod
    def compute_jacobian_coords(
        factors: Sequence[FactorType],
        local_storage_metadata: StorageMetadata,
    ) -> List[jnp.ndarray]:
        """Computes Jacobian coordinates for a factor stack. One array of indices per
        variable."""
        variable_types: List[Type[VariableBase]] = [
            type(v) for v in factors[0].variables
        ]

        # Get indices for each variable
        local_value_indices_list: Tuple[List[onp.ndarray], ...] = tuple(
            [] for _ in range(len(variable_types))
        )
        for factor in factors:
            for i, variable_type in enumerate(factor.variables):
                # Record local parameterization indices
                storage_pos = local_storage_metadata.index_from_variable[variable_type]
                local_value_indices_list[i].append(
                    onp.arange(
                        storage_pos,
                        storage_pos + variable_type.get_local_parameter_dim(),
                    )
                )

            # Stack: end result should be Tuple[array of shape (N, parameter_dim), ...]
            local_value_indices_stacked: Tuple[onp.ndarray, ...] = tuple(
                onp.array(indices) for indices in local_value_indices_list
            )

        # Get residual indices
        num_factors = len(factors)
        residual_dim = factors[0].get_residual_dim()
        residual_indices = onp.arange(num_factors * residual_dim).reshape(
            (num_factors, residual_dim)
        )

        # Get Jacobian coordinates
        jacobian_coords: List[jnp.ndarray] = []
        for variable_index, variable_type in enumerate(variable_types):
            variable_dim = variable_type.get_local_parameter_dim()

            coords = onp.stack(
                (
                    # Row indices
                    onp.broadcast_to(
                        residual_indices[:, :, None],
                        (num_factors, residual_dim, variable_dim),
                    ),
                    # Column indices
                    onp.broadcast_to(
                        local_value_indices_stacked[variable_index][:, None, :],
                        (num_factors, residual_dim, variable_dim),
                    ),
                ),
                axis=-1,
            ).reshape((num_factors * residual_dim * variable_dim, 2))

            jacobian_coords.append(coords)

        return jacobian_coords

    def get_residual_dim(self) -> int:
        return self.factor.get_residual_dim() * self.num_factors

    def compute_residual_vector(self, assignments: VariableAssignments) -> jnp.ndarray:
        """Compute a stacked residual vector.

        Shape of output should be `(N, stacked_factor.factor.get_residual_dim())`.
        """

        # Stack inputs to our factors
        values_stacked = tuple(
            jax.vmap(type(variable).unflatten)(assignments.storage[indices])
            for variable, indices in zip(self.factor.variables, self.value_indices)
        )

        # Vectorized residual computation
        residual_vector = jnp.einsum(
            "nij,nj->ni",
            self.factor.scale_tril_inv,
            jax.vmap(type(self.factor).compute_residual_vector)(
                self.factor, *values_stacked
            ),
        )

        assert residual_vector.shape == (
            self.factor.scale_tril_inv.shape[0],
            self.factor.get_residual_dim(),
        )
        return residual_vector

    def compute_residual_jacobian(
        self, assignments: VariableAssignments
    ) -> List[jnp.ndarray]:
        """Compute stacked and whitened Jacobian matrices, one for each variable.

        Shape of each Jacobian array should be `(N, local parameter dim, residual dim)`.
        """

        # Stack inputs to our factors
        values_stacked = tuple(
            jax.vmap(variable.unflatten)(assignments.storage[indices])
            for indices, variable in zip(self.value_indices, self.factor.variables)
        )

        # Compute Jacobians wrt local parameterizations
        jacobians = jax.vmap(type(self.factor).compute_residual_jacobians)(
            self.factor, *values_stacked
        )

        # Whiten Jacobians
        return [
            jnp.einsum("nij,njk->nik", self.factor.scale_tril_inv, jacobian)
            for jacobian in jacobians
        ]


@jax.partial(
    utils.register_dataclass_pytree,
    static_fields=("local_storage_metadata", "residual_dim"),
)
@dataclasses.dataclass
class StackedFactorGraph:
    """Dataclass for vectorized factor graph computations.

    Improves runtime by stacking factors based on their group key.
    """

    factor_stacks: List[FactorStack]
    jacobian_coords: types.Array
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

    def variables(self) -> Iterable[VariableBase]:
        return self.local_storage_metadata.ordered_variables()

    @staticmethod
    def make(factors: Iterable[FactorBase]) -> "StackedFactorGraph":
        """Create a factor graph from a set of factors.

        Args:
            factors (Iterable[FactorBase]): Factors in our graph.

        Returns:
            "StackedFactorGraph":
        """

        # Start by grouping our factors and grabbing a list of (ordered!) variables
        variables_ordered_set: Dict[VariableBase, None] = {}
        factors_from_group: Dict[GroupKey, List[FactorBase]] = {}
        for factor in factors:
            # In order to batch factor computations...
            group_key: GroupKey = (
                # (1) Factor types should match
                factor.__class__,
                # (2) Variable types and parameter dimensions should match
                tuple((type(v), v.get_parameter_dim()) for v in factor.variables),
                # (3) Residual dimension should match
                factor.get_residual_dim(),
            )
            if group_key not in factors_from_group:
                factors_from_group[group_key] = []

            factors_from_group[group_key].append(factor)
            for v in factor.variables:
                variables_ordered_set[v] = None
        variables = list(variables_ordered_set.keys())

        # Make sure factors are unique
        for factors in factors_from_group.values():
            assert len(factors) == len(set(factors))

        # Fields we want to populate
        stacked_factors: List[FactorStack] = []
        jacobian_coords: List[jnp.ndarray] = []

        # Create storage metadata: this determines which parts of our storage object is
        # allocated to each variable type
        storage_metadata = StorageMetadata.make(variables, local=False)
        local_storage_metadata = StorageMetadata.make(variables, local=True)

        # Prepare each factor group
        residual_offset = 0
        for group_key, group in factors_from_group.items():
            # Make factor stack
            stacked_factors.append(FactorStack.make(group, storage_metadata))

            # Compute Jacobian coordinates
            #
            # These should be N pairs of (row, col) indices, where rows correspond to
            # residual indices and columns correspond to local parameter indices
            jacobian_coords.extend(
                [
                    coords + onp.array([[residual_offset, 0]])
                    for coords in FactorStack.compute_jacobian_coords(
                        group, local_storage_metadata
                    )
                ]
            )
            residual_offset += stacked_factors[-1].get_residual_dim()

        return StackedFactorGraph(
            factor_stacks=stacked_factors,
            jacobian_coords=jnp.concatenate(jacobian_coords, axis=0),
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

            cov_determinants = jnp.log(
                jnp.linalg.det(stacked_factor.factor.scale_tril_inv) ** (-2)
            )
            assert cov_determinants.shape == (N,)

            joint_nll = joint_nll + jnp.sum(cov_determinants)

        return joint_nll

    @jax.jit
    def compute_residual_jacobian(
        self, assignments: VariableAssignments
    ) -> types.SparseMatrix:
        """Compute the Jacobian of a graph's residual vector with respect to the stacked
        local delta vectors. Shape should be `(residual_dim, local_delta_storage_dim)`."""

        # Linearize factors by group
        A_values_list: List[jnp.ndarray] = []
        for stacked_factor in self.factor_stacks:
            A_values_list.extend(stacked_factor.compute_residual_jacobian(assignments))

        # Build Jacobian
        A = types.SparseMatrix(
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
            A_all.coords[:, 1] >= start_col_index, A_all.coords[:, 1] < end_col_index
        )
        A_sliced = types.SparseMatrix(
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
            .at[A_sliced.coords[:, 0], A_sliced.coords[:, 1]]
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
