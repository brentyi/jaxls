import dataclasses
from typing import Dict, Iterable, List, Optional, Set, Tuple, Type, cast

import jax
from jax import numpy as jnp

from .. import types as types
from .. import utils
from ..solvers import GaussNewtonSolver, NonlinearSolverBase
from ._factors import FactorBase
from ._variable_assignments import StorageMetadata, VariableAssignments
from ._variables import VariableBase


@jax.partial(
    utils.register_dataclass_pytree,
    static_fields=("local_storage_metadata", "residual_dim"),
)
@dataclasses.dataclass(frozen=True)
class PreparedFactorGraph:
    """Dataclass for vectorized factor graph computations.

    Improves runtime by stacking factors based on their group key.
    """

    stacked_factors: List[FactorBase]
    jacobian_coords: List[jnp.ndarray]
    value_indices: List[
        Tuple[jnp.ndarray, ...]
    ]  # List index: factor #, tuple index: variable #
    local_storage_metadata: StorageMetadata
    residual_dim: int

    def __post_init__(self):
        """Check that inputs make sense!"""
        assert len(self.stacked_factors) == len(self.value_indices)
        for s, v in zip(self.stacked_factors, self.value_indices):
            N = s.scale_tril_inv.shape[0]
            variable: VariableBase
            for value_indices, variable in zip(v, s.variables):
                assert value_indices.shape == (N, variable.get_parameter_dim())

    @staticmethod
    def from_factors(factors: Iterable[FactorBase]) -> "PreparedFactorGraph":
        """Create a factor graph from a set of factors.

        Args:
            factors (Iterable[FactorBase]): Factors in our graph.

        Returns:
            "PreparedFactorGraph":
        """

        # Start by grouping our factors and grabbing a list of variables
        variables: Set[VariableBase] = set()
        factors_from_group: Dict[types.GroupKey, List[FactorBase]] = {}
        for factor in factors:
            group_key = factor.group_key()
            if group_key not in factors_from_group:
                factors_from_group[group_key] = []

            factors_from_group[group_key].append(factor)
            variables.update(factor.variables)

        # Make sure factors are unique
        for factors in factors_from_group.values():
            assert len(factors) == len(set(factors))

        # Fields we want to populate
        stacked_factors: List[FactorBase] = []
        jacobian_coords = []
        value_indices: List[Tuple[jnp.ndarray, ...]] = []

        # Create dummy assignments; these tell us how variables are stored
        storage_metadata = StorageMetadata.from_variables(variables, local=False)
        delta_storage_metadata = StorageMetadata.from_variables(variables, local=True)

        # Prepare each factor group
        residual_index = 0
        residual_indices: List[jnp.ndarray] = []
        for group_key, group in factors_from_group.items():
            # Stack factors in our group
            stacked_factor: FactorBase = jax.tree_multimap(
                lambda *arrays: jnp.stack(arrays, axis=0), *group
            )

            # Get indices for each variable
            value_indices_list: Tuple[List[jnp.ndarray], ...] = tuple(
                [] for _ in range(len(stacked_factor.variables))
            )
            local_value_indices_list: Tuple[List[jnp.ndarray], ...] = tuple(
                [] for _ in range(len(stacked_factor.variables))
            )
            for factor in group:
                for i, variable in enumerate(factor.variables):
                    # Record variable parameterization indices
                    storage_pos = storage_metadata.index_from_variable[variable]
                    value_indices_list[i].append(
                        jnp.arange(
                            storage_pos, storage_pos + variable.get_parameter_dim()
                        )
                    )

                    # Record local parameterization indices
                    storage_pos = delta_storage_metadata.index_from_variable[variable]
                    local_value_indices_list[i].append(
                        jnp.arange(
                            storage_pos,
                            storage_pos + variable.get_local_parameter_dim(),
                        )
                    )

            # Stack: end result should be Tuple[array of shape (N, *parameter_shape), ...]
            value_indices_stacked: Tuple[jnp.ndarray] = tuple(
                jnp.array(l) for l in value_indices_list
            )
            local_value_indices_stacked: Tuple[jnp.ndarray] = tuple(
                jnp.array(l) for l in local_value_indices_list
            )

            # Record values
            stacked_factors.append(stacked_factor)
            value_indices.append(value_indices_stacked)
            residual_indices.append(
                jnp.arange(
                    residual_index, residual_index + len(group) * factor.residual_dim
                ).reshape((len(group), factor.residual_dim))
            )
            residual_index += stacked_factor.residual_dim * len(group)

            # Get Jacobian coordinates
            num_factors = len(group)
            residual_dim = stacked_factor.residual_dim
            for variable_index, variable in enumerate(stacked_factor.variables):
                variable_dim = variable.get_local_parameter_dim()

                coords = jnp.stack(
                    (
                        # Row indices
                        jnp.broadcast_to(
                            residual_indices[-1][:, :, None],
                            (num_factors, residual_dim, variable_dim),
                        ),
                        # Column indices
                        jnp.broadcast_to(
                            local_value_indices_stacked[variable_index][:, None, :],
                            (num_factors, residual_dim, variable_dim),
                        ),
                    ),
                    axis=-1,
                ).reshape((num_factors * residual_dim * variable_dim, 2))

                jacobian_coords.append(coords)

        return PreparedFactorGraph(
            stacked_factors=stacked_factors,
            jacobian_coords=jacobian_coords,
            value_indices=value_indices,
            local_storage_metadata=delta_storage_metadata,
            residual_dim=residual_index,
        )

    def compute_residual_vector(self, assignments: VariableAssignments) -> jnp.ndarray:
        """Computes residual vector associated with our factor graph.

        Args:
            assignments (VariableAssignments): Variable assignments.

        Returns:
            jnp.ndarray: Residual vector.
        """

        # Get residual vector associated with each group
        residuals_list: List[jnp.ndarray] = []
        for stacked_factors, value_indices in zip(
            self.stacked_factors,
            self.value_indices,
        ):
            # Stack inputs to our factor
            variable: VariableBase
            values_stacked = tuple(
                jax.vmap(type(variable).unflatten)(assignments.storage[indices])
                for variable, indices in zip(stacked_factors.variables, value_indices)
            )

            # Vectorized residual computation
            residuals_list.append(
                jnp.einsum(
                    "nij,nj->ni",
                    stacked_factors.scale_tril_inv,
                    jax.vmap(type(stacked_factors).compute_residual_vector)(
                        stacked_factors, *values_stacked
                    ),
                ).flatten()
            )

        # Concatenate residuals from all groups
        residual_vector = jnp.concatenate(residuals_list, axis=0)
        assert residual_vector.shape == (self.residual_dim,)
        return residual_vector

    @jax.jit
    def compute_cost(
        self, assignments: VariableAssignments
    ) -> Tuple[float, jnp.ndarray]:
        """Compute the sum of squared residuals associated with a factor graph. Also
        returns intermediate residual vector.

        Args:
            assignments (VariableAssignments): Variable assignments.

        Returns:
            Tuple[float, jnp.ndarray]: Scalar cost, residual vector.
        """
        residual_vector = self.compute_residual_vector(assignments)
        cost = jnp.sum(residual_vector ** 2)
        return cost, residual_vector

    def solve(
        self,
        initial_assignments: VariableAssignments,
        solver: NonlinearSolverBase = GaussNewtonSolver(),
    ) -> VariableAssignments:
        """Solve MAP inference problem."""
        return solver.solve(graph=self, initial_assignments=initial_assignments)
