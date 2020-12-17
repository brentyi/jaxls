import dataclasses
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Type, cast

import jax
import numpy as onp
from jax import numpy as jnp
from overrides import overrides
from tqdm.auto import tqdm

from . import _types as types
from . import _utils
from ._factors import FactorBase, LinearFactor
from ._optimizers._nonlinear import GaussNewtonSolver, NonlinearSolver
from ._variable_assignments import VariableAssignments
from ._variables import AbstractRealVectorVariable, VariableBase


@jax.tree_util.register_pytree_node_class
@_utils.immutable_dataclass
class PreparedFactorGraph:
    """Dataclass for vectorized factor graph computations.

    Improves runtime by stacking factors based on their group key.
    """

    stacked_factors: List[FactorBase]
    jacobian_coords: List[jnp.ndarray]
    value_indices: List[
        Tuple[jnp.ndarray, ...]
    ]  # List index: factor #, tuple index: variable #
    local_delta_assignments: VariableAssignments

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

        # Start by grouping our factors, and grabbing a list of variables
        variables: Set[VariableBase] = set()
        factors_from_group: Dict[types.GroupKey, Set[FactorBase]] = {}
        for factor in factors:
            group_key = factor.group_key()
            if group_key not in factors_from_group:
                factors_from_group[group_key] = set()

            assert (
                factor not in factors_from_group[group_key]
            ), "Found a duplicate factor!"
            factors_from_group[group_key].add(factor)
            variables.update(factor.variables)

        # Fields we want to populate
        stacked_factors: List[FactorBase] = []
        jacobian_coords = []
        value_indices: List[Tuple[jnp.ndarray, ...]] = []

        # Create dummy assignments; these tell us how variables are stored
        dummy_assignments = VariableAssignments.create_default(variables)
        local_delta_assignments = dummy_assignments.create_local_deltas()

        # Prepare each factor group
        error_index = 0
        error_indices: List[jnp.ndarray] = []
        for group_key, group in factors_from_group.items():
            # Stack factors in our group
            stacked_factor: FactorBase = jax.tree_multimap(
                lambda *arrays: onp.stack(arrays, axis=0), *group
            )

            # Get indices for each variable
            value_indices_list = tuple([] for _ in range(len(stacked_factor.variables)))
            local_value_indices_list = tuple(
                [] for _ in range(len(stacked_factor.variables))
            )
            for factor in group:
                for i, variable in enumerate(factor.variables):
                    # Record variable parameterization indices
                    storage_pos = dummy_assignments.storage_pos_from_variable[variable]
                    value_indices_list[i].append(
                        onp.arange(
                            storage_pos, storage_pos + variable.get_parameter_dim()
                        ).reshape(variable.get_parameter_shape())
                    )

                    # Record local parameterization indices
                    storage_pos = local_delta_assignments.storage_pos_from_variable[
                        variable
                    ]
                    local_value_indices_list[i].append(
                        onp.arange(
                            storage_pos,
                            storage_pos + variable.get_local_parameter_dim(),
                        )
                    )

            # Stack: end result should be Tuple[array of shape (N, *parameter_shape), ...]
            value_indices_list = tuple(onp.array(l) for l in value_indices_list)
            local_value_indices_list = tuple(
                onp.array(l) for l in local_value_indices_list
            )

            # Record values
            stacked_factors.append(stacked_factor)
            value_indices.append(value_indices_list)
            error_indices.append(
                onp.arange(
                    error_index, error_index + len(group) * factor.error_dim
                ).reshape((len(group), factor.error_dim))
            )
            error_index += stacked_factor.error_dim * len(group)

            # Get Jacobian coordinates
            num_factors = len(group)
            error_dim = stacked_factor.error_dim
            for variable_index, variable in enumerate(stacked_factor.variables):
                variable_dim = variable.get_local_parameter_dim()

                coords = onp.stack(
                    (
                        # Row indices
                        onp.broadcast_to(
                            error_indices[-1][:, :, None],
                            (num_factors, error_dim, variable_dim),
                        ),
                        # Column indices
                        onp.broadcast_to(
                            local_value_indices_list[variable_index][:, None, :],
                            (num_factors, error_dim, variable_dim),
                        ),
                    ),
                    axis=-1,
                ).reshape((num_factors * error_dim * variable_dim, 2))

                jacobian_coords.append(coords)

        return PreparedFactorGraph(
            stacked_factors=stacked_factors,
            jacobian_coords=jacobian_coords,
            value_indices=value_indices,
            local_delta_assignments=local_delta_assignments,
        )

    def compute_error_vector(self, assignments: VariableAssignments) -> jnp.ndarray:
        """Computes error vector associated with our factor graph.

        Args:
            assignments (VariableAssignments): Variable assignments.

        Returns:
            jnp.ndarray: Error vector.
        """

        # Get error vector associated with each group
        errors_list: List[jnp.ndarray] = []
        for stacked_factors, value_indices in zip(
            self.stacked_factors,
            self.value_indices,
        ):
            # Stack inputs to our factor
            values_stacked = tuple(
                assignments.storage[indices] for indices in value_indices
            )

            # Vectorized error computation
            errors_list.append(
                jnp.einsum(
                    "nij,nj->ni",
                    stacked_factors.scale_tril_inv,
                    jax.vmap(type(stacked_factors).compute_error)(
                        stacked_factors, *values_stacked
                    ),
                ).flatten()
            )

        # Concatenate errors from all groups
        return jnp.concatenate(errors_list, axis=0)

    def solve(
        self,
        initial_assignments: VariableAssignments,
        solver: NonlinearSolver = GaussNewtonSolver(),
    ) -> VariableAssignments:
        """Solve MAP inference problem."""
        return solver.solve(graph=self, initial_assignments=initial_assignments)

    def tree_flatten(v: "PreparedFactorGraph") -> Tuple[Tuple[jnp.ndarray], Tuple]:
        """Flatten a factor for use as a PyTree/parameter stacking."""
        v_dict = vars(v)
        array_data = {k: v for k, v in v_dict.items()}
        return (tuple(array_data.values()), tuple(array_data.keys()))

    @classmethod
    def tree_unflatten(
        cls, treedef: Tuple, children: Tuple[jnp.ndarray]
    ) -> "PreparedFactorGraph":
        """Unflatten a factor for use as a PyTree/parameter stacking."""
        array_keys = treedef
        return cls(
            **dict(zip(array_keys, children)),
        )
