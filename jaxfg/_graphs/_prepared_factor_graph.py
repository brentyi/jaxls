import dataclasses
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Type, cast

import jax
import numpy as onp
from jax import numpy as jnp
from overrides import overrides
from tqdm.auto import tqdm

from .. import _types as types
from .. import _utils
from .._factors import FactorBase, LinearFactor
from .._optimizers._linear_solver import sparse_linear_solve
from .._variables import AbstractRealVectorVariable, VariableBase
from ._factor_graph_base import FactorGraphBase
from ._linear_factor_graph import LinearFactorGraph


@jax.tree_util.register_pytree_node_class
@_utils.immutable_dataclass
class PreparedFactorGraph:
    stacked_factors: List[FactorBase]
    jacobian_coords: List[jnp.ndarray]
    value_indices: List[
        Tuple[jnp.ndarray, ...]
    ]  # List index: factor #, tuple index: variable #
    error_indices: List[jnp.ndarray]
    local_delta_assignments: types.VariableAssignments

    def __post_init__(self):
        """Check that shapes of inputs make sense!"""
        assert (
            len(self.stacked_factors)
            == len(self.value_indices)
            == len(self.error_indices)
        )
        for s, v, e in zip(
            self.stacked_factors, self.value_indices, self.error_indices
        ):
            N = e.shape[0]
            assert e.shape == (N, s.error_dim), f"{e.shape} {s.error_dim} {N}"
            variable: VariableBase
            for value_indices, variable in zip(v, s.variables):
                assert value_indices.shape == (N, variable.get_parameter_dim())

    def solve(
        self, initial_assignments: types.VariableAssignments
    ) -> types.VariableAssignments:

        assignments = initial_assignments

        # Run some Gauss-Newton iterations
        # for i in range(10):
        error_tol = 1e-3
        max_iters = 10000
        prev_error = float("inf")
        for i in range(max_iters):
            assignments, error = self._gauss_newton_step(assignments)
            print(f"{i}: {error}")
            if onp.abs(prev_error - error) < error_tol:
                break
            prev_error = error

        return assignments

    def _gauss_newton_step(
        self, assignments: types.VariableAssignments
    ) -> Tuple[types.VariableAssignments, float]:
        # Linearize factors by group
        A_values_list = []
        errors_list = []
        for stacked_factors, value_indices in zip(
            self.stacked_factors,
            self.value_indices,
        ):
            # Stack inputs to our factors
            values_stacked = tuple(
                assignments.storage[indices] for indices in value_indices
            )

            # Helper for computing Jacobians wrt local parameterizations
            def compute_cost_with_local_delta(
                factor: FactorBase,
                values: Tuple[jnp.ndarray],
                local_deltas: Tuple[jnp.ndarray],
            ):
                variable_type: Type["VariableBase"]
                perturbed_values = [
                    variable.add_local(
                        x=x.reshape(variable.get_parameter_shape()),
                        local_delta=local_delta,
                    )
                    for variable, x, local_delta in zip(
                        factor.variables, values, local_deltas
                    )
                ]
                return factor.compute_error(*perturbed_values)

            # Vectorized Jacobian computation
            num_factors = stacked_factors.scale_tril_inv.shape[0]
            jacobians = jax.vmap(jax.jacfwd(compute_cost_with_local_delta, argnums=2))(
                stacked_factors,
                values_stacked,
                tuple(
                    onp.zeros((num_factors, variable.get_local_parameter_dim()))
                    for variable in stacked_factors.variables
                ),
            )
            for jacobian in jacobians:
                # Whiten Jacobian, then record flattened values
                A_values_list.append(
                    jnp.einsum(
                        "nij,njk->nik", stacked_factors.scale_tril_inv, jacobian
                    ).flatten()
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

        # Solve subproblem
        A_values = jnp.concatenate(A_values_list)
        A_coords = onp.concatenate(self.jacobian_coords)
        error_vector = jnp.concatenate(errors_list, axis=0)
        local_delta_values = sparse_linear_solve(
            A_values=A_values,
            A_coords=A_coords,
            initial_x=self.local_delta_assignments.storage,  # This is just all zeros
            b=-error_vector,
        )

        # Update on manifold
        new_storage = jnp.zeros_like(assignments.storage)
        variable_type: Type[VariableBase]
        for variable_type in assignments.storage_pos_from_variable_type.keys():

            # Get locations
            count = assignments.count_from_variable_type[variable_type]
            storage_pos = assignments.storage_pos_from_variable_type[variable_type]
            local_storage_pos = (
                self.local_delta_assignments.storage_pos_from_variable_type[
                    variable_type
                ]
            )
            dim = variable_type.get_parameter_dim()
            shape = variable_type.get_parameter_shape()
            local_dim = variable_type.get_local_parameter_dim()

            # Get batched variables
            batched_xs = assignments.storage[
                storage_pos : storage_pos + dim * count
            ].reshape((count,) + shape)
            batched_deltas = local_delta_values[
                local_storage_pos : local_storage_pos + local_dim * count
            ].reshape((count, local_dim))

            # Batched variable update
            new_storage = new_storage.at[storage_pos : storage_pos + dim * count].set(
                jax.vmap(variable_type.add_local)(batched_xs, batched_deltas).flatten()
            )

        return (
            dataclasses.replace(assignments, storage=new_storage),
            0.5 * jnp.sum(error_vector ** 2),
        )

    def tree_flatten(v: "PreparedFactorGraph") -> Tuple[Tuple[jnp.ndarray], Tuple]:
        """Flatten a factor for use as a PyTree/parameter stacking."""
        v_dict = dataclasses.asdict(v)
        array_data = {k: v for k, v in v_dict.items()}
        return (tuple(array_data.values()), tuple(array_data.keys()))

    @classmethod
    def tree_unflatten(
        cls, treedef: Tuple, children: Tuple[jnp.ndarray]
    ) -> "PreparedFactorGraph":
        """Unflatten a factor for use as a PyTree/parameter stacking."""
        array_keys = treedef[: len(children)]
        aux = treedef[len(children) :]
        aux_keys = aux[: len(aux) // 2]
        aux_values = aux[len(aux) // 2 :]

        # Create new dummy variables
        aux_dict = dict(zip(aux_keys, aux_values))
        aux_dict["variables"] = tuple(V() for V in aux_dict.pop("variable_types"))

        return cls(
            # variables=tuple(),
            **dict(zip(array_keys, children)),
            **aux_dict,
        )
