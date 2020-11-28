import dataclasses
from typing import Dict, Iterable, Optional, Set, Tuple, Type, cast

import jax
import numpy as onp
from jax import numpy as jnp
from overrides import overrides
from tqdm.auto import tqdm

from .. import _types as types
from .. import _utils
from .._factors import FactorBase, LinearFactor
from .._variables import AbstractRealVectorVariable, VariableBase
from ._factor_graph_base import FactorGraphBase
from ._linear_factor_graph import LinearFactorGraph


@_utils.immutable_dataclass
class FactorGraph(FactorGraphBase[FactorBase, VariableBase]):
    """General nonlinear factor graph."""

    def solve(
        self,
        initial_assignments: Optional[types.VariableAssignments] = None,
    ) -> types.VariableAssignments:

        # Make default initial assignments if unavailable
        if initial_assignments is None:
            assignments = types.VariableAssignments.create_default(self.variables)
        else:
            assignments = initial_assignments
        assert set(assignments.variables) == set(self.variables)

        # Run some Gauss-Newton iterations
        # for i in range(10):
        error_tol = 1e-3
        max_iters = 10000
        prev_error = float("inf")
        for i in range(max_iters):
            # print("Computing error")
            # print(
            #     onp.sum(
            #         [
            #             ( f.scale_tril_inv
            #                 @ f.compute_error(
            #                     *[
            #                         assignments.get_value(variable)
            #                         for variable in f.variables
            #                     ]
            #                 )
            #             )
            #             ** 2
            #             for f in self.factors
            #         ]
            #     )
            # )
            # print(assignments.storage)
            # with _utils.stopwatch("GN step"):
            assignments, error = self._gauss_newton_step(assignments)
            print(f"{i}: {error}")
            if onp.abs(prev_error - error) < error_tol:
                break
            prev_error = error
            # assignments.storage.block_until_ready()

        return assignments

    @jax.partial(jax.jit, static_argnums=0)
    def _gauss_newton_step(
        self, assignments: types.VariableAssignments
    ) -> types.VariableAssignments:
        # Create storage object with local deltas
        local_delta_assignments = assignments.create_local_deltas()

        A_matrices_from_shape = {}
        value_indices_from_shape = {}
        error_indices_from_shape = {}

        # Assign each factor to a place on our error vector
        # This can be moved
        error_indices_from_factor = {}
        error_index = 0
        for factor in self.factors:
            error_indices_from_factor[factor] = onp.arange(
                error_index, error_index + factor.error_dim
            )
            error_index += factor.error_dim

        # Linearize factors by group
        errors_list = []
        for group_key, group in self.factors_from_group.items():
            # Get some info about our group
            num_factors = len(group)
            example_factor = next(iter(group))
            variable_types: Iterable[Type["VariableBase"]] = (
                type(v) for v in example_factor.variables
            )

            # Helper for computing Jacobians wrt local parameterizations
            def compute_cost_with_local_delta(
                factor: FactorBase,
                values: Tuple[jnp.ndarray],
                local_deltas: Tuple[jnp.ndarray],
            ):
                variable_type: Type["VariableBase"]
                perturbed_values = [
                    variable_type.add_local(
                        x=x.reshape(variable_type.get_parameter_shape()),
                        local_delta=local_delta,
                    )
                    for variable_type, x, local_delta in zip(
                        variable_types, values, local_deltas
                    )
                ]
                return factor.compute_error(*perturbed_values)

            # Stack factors in our group
            factors_stacked: FactorBase = jax.tree_multimap(
                lambda *arrays: onp.stack(arrays, axis=0), *group
            )

            # Stack inputs to our factors
            values_indices = tuple([] for _ in range(len(example_factor.variables)))
            for factor in group:
                for i, variable in enumerate(factor.variables):
                    storage_pos = assignments.storage_pos_from_variable[variable]
                    values_indices[i].append(
                        onp.arange(
                            storage_pos, storage_pos + variable.get_parameter_dim()
                        ).reshape(variable.get_parameter_shape())
                    )
            values_stacked = tuple(
                assignments.storage[onp.array(indices)] for indices in values_indices
            )

            # Vectorized Jacobian computation
            jacobians = jax.vmap(jax.jacfwd(compute_cost_with_local_delta, argnums=2))(
                factors_stacked,
                values_stacked,
                tuple(
                    onp.zeros((num_factors, variable.get_local_parameter_dim()))
                    for variable in example_factor.variables
                ),
            )

            # Vectorized error computation
            errors_list.append(
                jnp.einsum(
                    "nij,nj->ni",
                    factors_stacked.scale_tril_inv,
                    jax.vmap(group_key.factor_type.compute_error)(
                        factors_stacked, *values_stacked
                    ),
                ).flatten()
            )

            # Put Jacobians into input array
            for variable_index, jacobian in enumerate(jacobians):
                # Get shape of Jacobian w/o batch dimension
                shape = jacobian.shape[1:]
                if shape not in A_matrices_from_shape:
                    A_matrices_from_shape[shape] = []
                    value_indices_from_shape[shape] = []
                    error_indices_from_shape[shape] = []

                A_matrices_from_shape[shape].append(
                    jnp.einsum("nij,njk->nik", factors_stacked.scale_tril_inv, jacobian)
                )

                value_indices_row = []
                error_indices_row = []
                for factor in group:
                    variable = factor.variables[variable_index]
                    storage_pos = local_delta_assignments.storage_pos_from_variable[
                        variable
                    ]
                    value_indices_row.append(
                        onp.arange(
                            storage_pos,
                            storage_pos + variable.get_local_parameter_dim(),
                        )
                    )
                    error_indices_row.append(error_indices_from_factor[factor])

                value_indices_from_shape[shape].append(value_indices_row)
                error_indices_from_shape[shape].append(error_indices_row)

        # Stack matrices together
        A_matrices_from_shape = {
            k: jnp.concatenate(v) for k, v in A_matrices_from_shape.items()
        }
        value_indices_from_shape = {
            k: onp.concatenate(v) for k, v in value_indices_from_shape.items()
        }
        error_indices_from_shape = {
            k: onp.concatenate(v) for k, v in error_indices_from_shape.items()
        }
        error_vector = -jnp.concatenate(errors_list)

        local_delta_values = LinearFactorGraph._solve(
            A_matrices_from_shape,
            value_indices_from_shape,
            error_indices_from_shape,
            initial_x=local_delta_assignments.storage,
            b=error_vector,
        )

        # Update on manifold
        new_storage = jnp.zeros_like(assignments.storage)
        variable_type: Type[VariableBase]
        for variable_type in assignments.storage_pos_from_variable_type.keys():

            # Get locations
            count = assignments.count_from_variable_type[variable_type]
            storage_pos = assignments.storage_pos_from_variable_type[variable_type]
            local_storage_pos = local_delta_assignments.storage_pos_from_variable_type[
                variable_type
            ]
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
