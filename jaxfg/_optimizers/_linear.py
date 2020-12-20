import dataclasses
from typing import TYPE_CHECKING, Tuple, Type

import jax
import jax.numpy as jnp
import numpy as onp

from .. import _types

if TYPE_CHECKING:
    from .._factors import FactorBase
    from .._prepared_factor_graph import PreparedFactorGraph
    from .._variable_assignments import VariableAssignments
    from .._variables import VariableBase


@jax.jit
def gauss_newton_step(
    graph: "PreparedFactorGraph",
    assignments: "VariableAssignments",
    tol: float,
) -> Tuple["VariableAssignments", float]:
    """Single GN step; linearize, then solve linear subproblem.

    Args:
        graph (PreparedFactorGraph): graph
        assignments (VariableAssignments): assignments

    Returns:
        Tuple[VariableAssignments, float]: Updated assignments, error.
    """
    # Linearize factors by group
    A_values_list = []
    for stacked_factors, value_indices in zip(
        graph.stacked_factors,
        graph.value_indices,
    ):
        # Stack inputs to our factors
        values_stacked = tuple(
            assignments.storage[indices] for indices in value_indices
        )

        # Helper for computing Jacobians wrt local parameterizations
        def compute_cost_with_local_delta(
            factor: "FactorBase",
            values: Tuple[_types.VariableValue, ...],
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

    # Solve subproblem
    A_values = jnp.concatenate(A_values_list)
    A_coords = jnp.concatenate(graph.jacobian_coords)
    error_vector = graph.compute_error_vector(assignments)
    local_delta_values = sparse_linear_solve(
        A_values=A_values,
        A_coords=A_coords,
        initial_x=jnp.zeros((graph.local_storage_metadata.dim,)),
        b=-error_vector,
        tol=tol,
    )

    # Update on manifold
    new_storage = jnp.zeros_like(assignments.storage)
    variable_type: Type["VariableBase"]
    for variable_type in assignments.storage_metadata.index_from_variable_type.keys():

        # Get locations
        count = assignments.storage_metadata.count_from_variable_type[variable_type]
        storage_index = assignments.storage_metadata.index_from_variable_type[
            variable_type
        ]
        local_storage_index = graph.local_storage_metadata.index_from_variable_type[
            variable_type
        ]
        dim = variable_type.get_parameter_dim()
        shape = variable_type.get_parameter_shape()
        local_dim = variable_type.get_local_parameter_dim()

        # Get batched variables
        batched_xs = assignments.storage[
            storage_index : storage_index + dim * count
        ].reshape((count,) + shape)
        batched_deltas = local_delta_values[
            local_storage_index : local_storage_index + local_dim * count
        ].reshape((count, local_dim))

        # Batched variable update
        new_storage = new_storage.at[storage_index : storage_index + dim * count].set(
            variable_type.flatten(
                jax.vmap(variable_type.add_local)(batched_xs, batched_deltas)
            ).flatten()
        )

    return (
        dataclasses.replace(assignments, storage=new_storage),
        0.5 * jnp.sum(error_vector ** 2),
    )


def sparse_linear_solve(
    A_values: jnp.ndarray,
    A_coords: jnp.ndarray,
    initial_x: jnp.ndarray,
    b: jnp.ndarray,
    tol: float,
) -> jnp.ndarray:
    """Solves a block-sparse `Ax = b` least squares problem via CGLS."""

    assert len(A_values.shape) == 1, "A_values should be 1D"
    assert A_coords.shape == (
        A_values.shape[0],
        2,
    ), "A_coords should be rows of (row, col)"
    assert len(b.shape) == 1, "b should be 1D!"

    def ATA_function(x: jnp.ndarray):
        # Compute Ax
        Ax = jnp.zeros_like(b).at[A_coords[:, 0]].add(A_values * x[A_coords[:, 1]])

        # Compute ATAx
        ATAx = jnp.zeros_like(x).at[A_coords[:, 1]].add(A_values * Ax[A_coords[:, 0]])

        return ATAx

    # Compute ATb
    ATb = jnp.zeros_like(initial_x).at[A_coords[:, 1]].add(A_values * b[A_coords[:, 0]])

    # Basic Jacobi preconditioning, using diagonals of ATA
    diagonals = jnp.zeros_like(initial_x).at[A_coords[:, 1]].add(A_values ** 2)

    def jacobi_preconditioner(x):
        return x / diagonals

    # Solve with conjugate gradient
    solution_values, _unused_info = jax.scipy.sparse.linalg.cg(
        A=ATA_function,
        b=ATb,
        x0=initial_x,
        maxiter=len(initial_x),  # Default value used by Eigen
        tol=tol,
        M=jacobi_preconditioner,
    )
    return solution_values
