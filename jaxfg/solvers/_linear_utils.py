import dataclasses
from typing import TYPE_CHECKING, Type

import jax
import jax.numpy as jnp
import numpy as onp

from .. import types
from ..core._variable_assignments import VariableAssignments

if TYPE_CHECKING:
    from ..core._factors import FactorBase
    from ..core._prepared_factor_graph import PreparedFactorGraph
    from ..core._variables import VariableBase


@jax.jit
def linearize_graph(
    graph: "PreparedFactorGraph",
    assignments: VariableAssignments,
) -> types.SparseMatrix:
    """Compute the Jacobian of a graph's residual vector with respect to the stacked
    local delta vectors."""

    # Linearize factors by group
    A_values_list = []
    for stacked_factors, value_indices in zip(
        graph.stacked_factors,
        graph.value_indices,
    ):
        # Stack inputs to our factors
        values_stacked = tuple(
            variable.unflatten(assignments.storage[indices])
            for indices, variable in zip(value_indices, stacked_factors.variables)
        )

        # Compute Jacobians wrt local parameterizations
        jacobians = jax.vmap(type(stacked_factors).compute_residual_jacobians)(
            stacked_factors, *values_stacked
        )
        for jacobian in jacobians:
            # Whiten Jacobian, then record flattened values
            A_values_list.append(
                jnp.einsum(
                    "nij,njk->nik", stacked_factors.scale_tril_inv, jacobian
                ).flatten()
            )

    # Build full Jacobian
    A = types.SparseMatrix(
        values=jnp.concatenate(A_values_list),
        coords=jnp.concatenate(graph.jacobian_coords),
        shape=(graph.residual_dim, graph.local_storage_metadata.dim),
    )
    return A


@jax.jit
def apply_local_deltas(
    assignments: VariableAssignments,
    local_delta_assignments: VariableAssignments,
) -> VariableAssignments:
    """Update variables on manifold."""

    new_storage = jnp.zeros_like(assignments.storage)
    variable_type: Type["VariableBase"]
    for variable_type in assignments.storage_metadata.index_from_variable_type.keys():

        # Get locations
        count = assignments.storage_metadata.count_from_variable_type[variable_type]
        storage_index = assignments.storage_metadata.index_from_variable_type[
            variable_type
        ]
        local_storage_index = (
            local_delta_assignments.storage_metadata.index_from_variable_type[
                variable_type
            ]
        )
        dim = variable_type.get_parameter_dim()
        local_dim = variable_type.get_local_parameter_dim()

        # Get batched variables
        batched_xs = assignments.storage[
            storage_index : storage_index + dim * count
        ].reshape((count, dim))
        batched_deltas = local_delta_assignments.storage[
            local_storage_index : local_storage_index + local_dim * count
        ].reshape((count, local_dim))

        # Batched variable update
        new_storage = new_storage.at[storage_index : storage_index + dim * count].set(
            variable_type.flatten(
                jax.vmap(variable_type.add_local)(
                    variable_type.unflatten(batched_xs), batched_deltas
                )
            ).flatten()
        )
    return dataclasses.replace(assignments, storage=new_storage)


@jax.jit
def sparse_linear_solve(
    A: types.SparseMatrix,
    ATb: jnp.ndarray,
    initial_x: jnp.ndarray,
    tol: float,
    lambd: float,
) -> jnp.ndarray:
    """Solves a block-sparse `Ax = b` least squares problem via CGLS.

    More specifically: solves `(A^TA + lambd * diag(A^TA)) x = b` if `diagonal_damping`
    is `True`, otherwise `(A^TA + lambd I) x = b`.

    TODO: consider adding `atol` term back in.
    """

    assert len(A.values.shape) == 1, "A.values should be 1D"
    assert A.coords.shape == (
        A.values.shape[0],
        2,
    ), "A.coords should be rows of (row, col)"
    assert len(ATb.shape) == 1, "ATb should be 1D!"

    # Get diagonals of ATA, for regularization + Jacobi preconditioning
    ATA_diagonals = jnp.zeros_like(initial_x).at[A.coords[:, 1]].add(A.values ** 2)

    # Form normal equation
    def ATA_function(x: jnp.ndarray):
        # Compute ATAx
        ATAx = A.T @ (A @ x)

        # Return regularized (scale-invariant)
        return ATAx + lambd * ATA_diagonals * x

        # Vanilla regularization
        # return ATAx + lambd * x

    def jacobi_preconditioner(x):
        return x / ATA_diagonals

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
