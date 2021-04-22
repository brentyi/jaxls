import dataclasses
from typing import Dict, List, Sequence, Tuple

import numpy as onp
import scipy
import sksparse

from .. import core


@dataclasses.dataclass
class SparseCovariance:
    """Helper class for recovering marginal covariances. Implements the algorithm
    described in [1].

    [1] Covariance Recovery from a Square Root Information Matrix for Data Association
    http://www.cs.cmu.edu/~kaess/pub/Kaess09ras.pdf
    """

    L: scipy.sparse.csc_matrix
    L_diag_inv: onp.ndarray
    local_storage_metadata: core.StorageMetadata

    _value_cache: Dict[Tuple[int, int], float]

    @staticmethod
    def make(graph: core.StackedFactorGraph, assignments: core.VariableAssignments):
        """Build the sparse covariance corresponding to a factor graph.

        Computed by linearizing the graph around a set of variable assignments and
        computing a square-root information matrix."""

        A: scipy.sparse.csc_matrix = (
            graph.compute_residual_jacobian(assignments)
            .T.as_scipy_coo_matrix()
            .tocsc(copy=False)
        )
        sqrt_information_matrix: scipy.sparse.csc_matrix = (
            sksparse.cholmod.cholesky_AAt(A=A).L()
        )
        return SparseCovariance(
            L=sqrt_information_matrix,
            L_diag_inv=1.0 / sqrt_information_matrix.diagonal(),
            local_storage_metadata=graph.local_storage_metadata,
            _value_cache={},
        )

    def as_dense(self, use_inverse: bool = True) -> onp.ndarray:
        """Return the full covariance as a dense array. Should only be used for
        debugging in small problems."""
        if use_inverse:
            L_inv = onp.linalg.inv(self.L.todense())
            return L_inv.T @ L_inv
        else:
            return self._compute_marginal(range(self.local_storage_metadata.dim))

    def compute_marginal(self, *variables: core.VariableBase) -> onp.ndarray:
        """Compute marginal covariance for a set of variables. Input order matters.

        Output will be a square matrix."""

        indices: List[int] = []
        for v in variables:
            start_index = self.local_storage_metadata.index_from_variable[v]
            indices.extend(
                range(start_index, start_index + v.get_local_parameter_dim())
            )
        return self._compute_marginal(indices)

    def _compute_marginal(self, indices: Sequence[int]) -> onp.ndarray:
        """Compute marginal covariance using a set of indices.

        Extracts a square matrix, where the source row and column indices are specified
        by `indices`."""

        dim = len(indices)
        marginal_covariance = onp.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                marginal_covariance[i, j] = self[indices[i], indices[j]]
        return marginal_covariance

    def __getitem__(self, indices: Tuple[int, int]) -> float:
        """Get a single value in our sparse Covariance matrix."""

        # Enforce symmetry
        if indices[0] > indices[1]:
            indices = indices[::-1]
        row, col = indices

        # Use cached value if available
        if indices in self._value_cache:
            return self._value_cache[indices]

        # Compute covariance value from square-root information matrix
        value: float

        if row == col:
            value = self.L_diag_inv[col] * (
                self.L_diag_inv[col] - self._sum_over_col(col, col)
            )
        else:
            value = -self.L_diag_inv[row] * self._sum_over_col(row, col)

        self._value_cache[indices] = value
        return value

    def _sum_over_col(self, col: int, col_l: int) -> float:
        column: scipy.sparse.csc_matrix = self.L.getcol(col)

        total: float = 0.0
        row: int
        val: float
        for row, val in zip(column.indices, column.data):
            if col != row:  # Skip diagonal
                total += val * self[row, col_l]

        return total
