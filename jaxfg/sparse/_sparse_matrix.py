from typing import Tuple

import jax_dataclasses as jdc
import scipy
from jax import numpy as jnp

from .. import hints


@jdc.pytree_dataclass
class SparseCooCoordinates:
    rows: hints.Array
    """Row indices of non-zero entries. Shape should be `(*, N)`."""
    cols: hints.Array
    """Column indices of non-zero entries. Shape should be `(*, N)`."""

    # Shape checks break under vmap
    # def __post_init__(self):
    #     assert self.rows.shape == self.cols.shape


@jdc.pytree_dataclass
class SparseCooMatrix:
    """Sparse matrix in COO form."""

    values: hints.Array
    """Non-zero matrix values. Shape should be `(*, N)`."""
    coords: SparseCooCoordinates
    """Row and column indices of non-zero entries. Shapes should be `(*, N)`."""
    shape: jdc.Static[Tuple[int, int]]
    """Shape of matrix."""

    # Shape checks break under vmap
    # def __post_init__(self):
    #     print(self)
    #     assert self.coords.rows.shape == self.coords.cols.shape == self.values.shape
    #     assert len(self.shape) == 2

    def __matmul__(self, other: hints.Array):
        """Compute `Ax`, where `x` is a 1D vector."""
        assert other.shape == (
            self.shape[1],
        ), "Inner product only supported for 1D vectors!"
        return (
            jnp.zeros(self.shape[0], dtype=other.dtype)
            .at[self.coords.rows]
            .add(self.values * other[self.coords.cols])
        )

    def as_dense(self) -> jnp.ndarray:
        """Convert to a dense JAX array."""
        # TODO: untested
        return (
            jnp.zeros(self.shape)
            .at[self.coords.rows, self.coords.cols]
            .set(self.values)
        )

    @staticmethod
    def from_scipy_coo_matrix(matrix: scipy.sparse.coo_matrix) -> "SparseCooMatrix":
        """Build from a sparse scipy matrix."""
        return SparseCooMatrix(
            values=matrix.data,
            coords=SparseCooCoordinates(
                rows=matrix.row,
                cols=matrix.col,
            ),
            shape=matrix.shape,
        )

    def as_scipy_coo_matrix(self) -> scipy.sparse.coo_matrix:
        """Convert to a sparse scipy matrix."""
        return scipy.sparse.coo_matrix(
            (self.values, (self.coords.rows, self.coords.cols)), shape=self.shape
        )

    @property
    def T(self):
        """Return transpose of our sparse matrix."""
        return SparseCooMatrix(
            values=self.values,
            coords=SparseCooCoordinates(
                rows=self.coords.cols,
                cols=self.coords.rows,
            ),
            shape=self.shape[::-1],
        )
