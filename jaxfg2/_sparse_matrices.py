import jax
import jax_dataclasses as jdc
import scipy.sparse
from jax import numpy as jnp


@jdc.pytree_dataclass
class SparseCsrCoordinates:
    row_starts: jax.Array
    """Index into `cols` for the start of each row."""
    cols: jax.Array
    """Column indices of non-zero entries. Shape should be `(*, N)`."""


@jdc.pytree_dataclass
class SparseCsrMatrix:
    """Sparse matrix in COO form."""

    values: jax.Array
    """Non-zero matrix values. Shape should be `(*, N)`."""
    coords: SparseCsrCoordinates
    """Row and column indices of non-zero entries. Shapes should be `(*, N)`."""
    shape: jdc.Static[tuple[int, int]]
    """Shape of matrix."""


@jdc.pytree_dataclass
class SparseCooCoordinates:
    rows: jax.Array
    """Row indices of non-zero entries. Shape should be `(*, N)`."""
    cols: jax.Array
    """Column indices of non-zero entries. Shape should be `(*, N)`."""


@jdc.pytree_dataclass
class SparseCooMatrix:
    """Sparse matrix in COO form."""

    values: jax.Array
    """Non-zero matrix values. Shape should be `(*, N)`."""
    coords: SparseCooCoordinates
    """Row and column indices of non-zero entries. Shapes should be `(*, N)`."""
    shape: jdc.Static[tuple[int, int]]
    """Shape of matrix."""

    def __matmul__(self, other: jax.Array):
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

    @property
    def T(self):
        """Return transpose of our sparse matrix."""
        h, w = self.shape
        return SparseCooMatrix(
            values=self.values,
            coords=SparseCooCoordinates(
                rows=self.coords.cols,
                cols=self.coords.rows,
            ),
            shape=(w, h),
        )

    def as_scipy_coo_matrix(self) -> scipy.sparse.coo_matrix:
        """Convert to a sparse scipy matrix."""
        assert len(self.values.shape) == 1
        return scipy.sparse.coo_matrix(
            (self.values, (self.coords.rows, self.coords.cols)), shape=self.shape
        )
