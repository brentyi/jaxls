import jax
import jax_dataclasses as jdc
from jax import numpy as jnp


@jdc.pytree_dataclass
class SparseCsrCoordinates:
    indices: jax.Array
    """Column indices of non-zero entries. Shape should be `(*, N)`."""
    indptr: jax.Array
    """Index of start to each row. Shape should be `(*, num_rows)`."""
    shape: jdc.Static[tuple[int, int]]


@jdc.pytree_dataclass
class SparseCsrMatrix:
    """Data structure for sparse CSR matrices."""

    values: jax.Array
    """Non-zero matrix values. Shape should be `(*, N)`."""
    coords: SparseCsrCoordinates


@jdc.pytree_dataclass
class SparseCooCoordinates:
    rows: jax.Array
    """Row indices of non-zero entries. Shape should be `(*, N)`."""
    cols: jax.Array
    """Column indices of non-zero entries. Shape should be `(*, N)`."""
    shape: jdc.Static[tuple[int, int]]
    """Shape of matrix."""


@jdc.pytree_dataclass
class SparseCooMatrix:
    """Sparse matrix in COO form."""

    values: jax.Array
    """Non-zero matrix values. Shape should be `(*, N)`."""
    coords: SparseCooCoordinates
    """Row and column indices of non-zero entries. Shapes should be `(*, N)`."""

    def __matmul__(self, other: jax.Array):
        """Compute `Ax`, where `x` is a 1D vector."""
        assert other.shape == (
            self.coords.shape[1],
        ), "Inner product only supported for 1D vectors!"
        return (
            jnp.zeros(self.coords.shape[0], dtype=other.dtype)
            .at[self.coords.rows]
            .add(self.values * other[self.coords.cols])
        )

    def as_dense(self) -> jnp.ndarray:
        """Convert to a dense JAX array."""
        return (
            jnp.zeros(self.coords.shape)
            .at[self.coords.rows, self.coords.cols]
            .set(self.values)
        )

    @property
    def T(self):
        """Return transpose of our sparse matrix."""
        h, w = self.coords.shape
        return SparseCooMatrix(
            values=self.values,
            coords=SparseCooCoordinates(
                rows=self.coords.cols,
                cols=self.coords.rows,
                shape=(w, h),
            ),
        )
