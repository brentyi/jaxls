import jax
import jax.experimental.sparse
import jax_dataclasses as jdc
from jax import numpy as jnp


@jdc.pytree_dataclass
class SparseCsrCoordinates:
    indices: jax.Array
    """Column indices of non-zero entries. Shape should be `(*, N)`."""
    indptr: jax.Array
    """Index of start to each row. Shape should be `(*, num_rows)`."""
    shape: jdc.Static[tuple[int, int]]
    """Shape of matrix."""


@jdc.pytree_dataclass
class SparseCsrMatrix:
    """Data structure for sparse CSR matrices."""

    values: jax.Array
    """Non-zero matrix values. Shape should be `(*, N)`."""
    coords: SparseCsrCoordinates
    """Indices describing non-zero entries."""


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
    """Indices describing non-zero entries."""

    def as_jax_bcoo(self) -> jax.experimental.sparse.BCOO:
        return jax.experimental.sparse.BCOO(
            args=(
                self.values,
                jnp.stack([self.coords.rows, self.coords.cols], axis=-1),
            ),
            shape=self.coords.shape,
            indices_sorted=True,
            unique_indices=True,
        )
