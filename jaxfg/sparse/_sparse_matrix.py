import dataclasses
from typing import Tuple

from jax import numpy as jnp

from .. import hints, utils


@utils.register_dataclass_pytree
@dataclasses.dataclass
class SparseCooCoordinates:
    rows: hints.Array
    """Row indices of non-zero entries. Shape should be `(*, N)`."""
    cols: hints.Array
    """Column indices of non-zero entries. Shape should be `(*, N)`."""


@utils.register_dataclass_pytree(static_fields=("shape",))
@dataclasses.dataclass
class SparseCooMatrix:
    """Sparse matrix in COO form."""

    values: hints.Array
    """Non-zero matrix values. Shape should be `(*, N)`."""
    coords: SparseCooCoordinates
    """Row and column indices of non-zero entries. Shapes should be `(*, N)`."""
    shape: Tuple[int, int]
    """Shape of matrix."""

    def __post_init__(self):
        assert self.coords.rows.shape == self.coords.cols.shape == self.values.shape
        assert len(self.shape) == 2

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
