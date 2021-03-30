import dataclasses
from typing import Tuple

from jax import numpy as jnp

from .. import utils
from ._aliases import Array


@utils.register_dataclass_pytree(static_fields=("shape",))
@dataclasses.dataclass
class SparseMatrix:
    """Sparse matrix in COO form."""

    values: Array
    """Non-zero matrix values. Shape should be `(*, N)`."""
    coords: Array
    """Row, column positions of non-zero entries. Shape should be `(*, N, 2)`."""
    shape: Tuple[int, int]
    """Shape of matrix."""

    def __post_init__(self):
        assert self.coords.shape == self.values.shape + (2,)
        assert len(self.shape) == 2

    def __matmul__(self, other: Array):
        """Compute `Ax`, where `x` is a 1D vector."""
        assert other.shape == (
            self.shape[1],
        ), "Inner product only supported for 1D vectors!"
        return (
            jnp.zeros(self.shape[0], dtype=other.dtype)
            .at[self.coords[:, 0]]
            .add(self.values * other[self.coords[:, 1]])
        )

    @property
    def T(self):
        """Return transpose of our sparse matrix."""
        return SparseMatrix(
            values=self.values,
            coords=jnp.flip(self.coords, axis=-1),
            shape=self.shape[::-1],
        )
