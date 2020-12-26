import dataclasses
from typing import TYPE_CHECKING, Any, Hashable, Tuple, Type

import jax
from jax import numpy as jnp

from . import utils

if TYPE_CHECKING:
    from ._factors import FactorBase

PyTree = Any
VariableValue = PyTree
LocalVariableValue = PyTree

ScaleTril = jnp.ndarray
ScaleTrilInv = jnp.ndarray


@jax.partial(utils.register_dataclass_pytree, static_fields=("shape",))
@dataclasses.dataclass(frozen=True)
class SparseMatrix:
    """Sparse matrix in COO form."""

    values: jnp.ndarray
    """Non-zero matrix values. Shape should be `(*, N)`."""
    coords: jnp.ndarray
    """Row, value positions of non-zero entries. Shape should be `(*, N, 2)`."""
    shape: Tuple[int, int]
    """Shape of matrix."""

    def __post_init__(self):
        assert self.coords.shape == self.values.shape + (2,)
        assert len(self.shape) == 2

    def __matmul__(self, other: jnp.ndarray):
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


@dataclasses.dataclass(frozen=True)
class GroupKey:
    """Key for grouping factors that can be computed in parallel."""

    factor_type: Type["FactorBase"]
    secondary_key: Hashable


__all__ = [
    "PyTree",
    "VariableValue",
    "LocalVariableValue",
    "ScaleTril",
    "ScaleTrilInv",
    "SparseMatrix",
    "GroupKey",
]
