import dataclasses
from typing import TYPE_CHECKING, Any, Hashable, Tuple, Type

import jax
from jax import numpy as jnp

from . import _utils

if TYPE_CHECKING:
    from ._factors import FactorBase

PyTree = Any
VariableValue = PyTree
LocalVariableValue = PyTree

ScaleTril = jnp.ndarray
ScaleTrilInv = jnp.ndarray


@jax.partial(_utils.register_dataclass_pytree, static_fields=("shape",))
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

    def inner(self, x: jnp.ndarray):
        """Compute `Ax`, where `x` is a 1D vector."""
        assert x.shape == (self.shape[1],)
        return (
            jnp.zeros(self.shape[0], dtype=x.dtype)
            .at[self.coords[:, 0]]
            .add(self.values * x[self.coords[:, 1]])
        )

    def transpose_inner(self, x: jnp.ndarray):
        """Compute `A^Tx`, where `x` is a 1D vector."""
        assert x.shape == (self.shape[0],)
        return (
            jnp.zeros(self.shape[1], dtype=x.dtype)
            .at[self.coords[:, 1]]
            .add(self.values * x[self.coords[:, 0]])
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
