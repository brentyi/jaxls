import dataclasses
from typing import TYPE_CHECKING, Any, Hashable, Type

from jax import numpy as jnp

if TYPE_CHECKING:
    from ._factors import FactorBase

PyTree = Any
VariableValue = PyTree
LocalVariableValue = PyTree

ScaleTril = jnp.ndarray
ScaleTrilInv = jnp.ndarray


@dataclasses.dataclass(frozen=True)
class GroupKey:
    factor_type: Type["FactorBase"]
    secondary_key: Hashable


__all__ = [
    "PyTree",
    "VariableValue",
    "LocalVariableValue",
    "ScaleTril",
    "ScaleTrilInv",
]
