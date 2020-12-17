import dataclasses
from typing import TYPE_CHECKING, Hashable, Type

from jax import numpy as jnp

if TYPE_CHECKING:
    from ._factors import FactorBase

ScaleTril = jnp.ndarray
ScaleTrilInv = jnp.ndarray


@dataclasses.dataclass(frozen=True)
class GroupKey:
    factor_type: Type["FactorBase"]
    secondary_key: Hashable


__all__ = [
    "ScaleTril",
    "ScaleTrilInv",
]
