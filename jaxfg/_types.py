import dataclasses
from typing import TYPE_CHECKING, Any, Dict, Hashable, Type, Union

import numpy as onp
from jax import numpy as jnp

if TYPE_CHECKING:
    from ._factors import FactorBase
    from ._variables import VariableBase

ScaleTril = Any  # jnp.ndarray
ScaleTrilInv = Any  # jnp.ndarray


@dataclasses.dataclass(frozen=True)
class GroupKey:
    factor_type: Type["FactorBase"]
    secondary_key: Hashable


VariableAssignments = Dict["VariableBase", Union[onp.ndarray, jnp.ndarray]]

__all__ = [
    "ScaleTril",
    "ScaleTrilInv",
    "VariableAssignments",
]
