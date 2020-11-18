from typing import Any, Dict

from jax import numpy as jnp

from ._variables import RealVectorVariable, VariableBase

ScaleTril = Any  # jnp.ndarray
ScaleTrilInv = Any  # jnp.ndarray

VariableAssignments = Dict[VariableBase, jnp.ndarray]
RealVectorVariableAssignments = Dict[RealVectorVariable, jnp.ndarray]

__all__ = [
    "ScaleTril",
    "ScaleTrilInv",
    "VariableAssignments",
    "RealVectorVariableAssignments",
]
