from typing import Mapping

from jax import numpy as jnp

from ._variables import RealVectorVariable, VariableBase

ScaleTril = jnp.ndarray
ScaleTrilInv = jnp.ndarray

VariableAssignments = Mapping[VariableBase, jnp.ndarray]
RealVectorVariableAssignments = Mapping[RealVectorVariable, jnp.ndarray]

__all__ = [
    "ScaleTril",
    "ScaleTrilInv",
    "VariableAssignments",
    "RealVectorVariableAssignments",
]
