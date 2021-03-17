from typing import Any, Union

import numpy as onp
from jax import numpy as jnp

Array = Union[jnp.ndarray, onp.ndarray]

ScaleTril = Array
ScaleTrilInv = Array

PyTree = Any
VariableValue = PyTree
LocalVariableValue = PyTree
