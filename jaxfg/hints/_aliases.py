from typing import Any, Union

import numpy as onp
from jax import numpy as jnp

Array = Union[jnp.ndarray, onp.ndarray]
Scalar = Union[Array, float]

PyTree = Any

VariableValue = PyTree
LocalVariableValue = Array
