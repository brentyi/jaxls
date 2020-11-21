import jax
from jax import numpy as jnp

x = jnp.arange(10) + 5
print(x[jnp.array([[0, 1, 2], [5, 6, 7]])])
