import time

import jax.numpy as jnp

# import numpy as jnp
from jax import vmap

# jit = lambda x: x


def plus(x: jnp.ndarray):
    return jnp.array([x[0], x.shape[0]])


print(plus(jnp.array([1.0, 2.0, 3.0])))
print(vmap(plus)(jnp.array([[1.0, 2.0, 3.0]])))


x = jnp.arange(100)

print(jnp.split(x, [0, 10, 20]))
