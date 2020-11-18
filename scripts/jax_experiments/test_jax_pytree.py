import time

import jax
import jax.numpy as jnp

A = {
    "a": [jnp.eye(4), jnp.eye(4)],
    "b": [jnp.eye(4), jnp.eye(2)],
}
x = {
    "a": jnp.ones(4),
    "b": jnp.ones(2),
}

print(jax.tree_multimap(lambda a, x: a @ x, A, x))
