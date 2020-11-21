import time

# import numpy as jnp
import jax
import jax.numpy as jnp


def A(x):
    print(x)
    return x


b = {
    "a": jnp.array([1.0, 2.0, 3.0]),
    "b": jnp.array([1.0, 2.0, 3.0]),
}

print(jax.scipy.sparse.linalg.cg(A, b))
