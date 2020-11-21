import time

import jax
import jax.numpy as jnp
import jaxfg
import numpy as onp
from jax import numpy as jnp

dim = 10
rows = 200


def make_vstack():
    return jnp.vstack([jnp.zeros(dim) for _ in range(rows)])


def make_array():
    return jnp.array([jnp.zeros(dim) for _ in range(rows)])


def make_array_onp():
    return jnp.array(onp.array([onp.zeros(dim) for _ in range(rows)]))


iters = 5


with jaxfg.utils.stopwatch("vstack"):
    for _ in range(iters):
        make_vstack()


with jaxfg.utils.stopwatch("vstack"):
    for _ in range(iters):
        jax.jit(make_vstack)()


with jaxfg.utils.stopwatch("array"):
    for _ in range(iters):
        make_array()


with jaxfg.utils.stopwatch("array jit"):
    for _ in range(iters):
        jax.jit(make_array)()


with jaxfg.utils.stopwatch("array onp"):
    for _ in range(iters):
        make_array_onp()


with jaxfg.utils.stopwatch("array onp jit"):
    for _ in range(iters):
        jax.jit(make_array_onp)()
