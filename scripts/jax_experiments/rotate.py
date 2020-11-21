import time

import jax
import jax.numpy as jnp
import jaxfg
import numpy as onp

iters = 500

for jit_label, jit in {
    "jit build: ": lambda x: jax.jit(x, static_argnums=(0,)),
    "jit: ": jax.jit,
    "": lambda x: x,
}.items():
    with jaxfg.utils.stopwatch(f"{jit_label}matrix"):

        @jit
        def rotate(theta):
            cos = jnp.cos(theta)
            sin = jnp.sin(theta)
            R = jnp.array(
                [[cos, -sin], [sin, cos]]
            )
            p = jnp.ones(2)
            return R @ R

        for i in range(iters):
            rotate(i)

    with jaxfg.utils.stopwatch(f"{jit_label}decomposed"):

        @jit
        def rotate(theta):
            cos = jnp.cos(theta)
            sin = jnp.sin(theta)
            p = jnp.ones(2)
            return jnp.array([cos * p[0] - sin * p[1], sin * p[0] + cos * p[1]])

        for i in range(iters):
            rotate(i)
