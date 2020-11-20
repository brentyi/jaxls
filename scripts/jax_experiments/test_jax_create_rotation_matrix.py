import time

import jax
import jax.numpy as jnp
import jaxfg
import numpy as onp

iters = 100

for jit_label, jit in {
    "jit build: ": lambda x: jax.jit(x, static_argnums=(0,)),
    "jit: ": jax.jit,
    "": lambda x: x,
}.items():
    with jaxfg.utils.stopwatch(f"{jit_label}jnp.array multi-trig"):

        @jit
        def create_R(theta):
            R = jnp.array(
                [[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]]
            )
            return R

        for i in range(iters):
            create_R(i)

    with jaxfg.utils.stopwatch(f"{jit_label}jnp.array re-use cos and sin"):

        @jit
        def create_R(theta):
            cos = jnp.cos(theta)
            sin = jnp.sin(theta)
            R = jnp.array([[cos, sin], [-sin, cos]])
            return R

        for i in range(iters):
            create_R(i)

    with jaxfg.utils.stopwatch(f"{jit_label}expm"):

        @jit
        def create_R(theta):
            so2 = jnp.array([
                [0, theta],
                [-theta, 0],
            ])
            R = jax.scipy.linalg.expm(so2)
            return R

        for i in range(iters):
            create_R(float(i))

    with jaxfg.utils.stopwatch(f"{jit_label}zeros and fill"):

        @jit
        def create_R(theta):
            cos = jnp.cos(theta)
            sin = jnp.sin(theta)
            R = jnp.zeros((2, 2))
            R.at[0, 0].set(cos)
            R.at[0, 1].set(sin)
            R.at[1, 1].set(cos)
            R.at[1, 0].set(-sin)
            return R

        for i in range(iters):
            create_R(i)
