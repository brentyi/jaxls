import time

import jax
import jax.numpy as jnp
import jaxfg
from jax import numpy as jnp


def func(x):
    return jnp.sin(x + 10.0)


N = 100
iters = 100
jacfwd = jax.jacfwd(func)
jacfwd_vmap = jax.vmap(jax.jacfwd(func))

with jaxfg.utils.stopwatch("vmap"):
    for i in range(iters):
        jacobian = jax.jit(jacfwd_vmap)(jnp.zeros((N, 5)))
        assert jacobian.shape == (N, 5, 5)

with jaxfg.utils.stopwatch("vmap no jit"):
    for i in range(iters):
        jacobian = jacfwd_vmap(jnp.zeros((N, 5)))
        assert jacobian.shape == (N, 5, 5)

with jaxfg.utils.stopwatch("loop no vec"):
    for i in range(iters):
        jacobian = jnp.array(list(jax.jit(jacfwd)(vec) for vec in jnp.zeros((N, 5))))
        assert jacobian.shape == (N, 5, 5)

with jaxfg.utils.stopwatch("loop no vec no jit"):
    for i in range(iters):
        jacobian = jnp.array(list(jacfwd(vec) for vec in jnp.zeros((N, 5))))
        assert jacobian.shape == (N, 5, 5)

# How can we generalize this to factors?
#   Make FactorBase.compute_error functional
#       This would be nice:
#           > FactorBase.compute_error(*variable values, **kwargs)?
#       ...but vmap doesn't support keyword arguments.
#   I think we'll have to go with:
#   vmap
