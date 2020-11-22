import jax
from jax import numpy as jnp

print((jnp.einsum("fev,fv->fe", jnp.zeros((500, 8, 5)), jnp.zeros((500, 5)))).shape)
