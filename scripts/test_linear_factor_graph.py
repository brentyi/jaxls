import jax
import jaxfg
from jax import numpy as jnp

graph = jaxfg.LinearFactorGraph()


position = jaxfg.RealVectorVariable(parameter_dim=2)

graph.add_factors(
    jaxfg.LinearFactor(
        {position: jnp.identity(2)},
        b=jnp.array([2.0, 3.0]),
        scale_tril_inv=jnp.identity(2),
    ),
)

jit_solve = jax.jit(graph.solve, static_argnums=(0,))

graph.add_factors(
    jaxfg.LinearFactor(
        {position: jnp.identity(2)},
        b=jnp.array([8.0, 7.0]),
        scale_tril_inv=jnp.identity(2),
    ),
)
print(jit_solve((position,)))  # , assignments={position: jnp.zeros(2)}))
graph.add_factors(
    jaxfg.LinearFactor(
        {position: jnp.identity(2)},
        b=jnp.array([8.0, 7.0]),
        scale_tril_inv=jnp.identity(2),
    ),
)
print(jit_solve((position,)))  # , assignments={position: jnp.zeros(2)}))
print(graph.solve((position,)))  # , assignments={position: jnp.zeros(2)}))
