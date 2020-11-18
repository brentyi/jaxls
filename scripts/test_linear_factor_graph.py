import time

import jax
import jaxfg
from jax import numpy as jnp

position = jaxfg.RealVectorVariable(parameter_dim=2)
position2 = jaxfg.RealVectorVariable(parameter_dim=2)


graph = jaxfg.FactorGraph().with_factors(
    jaxfg.LinearFactor(
        {position: lambda x: jnp.identity(2) @ x},
        b=jnp.array([2.0, 3.0]),
    ),
    jaxfg.LinearFactor(
        {position2: lambda x: jnp.identity(2) @ x},
        b=jnp.array([8.0, 3.0]),
    ),
)


# graph = graph.with_factors(
#     jaxfg.LinearFactor(
#         {position: lambda x: jnp.identity(2) @ x},
#         b=jnp.array([8.0, 7.0]),
#     ),
# )
# print(jit_solve(graph, (position,)))  # , assignments={position: jnp.zeros(2)}))
# graph = graph.with_factors(
#     jaxfg.LinearFactor(
#         {position: lambda x: jnp.identity(2) @ x},
#         b=jnp.array([800.0, 7.0]),
#     ),
# )
# start_time = time.time()
# print(jit_solve(graph, (position,)))  # , assignments={position: jnp.zeros(2)}))
# print(time.time() - start_time)

start_time = time.time()
print(graph.solve())  # , assignments={position: jnp.zeros(2)}))
print(time.time() - start_time)

start_time = time.time()
print(graph.solve())  # , assignments={position: jnp.zeros(2)}))
print(time.time() - start_time)
