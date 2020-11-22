import time

import jax
import jaxfg
import numpy as onp
from jax import numpy as jnp

position = jaxfg.RealVectorVariable(parameter_dim=2)
position2 = jaxfg.RealVectorVariable(parameter_dim=2)


graph = jaxfg.LinearFactorGraph()

for _ in range(1000):
    graph = graph.with_factors(
        jaxfg.LinearFactor(
            A_from_variable={
                position: onp.identity(2),
            },
            b=onp.array([2.0, 3.0]),
        ),
        jaxfg.LinearFactor(
            A_from_variable={
                position2: onp.identity(2),
            },
            b=onp.array([2.0, 3.0]),
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

with jaxfg.utils.stopwatch("solve"):
    print(graph.solve())  # , assignments={position: jnp.zeros(2)}))

with jaxfg.utils.stopwatch("solve"):
    print(graph.solve())  # , assignments={position: jnp.zeros(2)}))
