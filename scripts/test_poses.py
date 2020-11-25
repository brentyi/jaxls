import time

import jax
import jaxfg
import numpy as onp
from jax import numpy as jnp

variables = {
    "pose1": jaxfg.SE2Variable(),
    "pose2": jaxfg.SE2Variable(),
}

graph = jaxfg.FactorGraph().with_factors(
    jaxfg.PriorFactor.make(
        variable=variables["pose1"],
        mu=jnp.array([1.0, 0.0, 1.0, 0.0]),
        scale_tril_inv=jnp.eye(3),
    ),
    jaxfg.BetweenFactor.make(
        before=variables["pose1"],
        after=variables["pose2"],
        delta=jnp.array([0.0, 2.0, 0.000]),
        scale_tril_inv=jnp.eye(3),
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

initial_assignments = jaxfg.types.VariableAssignments.from_dict(
    {
        variables["pose1"]: onp.array([1.0, 0.0, 1.0, 0.0]),
        variables["pose2"]: onp.array([1.0, 2.0, 1.0, 0.0]),
    }
)

start_time = time.time()
solutions = graph.solve(initial_assignments)
print("\nFirst solve runtime: ", time.time() - start_time)


print(solutions)

start_time = time.time()
graph.solve(initial_assignments)  # , assignments={position: jnp.zeros(2)}))
print("Second solve runtime: ", time.time() - start_time)

start_time = time.time()
graph.solve(solutions)  # , assignments={position: jnp.zeros(2)}))
print("Third solve runtime: ", time.time() - start_time)
