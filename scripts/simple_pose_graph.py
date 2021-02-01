import time

import jaxlie
from jax import numpy as jnp

import jaxfg

variables = {
    "pose1": jaxfg.geometry.SE2Variable(),
}

graph = jaxfg.core.PreparedFactorGraph.from_factors(
    [
        jaxfg.geometry.PriorFactor.make(
            variable=variables["pose1"],
            mu=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
            scale_tril_inv=jnp.eye(3),
        ),
        jaxfg.geometry.PriorFactor.make(
            variable=variables["pose1"],
            mu=jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0),
            scale_tril_inv=jnp.eye(3),
        ),
        # jaxfg.BetweenFactor.make(
        #     variable_T_world_a=variables["pose1"],
        #     variable_T_world_b=variables["pose2"],
        #     T_a_b=jaxlie.SE2.from_xy_theta(1., 0., 0.).xy_unit_complex,
        #     scale_tril_inv=jnp.eye(3),
        # ),
    ]
)
initial_assignments = jaxfg.core.VariableAssignments.create_default(variables.values())
print(initial_assignments)

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
