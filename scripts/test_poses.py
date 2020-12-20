import time

import jax
import jaxlie
import numpy as onp
from jax import numpy as jnp

import jaxfg

variables = {
    "pose1": jaxfg.SE2Variable(),
    # "pose2": jaxfg.SE2Variable(),
}

graph = jaxfg.PreparedFactorGraph.from_factors(
    [
        jaxfg.PriorFactor.make(
            variable=variables["pose1"],
            mu=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0).xy_unit_complex,
            scale_tril_inv=jnp.eye(3),
        ),
        jaxfg.PriorFactor.make(
            variable=variables["pose1"],
            mu=jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0).xy_unit_complex,
            scale_tril_inv=jnp.eye(3),
        ),
        # jaxfg.BetweenFactor.make(
        #     before=variables["pose1"],
        #     after=variables["pose2"],
        #     delta=jaxlie.SE2.from_xy_theta(1., 0., 0.).xy_unit_complex,
        #     scale_tril_inv=jnp.eye(3),
        # ),
    ]
)
initial_assignments = jaxfg.VariableAssignments.create_default(variables.values())

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
