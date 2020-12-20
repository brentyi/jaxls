import dataclasses
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

f = jaxfg.PriorFactor.make(
    variable=variables["pose1"],
    mu=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
    scale_tril_inv=jnp.eye(3),
)


@jax.jit
def do_something_with_factor(factor: jaxfg.FactorBase) -> jaxfg.FactorBase:
    return factor


do_something_with_factor(f)
jax.vmap(do_something_with_factor)(jaxlie.SO2.identity())
# do_something_with_factor(f)
# do_something_with_factor([f])
# do_something_with_factor(
#     jaxfg.BetweenFactor.make(
#         before=variables["pose1"],
#         after=variables["pose1"],
#         delta=jaxlie.SE2.identity(),
#         scale_tril_inv=jnp.eye(3),
#     )
# )
#
#
initial_assignments = jaxfg.VariableAssignments.create_default(variables.values())

graph = jaxfg.PreparedFactorGraph.from_factors([f])
jaxfg.PriorFactor.compute_error(
    f, jaxlie.SE2(xy_unit_complex=jnp.array([1.0, 1.0, 0.0, 1.0]))
)
# jax.vmap(jaxfg.PriorFactor.compute_error)(
#     f,
#     jaxlie.SE2(
#         xy_unit_complex=jnp.array(
#             [
#                 [1.0, 1.0, 0.0, 1.0],
#                 [1.0, 1.0, 0.0, 1.0],
#             ]
#         )
#     )
# )
# graph.compute_error_vector(initial_assignments)

# .solve(initial_assignments)
