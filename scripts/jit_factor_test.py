import jax
import jaxlie
from jax import numpy as jnp

import jaxfg

variables = {
    "pose1": jaxfg.geometry.SE2Variable(),
    # "pose2": jaxfg.SE2Variable(),
}

f = jaxfg.geometry.PriorFactor.make(
    variable=variables["pose1"],
    mu=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
    scale_tril_inv=jnp.eye(3),
)


@jax.jit
def do_something_with_factor(factor: jaxfg.core.FactorBase) -> jaxfg.core.FactorBase:
    return factor


do_something_with_factor(f)
jax.vmap(do_something_with_factor)(jaxlie.SO2.identity())
# do_something_with_factor(f)
# do_something_with_factor([f])
# do_something_with_factor(
#     jaxfg.BetweenFactor.make(
#         before=variables["pose1"],
#         after=variables["pose1"],
#         between=jaxlie.SE2.identity(),
#         scale_tril_inv=jnp.eye(3),
#     )
# )
#
#
initial_assignments = jaxfg.core.VariableAssignments.create_default(variables.values())

graph = jaxfg.core.PreparedFactorGraph.from_factors([f])
jaxfg.geometry.PriorFactor.compute_residual_vector(
    f, jaxlie.SE2(xy_unit_complex=jnp.array([1.0, 1.0, 0.0, 1.0]))
)
