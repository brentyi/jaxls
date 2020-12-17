import time
import jaxlie

import jax
import numpy as onp
from jax import numpy as jnp

import jaxfg

variables = {
    "pose1": jaxfg.SE2Variable(),
    # "pose2": jaxfg.SE2Variable(),
}

f = jaxfg.PriorFactor.make(
    variable=variables["pose1"],
    mu=jaxlie.SE2.from_xy_theta(1., 0., 0.).xy_unit_complex,
    scale_tril_inv=jnp.eye(3),
)

@jax.jit
def do_something_with_factor(factor: jaxfg.FactorBase) -> jaxfg.FactorBase:
    return factor

initial_assignments = jaxfg.types.VariableAssignments.create_default(variables.values())
jaxfg.FactorGraph().with_factors(f).prepare().solve(initial_assignments)
