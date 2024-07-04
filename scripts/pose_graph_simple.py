"""Simple pose graph example with two pose variables and three factors:

    ┌────────┐             ┌────────┐
    │ Pose 0 ├───Between───┤ Pose 1 │
    └───┬────┘             └────┬───┘
        │                       │
        │                       │
      Prior                   Prior

"""

from typing import List

import jaxfg2
import jaxlie
from jax import numpy as jnp

# Create variables: each variable object represents something that we want to solve for.
#
# The variable objects themselves don't hold any values, but can be used as a key for
# accessing values from a VariableAssignments object. (see below)
pose_variables = (jaxfg2.SE2Var(-10), jaxfg2.SE2Var(0))

# Create factors: each defines a conditional probability distribution over some
# variables.


def prior_two(vals, var, orig, var1, orig1):
    return jnp.concatenate(
        [(vals[var] @ orig.inverse()).log(), (vals[var1] @ orig1.inverse()).log()]
    )


def prior_one(vals, var, orig):
    return (vals[var] @ orig.inverse()).log()


# def between(vals, var0, var1, delta):
#     return ((vals[var0] @ delta).inverse() @ vals[var1]).log()


factors = [
    jaxfg2.Factor.make(
        # lambda vals, var0, var, init: (
        #     vals[var] @ init
        # ).log(),
        prior_two,
        (
            pose_variables[0],
            jaxlie.SE2.from_translation(jnp.array([100.0, 10.0])),
            pose_variables[1],
            jaxlie.SE2.from_translation(jnp.array([200.0, 20.0])),
        ),
    ),
    jaxfg2.Factor.make(
        # lambda vals, var0, var, init: (
        #     vals[var] @ init
        # ).log(),
        prior_two,
        (
            pose_variables[0],
            jaxlie.SE2.from_translation(jnp.array([100.0, 10.0])),
            pose_variables[1],
            jaxlie.SE2.from_translation(jnp.array([200.0, 20.0])),
        ),
    ),
    jaxfg2.Factor.make(
        # lambda vals, var0, var, init: (
        #     vals[var] @ init
        # ).log(),
        prior_one,
        (
            pose_variables[0],
            jaxlie.SE2.from_translation(jnp.array([100.0, 10.0])),
        ),
    ),
    jaxfg2.Factor.make(
        # lambda vals, var0, var, init: (
        #     vals[var] @ init
        # ).log(),
        prior_one,
        (
            pose_variables[1],
            jaxlie.SE2.from_translation(jnp.array([200.0, 20.0])),
        ),
    ),
    # jaxfg2.Factor.make(
    #     # lambda vals, var: (
    #     #     vals[var] @ jaxlie.SE2.from_translation(jnp.array([200.0, 20.0]))
    #     # ).log(),
    #     # lambda *args: prior(*args),
    #     prior,
    #     (pose_variables[1], jaxlie.SE2.from_translation(jnp.array([200.0, 20.0]))),
    # ),
    # jaxfg2.Factor.make(
    #     # lambda vals, var: (
    #     #     vals[var] @ jaxlie.SE2.from_translation(jnp.array([200.0, 20.0]))
    #     # ).log(),
    #     # lambda *args: prior(*args),
    #     between,
    #     (
    #         pose_variables[0],
    #         pose_variables[1],
    #         jaxlie.SE2.from_translation(jnp.array([50.0, 10.0])),
    #     ),
    # ),
]

# Create our "stacked" factor graph. (this is the only kind of factor graph)
#
# This goes through factors, and preprocesses them to enable vectorization of
# computations. If we have 1000 PriorFactor objects, we stack all of the associated
# values and perform a batched operation that computes all 1000 residuals.
graph = jaxfg2.StackedFactorGraph.make(factors, vars=pose_variables)


# Create an assignments object, which you can think of as a (variable => value) mapping.
# These initial values will be used by our nonlinear optimizer.
#
# We just use each variables' default values here -- SE(2) identity -- but for bigger
# problems bad initializations => no convergence when we run our nonlinear optimizer.
initial_assignments = jaxfg2.VarValues.from_defaults(pose_variables)

solver = jaxfg2.GaussNewtonSolver()

output = solver.solve(graph, initial_assignments)
print(output)
