"""Simple pose graph example with two pose variables and three factors:

    ┌────────┐             ┌────────┐
    │ Pose 0 ├───Between───┤ Pose 1 │
    └───┬────┘             └────┬───┘
        │                       │
        │                       │
      Prior                   Prior

"""

from typing import List

import jaxlie
from jax import numpy as jnp

import jaxfg

# Create variables: each variable object represents something that we want to solve for.
#
# The variable objects themselves don't hold any values, but can be used as a key for
# accessing values from a VariableAssignments object. (see below)
pose_variables: List[jaxfg.geometry.SE2Variable] = [
    jaxfg.geometry.SE2Variable(),
    jaxfg.geometry.SE2Variable(),
]

# Create factors: each defines a conditional probability distribution over some
# variables.
factors: List[jaxfg.core.FactorBase] = [
    jaxfg.geometry.PriorFactor.make(
        variable=pose_variables[0],
        mu=jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0),
        noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
    ),
    jaxfg.geometry.PriorFactor.make(
        variable=pose_variables[1],
        mu=jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0),
        noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
    ),
    jaxfg.geometry.BetweenFactor.make(
        variable_T_world_a=pose_variables[0],
        variable_T_world_b=pose_variables[1],
        T_a_b=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
        noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
    ),
]

# Create our "stacked" factor graph. (this is the only kind of factor graph)
#
# This goes through factors, and preprocesses them to enable vectorization of
# computations. If we have 1000 PriorFactor objects, we stack all of the associated
# values and perform a batched operation that computes all 1000 residuals.
graph = jaxfg.core.StackedFactorGraph.make(factors)


# Create an assignments object, which you can think of as a (variable => value) mapping.
# These initial values will be used by our nonlinear optimizer.
#
# We just use each variables' default values here -- SE(2) identity -- but for bigger
# problems bad initializations => no convergence when we run our nonlinear optimizer.
initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(pose_variables)

print("Initial assignments:")
print(initial_assignments)

# A more flexible API for creating initial assignments also exists. Equivalently:
#
#     initial_assignments = jaxfg.core.VariableAssignments.make_from_dict({
#         pose_variables[0]: jaxlie.SE2.identity(),
#         pose_variables[1]: jaxlie.SE2.identity(),
#     })


# Solve. Note that the first call to solve() will be much slower than subsequent calls.
with jaxfg.utils.stopwatch("First solve (slower because of JIT compilation)"):
    solution_assignments = graph.solve(initial_assignments)
    solution_assignments.storage.block_until_ready()  # type: ignore

with jaxfg.utils.stopwatch("Solve after initial compilation"):
    solution_assignments = graph.solve(initial_assignments)
    solution_assignments.storage.block_until_ready()  # type: ignore

# Note that we can also specify what nonlinear solver to use. Gauss-Newton is the
# default; Levenberg-Marquardt is also implemented:
#
#     solution_assignments = graph.solve(
#         initial_assignments,
#         solver=jaxfg.solvers.LevenbergMarquardtSolver(),
#     )
#
# Both have optional parameters for specifying convergence criteria, forcing sequences,
# etc.

# Print all solved variable values.
print("Solutions (jaxfg.core.VariableAssignments):")
print(solution_assignments)
print()

# Grab and print a single variable value at a time.
print("First pose (jaxlie.SE2 object):")
print(solution_assignments.get_value(pose_variables[0]))
print()

print("Second pose (jaxlie.SE2 object):")
print(solution_assignments.get_value(pose_variables[1]))
print()

# Sometimes it's useful to grab all variables of a single type at once:
print("All poses (still a jaxlie.SE2 object, but underlying parameters are stacked):")
print(solution_assignments.get_stacked_value(jaxfg.geometry.SE2Variable))
print()
