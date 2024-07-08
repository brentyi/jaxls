"""Simple pose graph example with two pose variables and three factors:

┌────────┐             ┌────────┐
│ Pose 0 ├───Between───┤ Pose 1 │
└───┬────┘             └────┬───┘
    │                       │
    │                       │
  Prior                   Prior

"""


import jaxfg2
import jaxlie

# Create variables: each variable object represents something that we want to solve for.
#
# The variable objects themselves don't hold any values, but can be used as a key for
# accessing values from a VariableAssignments object. (see below)
pose_variables = (jaxfg2.SE2Var(-10), jaxfg2.SE2Var(0))

# Create factors: each defines a conditional probability distribution over some
# variables.
factors = [
    # Prior factor for pose 0.
    jaxfg2.Factor.make(
        lambda vals, var, init: (vals[var] @ init.inverse()).log(),
        (pose_variables[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
    ),
    # Prior factor for pose 1.
    jaxfg2.Factor.make(
        lambda vals, var, init: (vals[var] @ init.inverse()).log(),
        (pose_variables[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
    ),
    # "Between" factor.
    jaxfg2.Factor.make(
        lambda vals, var0, var1, delta: (
            (vals[var0].inverse() @ vals[var1]) @ delta.inverse()
        ).log(),
        (pose_variables[0], pose_variables[1], jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0)),
    ),
]

# Create our "stacked" factor graph. (this is the only kind of factor graph)
#
# This goes through factors, and preprocesses them to enable vectorization of
# computations. If we have 1000 PriorFactor objects, we stack all of the associated
# values and perform a batched operation that computes all 1000 residuals.
graph = jaxfg2.FactorGraph.make(factors, pose_variables)


# Create an assignments object, which you can think of as a (variable => value) mapping.
# These initial values will be used by our nonlinear optimizer.
#
# We just use each variables' default values here -- SE(2) identity -- but for bigger
# problems bad initializations => no convergence when we run our nonlinear optimizer.
initial_assignments = jaxfg2.VarValues.from_defaults(pose_variables)

output = graph.solve(initial_assignments)
print(output)
