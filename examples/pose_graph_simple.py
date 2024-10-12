"""Simple pose graph example with two pose variables and three factors:

┌────────┐             ┌────────┐
│ Pose 0 ├───Between───┤ Pose 1 │
└───┬────┘             └────┬───┘
    │                       │
    │                       │
  Prior                   Prior

"""

import jaxlie
import jaxls

# Create variables: each variable object represents something that we want to solve for.
#
# The variable objects themselves don't hold any values, but can be used as a key for
# accessing values from a VariableAssignments object. (see below)
vars = (jaxls.SE2Var(0), jaxls.SE2Var(1))

# Create factors: each defines a conditional probability distribution over some
# variables.
factors = [
    # Prior factor for pose 0.
    jaxls.Factor(
        lambda vals, var, init: (vals[var] @ init.inverse()).log(),
        (vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
    ),
    # Prior factor for pose 1.
    jaxls.Factor(
        lambda vals, var, init: (vals[var] @ init.inverse()).log(),
        (vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
    ),
    # "Between" factor.
    jaxls.Factor(
        lambda vals, var0, var1, delta: (
            (vals[var0].inverse() @ vals[var1]) @ delta.inverse()
        ).log(),
        (vars[0], vars[1], jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0)),
    ),
]

# Create our "stacked" factor graph. (this is the only kind of factor graph)
#
# This goes through factors, and preprocesses them to enable vectorization of
# computations. If we have 1000 PriorFactor objects, we stack all of the associated
# values and perform a batched operation that computes all 1000 residuals.
graph = jaxls.FactorGraph.make(factors, vars)

# Solve the optimization problem.
solution = graph.solve(linear_solver=jaxls.ConjugateGradientLinearSolver())
print("All solutions", solution)
print("Pose 0", solution[vars[0]])
print("Pose 1", solution[vars[1]])
