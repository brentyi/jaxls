"""Simple pose graph example with two pose variables and three costs:

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

# Costs take two main arguments:
# - A callable with signature `(jaxls.VarValues, *Args) -> jax.Array`.
# - A tuple of arguments: the type should be `tuple[*Args]`.
#
# All arguments should be PyTree structures. Variable types within the PyTree
# will be automatically detected.
costs = [
    # Prior cost for pose 0.
    jaxls.Cost(
        lambda vals, var, init: (vals[var] @ init.inverse()).log(),
        (vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
    ),
    # Prior cost for pose 1.
    jaxls.Cost(
        lambda vals, var, init: (vals[var] @ init.inverse()).log(),
        (vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
    ),
    # "Between" cost.
    jaxls.Cost(
        lambda vals, delta, var0, var1: (
            (vals[var0].inverse() @ vals[var1]) @ delta.inverse()
        ).log(),
        (jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0), vars[0], vars[1]),
    ),
]

# Create and analyze the least squares problem.
#
# This goes through costs, and preprocesses them to enable vectorization of
# computations. If we have 1000 prior costs, we will internally stack all of
# the associated values and batch computations.
problem = jaxls.LeastSquaresProblem(costs, vars).analyze()

# Solve the optimization problem.
solution = problem.solve()
print("All solutions", solution)
print("Pose 0", solution[vars[0]])
print("Pose 1", solution[vars[1]])
