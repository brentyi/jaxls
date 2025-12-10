"""Simple pose graph example with two pose variables and three costs:

┌────────┐             ┌────────┐
│ Pose 0 ├───Between───┤ Pose 1 │
└───┬────┘             └────┬───┘
    │                       │
    │                       │
  Prior                   Prior

"""

import jax
import jaxlie
import jaxls

# Create variables: each variable object represents something that we want to solve for.
#
# The variable objects themselves don't hold any values, but can be used as a key for
# accessing values from a VariableAssignments object. (see below)
vars = (jaxls.SE2Var(0), jaxls.SE2Var(1))


# Create costs. In this example, we use a decorator-based syntax.
@jaxls.Cost.factory
def prior_cost(
    vals: jaxls.VarValues, var: jaxls.SE2Var, target: jaxlie.SE2
) -> jax.Array:
    """Prior cost for a pose variable. Penalizes deviations from the target"""
    return (vals[var] @ target.inverse()).log()


@jaxls.Cost.factory
def between_cost(
    vals: jaxls.VarValues, delta: jaxlie.SE2, var0: jaxls.SE2Var, var1: jaxls.SE2Var
) -> jax.Array:
    """'Between' cost for two pose variables. Penalizes deviations from the delta."""
    return ((vals[var0].inverse() @ vals[var1]) @ delta.inverse()).log()


costs = [
    prior_cost(vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
    prior_cost(vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
    between_cost(jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0), vars[0], vars[1]),
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
