#!/usr/bin/env python3
"""Test augmented Lagrangian constraints implementation."""

import jax
import jax.numpy as jnp
import jaxls

# Simple test: minimize ||x - target||^2 subject to x >= lower_bound
def test_simple_constraint():
    print("Testing simple inequality constraint...")

    # Create a simple scalar variable
    x_var = jaxls.Var(id=0, default_value=jnp.array([0.0]))

    # Cost: try to get close to -5.0
    @jaxls.Cost.create_factory
    def distance_cost(vals: jaxls.VarValues, var: jaxls.Var) -> jax.Array:
        return vals[var] - jnp.array([-5.0])

    # Constraint: x >= -2.0 (i.e., -x + 2.0 <= 0)
    def lower_bound_constraint(vals: jaxls.VarValues, var: jaxls.Var) -> jax.Array:
        return -vals[var] + jnp.array([-2.0])  # x >= -2.0 => -x + (-2) <= 0

    constraint = jaxls.Constraint(
        compute_constraint=lower_bound_constraint,
        args=(x_var,),
        constraint_type="inequality",
        name="lower_bound",
    )

    # Create and solve problem
    problem = jaxls.LeastSquaresProblem(
        costs=[distance_cost(x_var)],
        variables=[x_var],
    )

    result = problem.solve_with_constraints(
        constraints=[constraint],
        augmented_lagrangian=jaxls.AugmentedLagrangianConfig(
            penalty_initial=1.0,
            penalty_factor=10.0,
            outer_iterations=5,
            constraint_tolerance=1e-4,
        ),
        verbose=True,
    )

    x_final = result[x_var].squeeze()
    print(f"Result: x = {x_final:.4f}")
    print(f"Constraint satisfied: x >= -2.0? {x_final >= -2.0}")
    print(f"Expected: x ≈ -2.0 (clamped to constraint)")

    # Verify constraint is satisfied
    assert x_final >= -2.0 - 1e-3, f"Constraint violated: {x_final} < -2.0"
    assert abs(x_final - (-2.0)) < 0.1, f"Solution not near constraint: {x_final}"

    print("✓ Test passed!\n")


def test_equality_constraint():
    print("Testing equality constraint...")

    # Create two scalar variables
    x_var = jaxls.Var(id=0, default_value=jnp.array([1.0]))
    y_var = jaxls.Var(id=1, default_value=jnp.array([1.0]))

    # Cost: minimize (x-3)^2 + (y-4)^2
    @jaxls.Cost.create_factory
    def distance_cost(vals: jaxls.VarValues, x: jaxls.Var, y: jaxls.Var) -> jax.Array:
        return jnp.concatenate([
            vals[x] - jnp.array([3.0]),
            vals[y] - jnp.array([4.0])
        ])

    # Constraint: x + y = 5
    def sum_constraint(vals: jaxls.VarValues, x: jaxls.Var, y: jaxls.Var) -> jax.Array:
        return vals[x] + vals[y] - jnp.array([5.0])

    constraint = jaxls.Constraint(
        compute_constraint=sum_constraint,
        args=(x_var, y_var),
        constraint_type="equality",
        name="sum_eq_5",
    )

    problem = jaxls.LeastSquaresProblem(
        costs=[distance_cost(x_var, y_var)],
        variables=[x_var, y_var],
    )

    result = problem.solve_with_constraints(
        constraints=[constraint],
        augmented_lagrangian=jaxls.AugmentedLagrangianConfig(
            penalty_initial=10.0,
            penalty_factor=5.0,
            outer_iterations=10,
            constraint_tolerance=1e-5,
        ),
        verbose=True,
    )

    x_final = result[x_var].squeeze()
    y_final = result[y_var].squeeze()

    print(f"Result: x = {x_final:.4f}, y = {y_final:.4f}")
    print(f"Constraint: x + y = {x_final + y_final:.4f} (should be 5.0)")
    print(f"Expected: x ≈ 2.5, y ≈ 2.5 (by symmetry)")

    # Verify constraint is satisfied
    assert abs(x_final + y_final - 5.0) < 1e-3, f"Constraint violated: {x_final} + {y_final} != 5.0"

    print("✓ Test passed!\n")


if __name__ == "__main__":
    test_simple_constraint()
    test_equality_constraint()
    print("All tests passed! ✓")
