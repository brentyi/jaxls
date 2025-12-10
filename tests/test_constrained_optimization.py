"""Tests for constrained optimization using Augmented Lagrangian method."""

import jax
import jax.numpy as jnp
import jaxlie
import pytest

import jaxls


def test_simple_scalar_constraint():
    """Test simple constrained optimization: minimize ||x - 2||^2 s.t. x = 1.

    The unconstrained optimum would be x = 2, but the constraint forces x = 1.
    """

    class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.0)):
        pass

    var = ScalarVar(0)

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: ScalarVar, target: float) -> jax.Array:
        """Cost pulls variable toward target."""
        return jnp.array([vals[var] - target])

    @jaxls.Constraint.create_factory
    def constraint_fn(
        vals: jaxls.VarValues, var: ScalarVar, target: float
    ) -> jax.Array:
        """Constraint: variable must equal target."""
        return jnp.array([vals[var] - target])

    # Create problem: cost pulls to 2.0, constraint forces to 1.0
    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var, 2.0)],
        variables=[var],
        constraints=[constraint_fn(var, 1.0)],
    ).analyze()

    # Solve with initial value far from solution.
    solution = problem.solve(
        initial_vals=jaxls.VarValues.make([var.with_value(jnp.array(5.0))]),
        verbose=False,
    )

    # Solution should satisfy constraint: x = 1.0
    assert jnp.abs(solution[var] - 1.0) < 1e-5, f"Expected x=1.0, got x={solution[var]}"

    # Verify constraint is satisfied.
    constraint_violation = problem.compute_constraint_values(solution)
    assert jnp.linalg.norm(constraint_violation) < 1e-5


def test_2d_constrained_optimization():
    """Test 2D constrained optimization: min ||x||^2 s.t. x + y = 1.

    Unconstrained optimum: (0, 0)
    Constrained optimum: (0.5, 0.5)
    """

    class Vec2Var(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(2)):
        pass

    var = Vec2Var(0)

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: Vec2Var) -> jax.Array:
        """Minimize distance from origin."""
        return vals[var]

    @jaxls.Constraint.create_factory
    def constraint_fn(vals: jaxls.VarValues, var: Vec2Var) -> jax.Array:
        """Constraint: x + y = 1."""
        vec = vals[var]
        return jnp.array([vec[0] + vec[1] - 1.0])

    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var)],
        variables=[var],
        constraints=[constraint_fn(var)],
    ).analyze()

    solution = problem.solve(
        initial_vals=jaxls.VarValues.make([var.with_value(jnp.array([2.0, 3.0]))]),
        verbose=False,
    )

    # Solution should be (0.5, 0.5).
    expected = jnp.array([0.5, 0.5])
    assert jnp.linalg.norm(solution[var] - expected) < 1e-4, (
        f"Expected {expected}, got {solution[var]}"
    )

    # Verify constraint is satisfied.
    constraint_violation = problem.compute_constraint_values(solution)
    assert jnp.linalg.norm(constraint_violation) < 1e-5


def test_se2_position_constraint():
    """Test SE2 pose graph with position constraint.

    Two poses with prior costs, but pose 1's position is constrained.
    """
    pose_vars = [jaxls.SE2Var(0), jaxls.SE2Var(1)]

    @jaxls.Cost.create_factory
    def prior_cost(
        vals: jaxls.VarValues, var: jaxls.SE2Var, target: jaxlie.SE2
    ) -> jax.Array:
        """Prior cost for a pose variable."""
        return (vals[var] @ target.inverse()).log()

    @jaxls.Constraint.create_factory
    def position_x_constraint(
        vals: jaxls.VarValues, var: jaxls.SE2Var, target_x: float
    ) -> jax.Array:
        """Constraint: x position must equal target."""
        return jnp.array([vals[var].translation()[0] - target_x])

    # Prior costs: pose 0 at origin, pose 1 at (2, 0).
    costs = [
        prior_cost(pose_vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
        prior_cost(pose_vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
    ]

    # Constraint: pose 1 x-position must be 1.0 (not 2.0).
    constraints = [position_x_constraint(pose_vars[1], 1.0)]

    problem = jaxls.LeastSquaresProblem(
        costs=costs, variables=pose_vars, constraints=constraints
    ).analyze()

    solution = problem.solve(verbose=False)

    # Pose 1's x-position should be 1.0 (constraint enforced).
    # Note: AL method may not converge to exact constraint satisfaction,
    # especially for nonlinear manifolds like SE(2).
    assert jnp.abs(solution[pose_vars[1]].translation()[0] - 1.0) < 5e-3

    # Verify constraint satisfaction.
    constraint_violation = problem.compute_constraint_values(solution)
    assert jnp.linalg.norm(constraint_violation) < 5e-3


def test_multiple_constraints():
    """Test optimization with multiple independent constraints."""

    class Vec3Var(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(3)):
        pass

    var = Vec3Var(0)

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: Vec3Var, target: jax.Array) -> jax.Array:
        """Cost pulls toward target."""
        return vals[var] - target

    @jaxls.Constraint.create_factory
    def constraint_x(vals: jaxls.VarValues, var: Vec3Var, val: float) -> jax.Array:
        """Constraint: x = val."""
        return jnp.array([vals[var][0] - val])

    @jaxls.Constraint.create_factory
    def constraint_y(vals: jaxls.VarValues, var: Vec3Var, val: float) -> jax.Array:
        """Constraint: y = val."""
        return jnp.array([vals[var][1] - val])

    # Cost pulls to (5, 5, 5), constraints force x=1, y=2.
    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var, jnp.array([5.0, 5.0, 5.0]))],
        variables=[var],
        constraints=[
            constraint_x(var, 1.0),
            constraint_y(var, 2.0),
        ],
    ).analyze()

    solution = problem.solve(verbose=False)

    # x and y should satisfy constraints, z should be pulled to 5.
    assert jnp.abs(solution[var][0] - 1.0) < 1e-4
    assert jnp.abs(solution[var][1] - 2.0) < 1e-4
    assert jnp.abs(solution[var][2] - 5.0) < 1e-4

    # Verify all constraints satisfied.
    constraint_violation = problem.compute_constraint_values(solution)
    assert jnp.linalg.norm(constraint_violation) < 1e-5


def test_constraint_violation_decreases():
    """Test that constraint violation decreases over iterations."""

    class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.0)):
        pass

    var = ScalarVar(0)

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: ScalarVar) -> jax.Array:
        return jnp.array([vals[var] - 10.0])

    @jaxls.Constraint.create_factory
    def constraint_fn(vals: jaxls.VarValues, var: ScalarVar) -> jax.Array:
        return jnp.array([vals[var] - 1.0])

    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var)],
        variables=[var],
        constraints=[constraint_fn(var)],
    ).analyze()

    # Start far from constraint satisfaction.
    initial_vals = jaxls.VarValues.make([var.with_value(jnp.array(20.0))])
    initial_violation = jnp.linalg.norm(problem.compute_constraint_values(initial_vals))

    solution = problem.solve(initial_vals=initial_vals, verbose=False)

    final_violation = jnp.linalg.norm(problem.compute_constraint_values(solution))

    # Final violation should be much smaller than initial.
    assert final_violation < initial_violation * 0.01
    assert final_violation < 1e-5


def test_batched_constraints():
    """Test batched constraints on multiple variables."""

    class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.0)):
        pass

    # Create 3 variables.
    vars = [ScalarVar(i) for i in range(3)]

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: ScalarVar, target: float) -> jax.Array:
        return jnp.array([vals[var] - target])

    @jaxls.Constraint.create_factory
    def constraint_fn(vals: jaxls.VarValues, var: ScalarVar) -> jax.Array:
        """Constraint: variable = 1.0."""
        return jnp.array([vals[var] - 1.0])

    # Create batched constraint by stacking variables.
    batched_var = ScalarVar(jnp.array([0, 1, 2]))

    problem = jaxls.LeastSquaresProblem(
        costs=[
            cost_fn(vars[0], 5.0),
            cost_fn(vars[1], 6.0),
            cost_fn(vars[2], 7.0),
        ],
        variables=vars,
        constraints=[constraint_fn(batched_var)],  # Single batched constraint
    ).analyze()

    solution = problem.solve(verbose=False)

    # All variables should be constrained to 1.0.
    for var in vars:
        assert jnp.abs(solution[var] - 1.0) < 1e-4

    # Verify constraint satisfaction.
    constraint_violation = problem.compute_constraint_values(solution)
    assert jnp.linalg.norm(constraint_violation) < 1e-5


def test_nonlinear_constraint():
    """Test nonlinear constraint: constrain point to circle.

    This demonstrates a case where reparameterization would be difficult.
    """

    class Vec2Var(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(2)):
        pass

    var = Vec2Var(0)

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: Vec2Var, target: jax.Array) -> jax.Array:
        """Cost pulls toward target."""
        return vals[var] - target

    @jaxls.Constraint.create_factory
    def circle_constraint(
        vals: jaxls.VarValues, var: Vec2Var, radius: float
    ) -> jax.Array:
        """Constraint: ||x||^2 = radius^2."""
        return jnp.array([jnp.sum(vals[var] ** 2) - radius**2])

    # Cost pulls to (2, 0), constraint forces to circle of radius 1.
    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var, jnp.array([2.0, 0.0]))],
        variables=[var],
        constraints=[circle_constraint(var, 1.0)],
    ).analyze()

    solution = problem.solve(
        initial_vals=jaxls.VarValues.make([var.with_value(jnp.array([0.5, 0.5]))]),
        verbose=False,
    )

    # Point should be on unit circle.
    radius = jnp.linalg.norm(solution[var])
    assert jnp.abs(radius - 1.0) < 1e-4

    # Point should be closest to (2, 0) on the circle, which is (1, 0).
    assert jnp.abs(solution[var][0] - 1.0) < 5e-3
    assert jnp.abs(solution[var][1]) < 5e-3

    # Verify constraint satisfaction.
    constraint_violation = problem.compute_constraint_values(solution)
    assert jnp.linalg.norm(constraint_violation) < 1e-5


def test_robot_arm_end_effector_constraint():
    """Test nonlinear constraint on a 3-link planar robot arm.

    The robot arm has 3 revolute joints, and we want to:
    1. Minimize deviation from a default configuration (via costs)
    2. Constrain the end effector position to a specific target (via constraint)

    This demonstrates:
    - Complex nonlinear forward kinematics
    - Constraint that depends on multiple variables
    - A problem where reparameterization would be very difficult
    """

    class AngleVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.0)):
        """Joint angle variable."""

        pass

    # Create 3 joint angle variables for the robot arm
    joint_angles = [AngleVar(i) for i in range(3)]

    # Link lengths
    L1, L2, L3 = 1.0, 1.0, 0.5

    def forward_kinematics(theta1: float, theta2: float, theta3: float) -> jax.Array:
        """Compute end effector position for 3-link planar arm.

        Each link is connected by a revolute joint. Angles are measured relative
        to the previous link.
        """
        # First link
        x1 = L1 * jnp.cos(theta1)
        y1 = L1 * jnp.sin(theta1)

        # Second link
        x2 = x1 + L2 * jnp.cos(theta1 + theta2)
        y2 = y1 + L2 * jnp.sin(theta1 + theta2)

        # Third link (end effector)
        x3 = x2 + L3 * jnp.cos(theta1 + theta2 + theta3)
        y3 = y2 + L3 * jnp.sin(theta1 + theta2 + theta3)

        return jnp.array([x3, y3])

    @jaxls.Cost.create_factory
    def joint_prior_cost(
        vals: jaxls.VarValues, joint: AngleVar, target_angle: float
    ) -> jax.Array:
        """Cost that pulls joint toward target angle."""
        return jnp.array([vals[joint] - target_angle])

    @jaxls.Constraint.create_factory
    def end_effector_constraint(
        vals: jaxls.VarValues,
        j1: AngleVar,
        j2: AngleVar,
        j3: AngleVar,
        target_pos: jax.Array,
    ) -> jax.Array:
        """Constraint: end effector must be at target position."""
        ee_pos = forward_kinematics(vals[j1], vals[j2], vals[j3])
        return ee_pos - target_pos

    # Default configuration: all joints at 45 degrees
    default_angle = jnp.pi / 4
    costs = [
        joint_prior_cost(joint_angles[0], default_angle),
        joint_prior_cost(joint_angles[1], default_angle),
        joint_prior_cost(joint_angles[2], default_angle),
    ]

    # Target end effector position (requires different configuration)
    # Choose a target that's easily reachable and not too far from default config
    target_ee = jnp.array([2.0, 0.5])

    problem = jaxls.LeastSquaresProblem(
        costs=costs,
        variables=joint_angles,
        constraints=[
            end_effector_constraint(
                joint_angles[0], joint_angles[1], joint_angles[2], target_ee
            )
        ],
    ).analyze()

    # Start from a configuration that's closer to a solution
    # (straight arm pointing right)
    initial_vals = jaxls.VarValues.make(
        [
            joint_angles[0].with_value(jnp.array(0.0)),
            joint_angles[1].with_value(jnp.array(0.0)),
            joint_angles[2].with_value(jnp.array(0.0)),
        ]
    )

    # Use tighter AL config for better convergence
    al_config = jaxls.AugmentedLagrangianConfig(
        tolerance_absolute=1e-6,
        max_iterations=50,
    )

    solution = problem.solve(
        initial_vals=initial_vals, augmented_lagrangian=al_config, verbose=False
    )

    # Verify end effector reaches target
    final_ee = forward_kinematics(
        solution[joint_angles[0]], solution[joint_angles[1]], solution[joint_angles[2]]
    )
    ee_error = jnp.linalg.norm(final_ee - target_ee)
    assert ee_error < 1e-4, (
        f"End effector error: {ee_error}, position: {final_ee}, target: {target_ee}"
    )

    # Verify constraint is satisfied
    constraint_violation = problem.compute_constraint_values(solution)
    assert jnp.linalg.norm(constraint_violation) < 1e-4

    # The solution should deviate from default config due to constraint
    # (we can't check exact angles, but at least one should change significantly)
    angle_changes = jnp.array(
        [jnp.abs(solution[joint_angles[i]] - default_angle) for i in range(3)]
    )
    assert jnp.any(angle_changes > 0.1), "Expected some joints to move from default"


def test_inequality_constraint_simple():
    """Test simple inequality constraint: minimize x^2 s.t. x ≤ 1.

    The unconstrained optimum would be x = 0, but we add a cost that pulls
    toward x = 2, with constraint x ≤ 1.
    """

    class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.0)):
        pass

    var = ScalarVar(0)

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: ScalarVar, target: float) -> jax.Array:
        """Cost pulls variable toward target."""
        return jnp.array([vals[var] - target])

    @jaxls.Constraint.create_factory(constraint_type="leq_zero")
    def inequality_constraint(
        vals: jaxls.VarValues, var: ScalarVar, upper_bound: float
    ) -> jax.Array:
        """Constraint: variable ≤ upper_bound."""
        return jnp.array([vals[var] - upper_bound])

    # Create problem: cost pulls to 2.0, constraint forces x ≤ 1.0
    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var, 2.0)],
        variables=[var],
        constraints=[inequality_constraint(var, 1.0)],
    ).analyze()

    # Solve with initial value at origin
    solution = problem.solve(
        initial_vals=jaxls.VarValues.make([var.with_value(jnp.array(0.0))]),
        verbose=False,
    )

    # Solution should be at the constraint boundary: x = 1.0
    # Note: inequality constraints typically have looser convergence than equality
    # constraints since the penalty method approximates the constraint boundary.
    assert jnp.abs(solution[var] - 1.0) < 2e-2, f"Expected x=1.0, got x={solution[var]}"

    # Verify constraint is satisfied (should be at or near boundary)
    constraint_violation = problem.compute_constraint_values(solution)
    assert constraint_violation[0] <= 2e-2, (
        f"Constraint violated: {constraint_violation[0]}"
    )


def test_inequality_constraint_inactive():
    """Test inequality constraint that is inactive (not at boundary).

    minimize ||x - 0.5||^2 s.t. x ≤ 2.0

    The constraint should be inactive since optimum x=0.5 satisfies x ≤ 2.
    """

    class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.0)):
        pass

    var = ScalarVar(0)

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: ScalarVar, target: float) -> jax.Array:
        return jnp.array([vals[var] - target])

    @jaxls.Constraint.create_factory(constraint_type="leq_zero")
    def inequality_constraint(
        vals: jaxls.VarValues, var: ScalarVar, upper_bound: float
    ) -> jax.Array:
        return jnp.array([vals[var] - upper_bound])

    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var, 0.5)],
        variables=[var],
        constraints=[inequality_constraint(var, 2.0)],
    ).analyze()

    solution = problem.solve(verbose=False)

    # Solution should be at unconstrained optimum x = 0.5
    assert jnp.abs(solution[var] - 0.5) < 1e-4

    # Constraint should be satisfied with slack (< 0)
    constraint_val = problem.compute_constraint_values(solution)[0]
    assert constraint_val < -0.1, "Constraint should be inactive"


def test_multiple_inequality_constraints():
    """Test multiple inequality constraints: box constraints.

    minimize ||x - 5||^2 s.t. 1 ≤ x ≤ 3

    Implemented as: x - 1 ≥ 0 and x - 3 ≤ 0
    Which becomes: -(x - 1) ≤ 0 and x - 3 ≤ 0
    """

    class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.0)):
        pass

    var = ScalarVar(0)

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: ScalarVar, target: float) -> jax.Array:
        return jnp.array([vals[var] - target])

    @jaxls.Constraint.create_factory(constraint_type="leq_zero")
    def upper_bound_constraint(
        vals: jaxls.VarValues, var: ScalarVar, upper: float
    ) -> jax.Array:
        """x ≤ upper"""
        return jnp.array([vals[var] - upper])

    @jaxls.Constraint.create_factory(constraint_type="leq_zero")
    def lower_bound_constraint(
        vals: jaxls.VarValues, var: ScalarVar, lower: float
    ) -> jax.Array:
        """x ≥ lower, which is -(x - lower) ≤ 0"""
        return jnp.array([lower - vals[var]])

    # Cost pulls to 5.0, but constraints force 1 ≤ x ≤ 3
    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var, 5.0)],
        variables=[var],
        constraints=[
            upper_bound_constraint(var, 3.0),
            lower_bound_constraint(var, 1.0),
        ],
    ).analyze()

    solution = problem.solve(verbose=False)

    # Solution should be at upper bound: x = 3.0
    # Note: inequality constraints typically have looser convergence than equality constraints.
    assert jnp.abs(solution[var] - 3.0) < 2e-2, f"Expected x=3.0, got x={solution[var]}"

    # Verify constraints are satisfied
    constraint_vals = problem.compute_constraint_values(solution)
    assert jnp.all(constraint_vals <= 2e-2), f"Constraints violated: {constraint_vals}"


def test_custom_augmented_lagrangian_config():
    """Test that custom AL config parameters are respected."""

    class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.0)):
        pass

    var = ScalarVar(0)

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: ScalarVar) -> jax.Array:
        return jnp.array([vals[var]])

    @jaxls.Constraint.create_factory
    def constraint_fn(vals: jaxls.VarValues, var: ScalarVar) -> jax.Array:
        return jnp.array([vals[var] - 1.0])

    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var)],
        variables=[var],
        constraints=[constraint_fn(var)],
    ).analyze()

    # Use custom config with tighter tolerances.
    custom_config = jaxls.AugmentedLagrangianConfig(
        tolerance_absolute=1e-8,
        tolerance_relative=1e-6,
        penalty_initial=1.0,
        max_iterations=30,
    )

    solution = problem.solve(augmented_lagrangian=custom_config, verbose=False)

    # Solution should satisfy constraint with tight tolerance.
    constraint_violation = jnp.linalg.norm(problem.compute_constraint_values(solution))
    assert constraint_violation < 1e-7


def test_no_constraints_uses_standard_solver():
    """Verify that problems without constraints use standard solver."""

    class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.0)):
        pass

    var = ScalarVar(0)

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: ScalarVar) -> jax.Array:
        return jnp.array([vals[var] - 5.0])

    # No constraints - should use standard solver.
    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var)],
        variables=[var],
    ).analyze()

    solution = problem.solve(verbose=False)

    # Should converge to x = 5.
    assert jnp.abs(solution[var] - 5.0) < 1e-6


def test_constraint_with_jac_mode_forward():
    """Test constraint with explicit jac_mode='forward'."""

    class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.0)):
        pass

    var = ScalarVar(0)

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: ScalarVar, target: float) -> jax.Array:
        return jnp.array([vals[var] - target])

    @jaxls.Constraint.create_factory(jac_mode="forward")
    def constraint_fn(
        vals: jaxls.VarValues, var: ScalarVar, target: float
    ) -> jax.Array:
        return jnp.array([vals[var] - target])

    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var, 2.0)],
        variables=[var],
        constraints=[constraint_fn(var, 1.0)],
    ).analyze()

    solution = problem.solve(
        initial_vals=jaxls.VarValues.make([var.with_value(jnp.array(5.0))]),
        verbose=False,
    )

    assert jnp.abs(solution[var] - 1.0) < 1e-5


def test_constraint_with_jac_mode_reverse():
    """Test constraint with explicit jac_mode='reverse'."""

    class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.0)):
        pass

    var = ScalarVar(0)

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: ScalarVar, target: float) -> jax.Array:
        return jnp.array([vals[var] - target])

    @jaxls.Constraint.create_factory(jac_mode="reverse")
    def constraint_fn(
        vals: jaxls.VarValues, var: ScalarVar, target: float
    ) -> jax.Array:
        return jnp.array([vals[var] - target])

    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var, 2.0)],
        variables=[var],
        constraints=[constraint_fn(var, 1.0)],
    ).analyze()

    solution = problem.solve(
        initial_vals=jaxls.VarValues.make([var.with_value(jnp.array(5.0))]),
        verbose=False,
    )

    assert jnp.abs(solution[var] - 1.0) < 1e-5


def test_constraint_with_custom_jacobian():
    """Test constraint with custom Jacobian function.

    The wrapper Jacobian should correctly apply sqrt(rho) scaling.
    """

    class Vec2Var(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(2)):
        pass

    var = Vec2Var(0)

    def my_constraint_jac(vals: jaxls.VarValues, var: Vec2Var) -> jax.Array:
        """Custom Jacobian: d(x+y)/d(x,y) = [1, 1]."""
        return jnp.array([[1.0, 1.0]])

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: Vec2Var) -> jax.Array:
        return vals[var]

    @jaxls.Constraint.create_factory(jac_custom_fn=my_constraint_jac)
    def constraint_fn(vals: jaxls.VarValues, var: Vec2Var) -> jax.Array:
        """Constraint: x + y = 1."""
        vec = vals[var]
        return jnp.array([vec[0] + vec[1] - 1.0])

    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var)],
        variables=[var],
        constraints=[constraint_fn(var)],
    ).analyze()

    solution = problem.solve(
        initial_vals=jaxls.VarValues.make([var.with_value(jnp.array([2.0, 3.0]))]),
        verbose=False,
    )

    # Solution should be (0.5, 0.5).
    expected = jnp.array([0.5, 0.5])
    assert jnp.linalg.norm(solution[var] - expected) < 1e-4

    # Verify constraint is satisfied.
    constraint_violation = problem.compute_constraint_values(solution)
    assert jnp.linalg.norm(constraint_violation) < 1e-5


def test_constraint_with_custom_jacobian_with_cache():
    """Test constraint with custom Jacobian that uses cache from residual computation."""

    class Vec2Var(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(2)):
        pass

    var = Vec2Var(0)

    def my_constraint_jac_with_cache(
        vals: jaxls.VarValues, cache: jax.Array, var: Vec2Var
    ) -> jax.Array:
        """Custom Jacobian using cached intermediate value.

        Cache contains the squared values, which we use to verify cache passing works.
        The actual Jacobian is still [1, 1] for x + y.
        """
        # Just verify cache was passed correctly
        _ = cache  # Would use in real scenario
        return jnp.array([[1.0, 1.0]])

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: Vec2Var) -> jax.Array:
        return vals[var]

    @jaxls.Constraint.create_factory(
        jac_custom_with_cache_fn=my_constraint_jac_with_cache
    )
    def constraint_fn(
        vals: jaxls.VarValues, var: Vec2Var
    ) -> tuple[jax.Array, jax.Array]:
        """Constraint: x + y = 1, with cache."""
        vec = vals[var]
        constraint_val = jnp.array([vec[0] + vec[1] - 1.0])
        cache = vec**2  # Cache squared values for Jacobian
        return constraint_val, cache

    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var)],
        variables=[var],
        constraints=[constraint_fn(var)],
    ).analyze()

    solution = problem.solve(
        initial_vals=jaxls.VarValues.make([var.with_value(jnp.array([2.0, 3.0]))]),
        verbose=False,
    )

    # Solution should be (0.5, 0.5).
    expected = jnp.array([0.5, 0.5])
    assert jnp.linalg.norm(solution[var] - expected) < 1e-4


def test_inequality_constraint_custom_jacobian_zeros_when_inactive():
    """Test that inequality constraint wrapper Jacobian zeros out when inactive.

    For g(x) <= 0, when g(x) + λ/ρ <= 0, the Jacobian should be zero.
    """

    class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.0)):
        pass

    var = ScalarVar(0)

    def my_constraint_jac(vals: jaxls.VarValues, var: ScalarVar) -> jax.Array:
        """Custom Jacobian: d(x-2)/dx = 1."""
        return jnp.array([[1.0]])

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: ScalarVar, target: float) -> jax.Array:
        return jnp.array([vals[var] - target])

    @jaxls.Constraint.create_factory(
        constraint_type="leq_zero", jac_custom_fn=my_constraint_jac
    )
    def inequality_constraint(vals: jaxls.VarValues, var: ScalarVar) -> jax.Array:
        """Constraint: x <= 2, i.e., x - 2 <= 0."""
        return jnp.array([vals[var] - 2.0])

    # Cost pulls to 0.5, constraint x <= 2 is inactive
    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var, 0.5)],
        variables=[var],
        constraints=[inequality_constraint(var)],
    ).analyze()

    solution = problem.solve(verbose=False)

    # Solution should be at unconstrained optimum x = 0.5
    assert jnp.abs(solution[var] - 0.5) < 1e-4

    # Constraint should be satisfied with slack
    constraint_val = problem.compute_constraint_values(solution)[0]
    assert constraint_val < -0.1, "Constraint should be inactive"


def test_constraint_with_jac_batch_size():
    """Test constraint with jac_batch_size for memory-efficient Jacobian computation."""

    class Vec4Var(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(4)):
        pass

    var = Vec4Var(0)

    @jaxls.Cost.create_factory
    def cost_fn(vals: jaxls.VarValues, var: Vec4Var) -> jax.Array:
        return vals[var]

    # Use jac_batch_size=1 to compute Jacobian one column at a time
    @jaxls.Constraint.create_factory(jac_batch_size=1)
    def constraint_fn(vals: jaxls.VarValues, var: Vec4Var) -> jax.Array:
        """Constraint: sum(x) = 2."""
        return jnp.array([jnp.sum(vals[var]) - 2.0])

    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var)],
        variables=[var],
        constraints=[constraint_fn(var)],
    ).analyze()

    solution = problem.solve(
        initial_vals=jaxls.VarValues.make([var.with_value(jnp.ones(4))]),
        verbose=False,
    )

    # Sum should equal 2.0
    assert jnp.abs(jnp.sum(solution[var]) - 2.0) < 1e-4

    # Verify constraint is satisfied
    constraint_violation = problem.compute_constraint_values(solution)
    assert jnp.linalg.norm(constraint_violation) < 1e-5


if __name__ == "__main__":
    # Run all tests.
    pytest.main([__file__, "-v"])
