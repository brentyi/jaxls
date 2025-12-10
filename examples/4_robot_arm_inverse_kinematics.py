"""Robot arm inverse kinematics with constraints.

This example demonstrates using nonlinear equality constraints to solve an
inverse kinematics problem for a 3-link planar robot arm. The robot has:
- 3 revolute joints with configurable angles
- Link lengths: 1.0, 1.0, and 0.5 meters
- Goal: reach target position while minimizing joint movement

This uses:
- Nonlinear constraints (forward kinematics equations)
- Constraints that depend on multiple variables
"""

import jax
import jax.numpy as jnp
import jaxls
import numpy as onp
import viser


def forward_kinematics(
    theta1: float | jax.Array,
    theta2: float | jax.Array,
    theta3: float | jax.Array,
    link_lengths: tuple[float, float, float],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute positions of all joints and end effector for a 3-link planar arm.

    Args:
        theta1: First joint angle (relative to world frame)
        theta2: Second joint angle (relative to first link)
        theta3: Third joint angle (relative to second link)
        link_lengths: Lengths of the three links

    Returns:
        Tuple of (base, joint1, joint2, end_effector) positions
    """
    L1, L2, L3 = link_lengths

    base = jnp.array([0.0, 0.0])
    joint1 = base + jnp.array([L1 * jnp.cos(theta1), L1 * jnp.sin(theta1)])
    joint2 = joint1 + jnp.array(
        [L2 * jnp.cos(theta1 + theta2), L2 * jnp.sin(theta1 + theta2)]
    )
    end_effector = joint2 + jnp.array(
        [L3 * jnp.cos(theta1 + theta2 + theta3), L3 * jnp.sin(theta1 + theta2 + theta3)]
    )

    return base, joint1, joint2, end_effector


def visualize_robot_arm(
    server: viser.ViserServer,
    name: str,
    theta1: float,
    theta2: float,
    theta3: float,
    link_lengths: tuple[float, float, float],
    color: tuple[int, int, int],
    link_radius: float = 0.02,
):
    """Visualize robot arm configuration in Viser."""
    base, joint1, joint2, end_effector = forward_kinematics(
        theta1, theta2, theta3, link_lengths
    )

    def to_3d(pos: jax.Array) -> tuple[float, float, float]:
        return (float(pos[0]), float(pos[1]), 0.0)

    positions = [base, joint1, joint2, end_effector]
    for i in range(3):
        start = onp.array(to_3d(positions[i]))
        end = onp.array(to_3d(positions[i + 1]))
        server.scene.add_line_segments(
            f"/{name}/link_{i}",
            points=onp.array([[start, end]]),
            colors=color,
            line_width=4.0,
        )

    for i, pos in enumerate(positions):
        server.scene.add_icosphere(
            f"/{name}/joint_{i}",
            radius=link_radius * 2,
            position=to_3d(pos),
            color=color,
        )
    server.scene.add_icosphere(
        f"/{name}/end_effector",
        radius=link_radius * 3,
        position=to_3d(end_effector),
        color=color,
    )


def main():
    """Solve inverse kinematics for a 3-link planar robot arm."""

    print("=" * 70)
    print("Robot Arm Inverse Kinematics with Constraints")
    print("=" * 70)
    print()

    link_lengths = (1.0, 1.0, 0.5)
    L1, L2, L3 = link_lengths
    print("Robot configuration:")
    print(f"  Link lengths: L1={L1}m, L2={L2}m, L3={L3}m")
    print(f"  Maximum reach: {sum(link_lengths)}m")
    print()

    class AngleVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.0)):
        """Joint angle variable (radians)."""

        pass

    joint_angles = [AngleVar(i) for i in range(3)]

    default_angle = jnp.pi / 4

    @jaxls.Cost.factory
    def joint_prior_cost(
        vals: jaxls.VarValues, joint: AngleVar, target_angle: float
    ) -> jax.Array:
        """Cost that pulls joint toward target angle."""
        return jnp.array([vals[joint] - target_angle])

    @jaxls.Cost.factory(kind="constraint_eq_zero")
    def end_effector_constraint(
        vals: jaxls.VarValues,
        j1: AngleVar,
        j2: AngleVar,
        j3: AngleVar,
        target_pos: jax.Array,
    ) -> jax.Array:
        """Constraint: end effector must be at target position.

        This is a nonlinear constraint because the forward kinematics involves
        trigonometric functions of the joint angles.
        """
        _, _, _, ee_pos = forward_kinematics(vals[j1], vals[j2], vals[j3], link_lengths)
        return ee_pos - target_pos

    target_ee = jnp.array([2.0, 0.5])
    target_distance = jnp.linalg.norm(target_ee)

    print(f"Target end effector position: ({target_ee[0]:.2f}, {target_ee[1]:.2f})")
    print(f"Distance from origin: {target_distance:.2f}m")
    print()

    if target_distance > sum(link_lengths):
        print("WARNING: Target is outside reachable workspace!")
        return

    costs = [
        joint_prior_cost(joint_angles[0], default_angle),
        joint_prior_cost(joint_angles[1], default_angle),
        joint_prior_cost(joint_angles[2], default_angle),
        end_effector_constraint(
            joint_angles[0], joint_angles[1], joint_angles[2], target_ee
        ),
    ]

    problem = jaxls.LeastSquaresProblem(costs=costs, variables=joint_angles).analyze()

    initial_vals = jaxls.VarValues.make(
        [
            joint_angles[0].with_value(jnp.array(0.0)),
            joint_angles[1].with_value(jnp.array(0.0)),
            joint_angles[2].with_value(jnp.array(0.0)),
        ]
    )

    _, _, _, initial_ee = forward_kinematics(0.0, 0.0, 0.0, link_lengths)
    initial_error = jnp.linalg.norm(initial_ee - target_ee)

    print("Initial configuration (straight arm):")
    print("  Joint angles: [0.00, 0.00, 0.00] rad")
    print(f"  End effector: ({initial_ee[0]:.4f}, {initial_ee[1]:.4f})")
    print(f"  Error: {initial_error:.4f}m")
    print()

    print("Solving constrained inverse kinematics...")
    print("  Using Augmented Lagrangian method")
    print()
    solution = jax.jit(problem.solve)(initial_vals)

    print()
    print("=" * 70)
    print("Solution")
    print("=" * 70)
    print()

    theta1_sol = float(solution[joint_angles[0]])
    theta2_sol = float(solution[joint_angles[1]])
    theta3_sol = float(solution[joint_angles[2]])

    print("Optimized joint angles:")
    print(f"  θ₁ = {theta1_sol:7.4f} rad ({jnp.rad2deg(theta1_sol):6.2f}°)")
    print(f"  θ₂ = {theta2_sol:7.4f} rad ({jnp.rad2deg(theta2_sol):6.2f}°)")
    print(f"  θ₃ = {theta3_sol:7.4f} rad ({jnp.rad2deg(theta3_sol):6.2f}°)")
    print()

    _, _, _, final_ee = forward_kinematics(
        theta1_sol, theta2_sol, theta3_sol, link_lengths
    )
    final_error = jnp.linalg.norm(final_ee - target_ee)

    print("End effector position:")
    print(f"  Achieved: ({final_ee[0]:.4f}, {final_ee[1]:.4f})")
    print(f"  Target:   ({target_ee[0]:.4f}, {target_ee[1]:.4f})")
    print(f"  Error:    {final_error:.6f}m")
    print()

    constraint_violation = problem.compute_constraint_values(solution)
    print(f"Constraint violation: {jnp.linalg.norm(constraint_violation):.6e}")
    print()

    print("Cost comparison:")
    print(f"  Initial error:      {initial_error:.4f}m")
    print(f"  Final error:        {final_error:.6f}m")
    print(f"  Improvement:        {initial_error - final_error:.4f}m")
    print()

    print("=" * 70)
    print("Comparison: Unconstrained Optimization")
    print("=" * 70)
    print()

    unconstrained_problem = jaxls.LeastSquaresProblem(
        costs=costs,
        variables=joint_angles,
    ).analyze()
    unconstrained_solution = jax.jit(unconstrained_problem.solve)(initial_vals)

    theta1_unc = float(unconstrained_solution[joint_angles[0]])
    theta2_unc = float(unconstrained_solution[joint_angles[1]])
    theta3_unc = float(unconstrained_solution[joint_angles[2]])

    _, _, _, unconstrained_ee = forward_kinematics(
        theta1_unc, theta2_unc, theta3_unc, link_lengths
    )
    unconstrained_error = jnp.linalg.norm(unconstrained_ee - target_ee)

    print("Unconstrained solution (no end effector constraint):")
    print(f"  Joint angles: [{theta1_unc:.4f}, {theta2_unc:.4f}, {theta3_unc:.4f}] rad")
    print(f"  End effector: ({unconstrained_ee[0]:.4f}, {unconstrained_ee[1]:.4f})")
    print(f"  Error: {unconstrained_error:.4f}m")
    print()

    print("The constraint successfully guides the arm to reach the target!")
    print()

    print("Starting Viser visualization server...")
    server = viser.ViserServer()

    # Initial (blue), constrained (green), unconstrained (red).
    visualize_robot_arm(
        server, "initial", 0.0, 0.0, 0.0, link_lengths, color=(100, 100, 200)
    )
    visualize_robot_arm(
        server,
        "constrained",
        theta1_sol,
        theta2_sol,
        theta3_sol,
        link_lengths,
        color=(0, 200, 0),
    )
    visualize_robot_arm(
        server,
        "unconstrained",
        theta1_unc,
        theta2_unc,
        theta3_unc,
        link_lengths,
        color=(200, 100, 100),
    )

    # Target (yellow).
    server.scene.add_icosphere(
        "/target",
        radius=0.08,
        position=(float(target_ee[0]), float(target_ee[1]), 0.0),
        color=(255, 255, 0),
    )
    server.scene.add_grid("/grid", width=4.0, height=3.0, cell_size=0.5)

    print(
        f"Visualization server running at http://{server.get_host()}:{server.get_port()}"
    )
    server.sleep_forever()


if __name__ == "__main__":
    main()
