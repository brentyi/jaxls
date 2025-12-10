"""Constrained pose graph optimization example.

This example demonstrates using equality constraints to fix landmark positions
while optimizing robot poses. This is useful when for landmarks with known
coordinates, loop closure, etc.
"""

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as onp
import viser


def lift_se2_to_se3(pose: jaxlie.SE2) -> jaxlie.SE3:
    """Lift SE2 pose to SE3 for 3D visualization."""
    matrix = onp.eye(4) + onp.zeros((*pose.get_batch_axes(), 4, 4))
    matrix[..., :2, :2] = pose.rotation().as_matrix()
    matrix[..., :2, 3] = pose.translation()
    return jaxlie.SE3.from_matrix(matrix)


def main():
    """Constrained pose graph with fixed landmark."""

    # Create 3 poses forming a path.
    pose_vars = [jaxls.SE2Var(i) for i in range(3)]

    @jaxls.Cost.factory
    def prior_cost(
        vals: jaxls.VarValues, var: jaxls.SE2Var, target: jaxlie.SE2
    ) -> jax.Array:
        """Prior cost: penalizes deviation from target pose."""
        return (vals[var] @ target.inverse()).log()

    @jaxls.Cost.factory
    def between_cost(
        vals: jaxls.VarValues,
        var0: jaxls.SE2Var,
        var1: jaxls.SE2Var,
        delta: jaxlie.SE2,
    ) -> jax.Array:
        """Between cost: penalizes deviation from relative pose."""
        return ((vals[var0].inverse() @ vals[var1]) @ delta.inverse()).log()

    @jaxls.Cost.factory(kind="constraint_eq_zero")
    def position_constraint(
        vals: jaxls.VarValues, var: jaxls.SE2Var, target_xy: jax.Array
    ) -> jax.Array:
        """Constraint: pose position must equal target."""
        return vals[var].translation() - target_xy

    # Scenario: Robot travels along path, but we have GPS measurement at pose 2.
    # Prior costs give rough estimates of poses.
    # GPS measurement says pose 2 is at (2.5, 1.0). Contradicts noisy priors.
    costs = [
        prior_cost(pose_vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
        prior_cost(pose_vars[1], jaxlie.SE2.from_xy_theta(1.0, 0.5, 0.2)),
        prior_cost(pose_vars[2], jaxlie.SE2.from_xy_theta(2.0, 0.8, 0.3)),
        # Between factors from odometry (noisy measurements).
        between_cost(
            pose_vars[0], pose_vars[1], jaxlie.SE2.from_xy_theta(1.1, 0.4, 0.15)
        ),
        between_cost(
            pose_vars[1], pose_vars[2], jaxlie.SE2.from_xy_theta(0.9, 0.3, 0.1)
        ),
        # GPS constraint on pose 2.
        position_constraint(pose_vars[2], jnp.array([2.5, 1.0])),
    ]

    print("=" * 60)
    print("Constrained Pose Graph Optimization")
    print("=" * 60)
    print()
    print("Setup:")
    print("  - 3 poses along a path")
    print("  - Prior costs from noisy estimates")
    print("  - Odometry measurements (between factors)")
    print("  - GPS constraint: Pose 2 must be at (2.5, 1.0)")
    print()

    # Build and analyze problem.
    problem = jaxls.LeastSquaresProblem(costs=costs, variables=pose_vars).analyze()

    print("Solving constrained optimization...")
    print()

    # Solve the constrained problem.
    solution = jax.jit(problem.solve)()

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()

    for i, var in enumerate(pose_vars):
        pose = solution[var]
        print(f"Pose {i}:")
        print(f"  Position: ({pose.translation()[0]:.4f}, {pose.translation()[1]:.4f})")
        print(f"  Rotation: {float(pose.rotation().log()[0]):.4f} rad")

    # Verify constraint satisfaction.
    constraint_violation = problem.compute_constraint_values(solution)
    print()
    print(f"Constraint violation: {jnp.linalg.norm(constraint_violation):.6e}")
    print()

    # Compare with unconstrained solution.
    print("=" * 60)
    print("Comparison: Unconstrained Optimization")
    print("=" * 60)
    print()

    unconstrained_problem = jaxls.LeastSquaresProblem(
        costs=costs,
        variables=pose_vars,
    ).analyze()

    unconstrained_solution = jax.jit(unconstrained_problem.solve)()

    print("Unconstrained solution (no GPS constraint):")
    for i, var in enumerate(pose_vars):
        pose = unconstrained_solution[var]
        print(f"Pose {i}:")
        print(f"  Position: ({pose.translation()[0]:.4f}, {pose.translation()[1]:.4f})")
        print(f"  Rotation: {float(pose.rotation().log()[0]):.4f} rad")

    print()
    print("Pose 2 difference (constrained vs unconstrained):")
    constrained_pos = solution[pose_vars[2]].translation()
    unconstrained_pos = unconstrained_solution[pose_vars[2]].translation()
    diff = jnp.linalg.norm(constrained_pos - unconstrained_pos)
    print(f"  Position difference: {diff:.4f}")
    print(f"  Constrained: ({constrained_pos[0]:.4f}, {constrained_pos[1]:.4f})")
    print(f"  Unconstrained: ({unconstrained_pos[0]:.4f}, {unconstrained_pos[1]:.4f})")
    print("  GPS target: (2.5000, 1.0000)")
    print()
    print("The constraint successfully pulls Pose 2 to match the GPS measurement!")
    print()

    print("Starting Viser visualization server...")
    server = viser.ViserServer()

    initial_vals = jaxls.VarValues.make(
        [
            pose_vars[0].with_value(jaxlie.SE2.from_xy_theta(-0.2, 0.1, 0.1)),
            pose_vars[1].with_value(jaxlie.SE2.from_xy_theta(0.8, 0.3, -0.1)),
            pose_vars[2].with_value(jaxlie.SE2.from_xy_theta(1.8, 0.5, 0.2)),
        ]
    )

    initial_poses = [initial_vals[var] for var in pose_vars]
    constrained_poses = [solution[var] for var in pose_vars]
    unconstrained_poses = [unconstrained_solution[var] for var in pose_vars]

    def get_position_3d(pose: jaxlie.SE2) -> tuple[float, float, float]:
        t = pose.translation()
        return (float(t[0]), float(t[1]), 0.0)

    def get_quaternion(pose: jaxlie.SE2) -> tuple[float, float, float, float]:
        angle = float(pose.rotation().log()[0])
        return (
            float(onp.cos(angle / 2)),
            0.0,
            0.0,
            float(onp.sin(angle / 2)),
        )

    # Initial poses (blue).
    init_positions = onp.array([get_position_3d(p) for p in initial_poses])
    init_wxyzs = onp.array([get_quaternion(p) for p in initial_poses])
    server.scene.add_batched_axes(
        "/initial/poses",
        batched_wxyzs=init_wxyzs,
        batched_positions=init_positions,
        axes_length=0.15,
        axes_radius=0.01,
    )
    server.scene.add_line_segments(
        "/initial/path",
        points=onp.stack([init_positions[:-1], init_positions[1:]], axis=1),
        colors=(100, 100, 200),
        line_width=2.0,
    )

    # Constrained solution (green).
    constrained_positions = onp.array([get_position_3d(p) for p in constrained_poses])
    constrained_wxyzs = onp.array([get_quaternion(p) for p in constrained_poses])
    server.scene.add_batched_axes(
        "/constrained/poses",
        batched_wxyzs=constrained_wxyzs,
        batched_positions=constrained_positions,
        axes_length=0.15,
        axes_radius=0.01,
    )
    server.scene.add_line_segments(
        "/constrained/path",
        points=onp.stack(
            [constrained_positions[:-1], constrained_positions[1:]], axis=1
        ),
        colors=(0, 200, 0),
        line_width=3.0,
    )

    # Unconstrained solution (red).
    unconstrained_positions = onp.array(
        [get_position_3d(p) for p in unconstrained_poses]
    )
    unconstrained_wxyzs = onp.array([get_quaternion(p) for p in unconstrained_poses])
    server.scene.add_batched_axes(
        "/unconstrained/poses",
        batched_wxyzs=unconstrained_wxyzs,
        batched_positions=unconstrained_positions,
        axes_length=0.15,
        axes_radius=0.01,
    )
    server.scene.add_line_segments(
        "/unconstrained/path",
        points=onp.stack(
            [unconstrained_positions[:-1], unconstrained_positions[1:]], axis=1
        ),
        colors=(200, 100, 100),
        line_width=2.0,
    )

    # GPS target (yellow).
    gps_target = jnp.array([2.5, 1.0])
    server.scene.add_icosphere(
        "/gps_target",
        radius=0.08,
        position=(float(gps_target[0]), float(gps_target[1]), 0.0),
        color=(255, 255, 0),
    )

    print(
        f"Visualization server running at http://{server.get_host()}:{server.get_port()}"
    )
    server.sleep_forever()


if __name__ == "__main__":
    main()
