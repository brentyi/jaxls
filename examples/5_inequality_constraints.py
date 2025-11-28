"""Inequality constraints example: path planning with obstacles.

We have:
- A sequence of waypoints that we want to be smooth
- Circular obstacles that waypoints must avoid
- Goal: minimize path length while avoiding obstacles

This relies on:
- Inequality constraints using constraint_type="leq_zero"
- Multiple waypoints and obstacles
- Active vs inactive constraints (waypoints near vs far from obstacles)
"""

import jax
import jax.numpy as jnp
import jaxls
import numpy as onp
import viser


def main():
    """Plan a smooth path that avoids circular obstacles."""

    print("=" * 70)
    print("Path Planning with Obstacle Avoidance (Inequality Constraints)")
    print("=" * 70)
    print()

    class WaypointVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(2)):
        """2D waypoint position."""

    num_waypoints = 50
    waypoints = [WaypointVar(i) for i in range(num_waypoints)]

    start = jnp.array([0.0, 0.0])
    goal = jnp.array([4.0, 0.0])

    print("Path planning problem:")
    print(f"  Start: ({start[0]:.1f}, {start[1]:.1f})")
    print(f"  Goal:  ({goal[0]:.1f}, {goal[1]:.1f})")
    print(f"  Waypoints: {num_waypoints}")
    print()

    # Obstacles positioned to block the straight-line path.
    obstacles = [
        (jnp.array([1.0, 0.3]), 0.6),  # Obstacle 1: blocks waypoint 1
        (jnp.array([3.0, -0.3]), 0.6),  # Obstacle 2: blocks waypoint 3
    ]

    print("Obstacles:")
    for i, (center, radius) in enumerate(obstacles):
        print(
            f"  {i + 1}. Center: ({center[0]:.1f}, {center[1]:.1f}), Radius: {radius:.1f}m"
        )
    print()

    # Smoothness cost: minimize distance between consecutive waypoints.
    @jaxls.Cost.create_factory
    def smoothness_cost(
        vals: jaxls.VarValues, wp1: WaypointVar, wp2: WaypointVar
    ) -> jax.Array:
        """Cost penalizes large gaps between waypoints."""
        return vals[wp1] - vals[wp2]

    @jaxls.Constraint.create_factory(constraint_type="eq_zero")
    def fix_waypoint(
        vals: jaxls.VarValues, waypoint: WaypointVar, target: jax.Array
    ) -> jax.Array:
        """Equality constraint: waypoint = target."""
        return vals[waypoint] - target

    @jaxls.Constraint.create_factory(constraint_type="leq_zero")
    def obstacle_avoidance(
        vals: jaxls.VarValues,
        waypoint: WaypointVar,
        obstacle_center: jax.Array,
        obstacle_radius: jax.Array,
    ) -> jax.Array:
        """Inequality: waypoint must be outside obstacle circle.

        Constraint: r^2 - ||p - c||^2 <= 0
        which ensures ||p - c|| >= r (waypoint is outside circle)
        """
        wp_pos = vals[waypoint]
        dist_sq = jnp.sum((wp_pos - obstacle_center) ** 2)
        return jnp.array([obstacle_radius**2 - dist_sq])

    costs = []
    for i in range(num_waypoints - 1):
        costs.append(smoothness_cost(waypoints[i], waypoints[i + 1]))

    constraints = [
        fix_waypoint(waypoints[0], start),
        fix_waypoint(waypoints[-1], goal),
    ]

    # Add obstacle avoidance for intermediate waypoints.
    for i in range(1, num_waypoints - 1):
        for obs_center, obs_radius in obstacles:
            constraints.append(
                obstacle_avoidance(waypoints[i], obs_center, jnp.array(obs_radius))
            )

    problem = jaxls.LeastSquaresProblem(
        costs=costs,
        variables=waypoints,
        constraints=constraints,
    ).analyze()

    # Initial guess: straight line from start to goal.
    initial_vals = jaxls.VarValues.make(
        [
            waypoint.with_value(start + (goal - start) * i / (num_waypoints - 1))
            for i, waypoint in enumerate(waypoints)
        ]
    )

    print("Initial path (straight line):")
    for i, wp in enumerate(waypoints):
        pos = initial_vals[wp]
        print(f"  Waypoint {i}: ({pos[0]:.2f}, {pos[1]:.2f})")

    # Check initial violations. First 4 values are equality constraints.
    initial_constraint_vals = problem.compute_constraint_values(initial_vals)
    initial_violations = initial_constraint_vals[4:]
    print()
    print(
        f"Initial obstacle constraint violations: {jnp.sum(initial_violations > 0)} / {len(initial_violations)}"
    )
    print()

    print("Solving with obstacle avoidance constraints...")
    print()

    solution = problem.solve(initial_vals=initial_vals)

    print("=" * 70)
    print("Solution")
    print("=" * 70)
    print()

    print("Optimized path:")
    for i, wp in enumerate(waypoints):
        pos = solution[wp]
        print(f"  Waypoint {i}: ({pos[0]:.2f}, {pos[1]:.2f})")

    print()

    # Check which constraints are active.
    final_constraint_vals = problem.compute_constraint_values(solution)
    inequality_constraints = final_constraint_vals[4:]

    print("Obstacle avoidance status:")
    constraint_idx = 0
    for i in range(1, num_waypoints - 1):
        wp_pos = solution[waypoints[i]]
        print(f"  Waypoint {i}:")

        for obs_idx, (obs_center, obs_radius) in enumerate(obstacles):
            dist = jnp.linalg.norm(wp_pos - obs_center)
            clearance = dist - obs_radius
            constraint_val = inequality_constraints[constraint_idx]

            status = ""
            if constraint_val > -0.01:
                status = " (ACTIVE - near obstacle)"
            else:
                status = f" (inactive, clearance: {clearance:.2f}m)"

            print(f"    Obstacle {obs_idx + 1}: distance={dist:.2f}m{status}")
            constraint_idx += 1

    print()

    violations = jnp.sum(inequality_constraints > 1e-3)
    print(f"Final constraint violations: {violations} / {len(inequality_constraints)}")

    if violations == 0:
        print("✓ All obstacles successfully avoided!")
    print()

    path_length = 0.0
    for i in range(num_waypoints - 1):
        seg_length = jnp.linalg.norm(
            solution[waypoints[i + 1]] - solution[waypoints[i]]
        )
        path_length += seg_length

    straight_line_length = jnp.linalg.norm(goal - start)

    print("Path statistics:")
    print(f"  Path length: {path_length:.2f}m")
    print(f"  Straight line: {straight_line_length:.2f}m")
    print(f"  Detour factor: {path_length / straight_line_length:.2f}x")
    print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Inequality constraints enabled us to:")
    print("  - Avoid circular obstacles along the path")
    print("  - Minimize path length subject to safety constraints")
    print("  - Identify which waypoints are near obstacles (active constraints)")
    print()
    print("This is a common pattern in robotics for collision-free motion planning!")
    print()

    print("Starting Viser visualization server...")
    server = viser.ViserServer()

    def to_3d(pos: jax.Array) -> tuple[float, float, float]:
        return (float(pos[0]), float(pos[1]), 0.0)

    # Initial path (blue).
    initial_positions = onp.array(
        [onp.array(to_3d(initial_vals[wp])) for wp in waypoints]
    )
    server.scene.add_line_segments(
        "/initial/path",
        points=onp.stack([initial_positions[:-1], initial_positions[1:]], axis=1),
        colors=(100, 100, 200),
        line_width=2.0,
    )
    for i, wp in enumerate(waypoints):
        pos = initial_vals[wp]
        server.scene.add_icosphere(
            f"/initial/waypoint_{i}",
            radius=0.02,
            position=to_3d(pos),
            color=(100, 100, 200),
        )

    # Optimized path (green).
    solution_positions = onp.array([onp.array(to_3d(solution[wp])) for wp in waypoints])
    server.scene.add_line_segments(
        "/solution/path",
        points=onp.stack([solution_positions[:-1], solution_positions[1:]], axis=1),
        colors=(0, 200, 0),
        line_width=3.0,
    )
    for i, wp in enumerate(waypoints):
        pos = solution[wp]
        server.scene.add_icosphere(
            f"/solution/waypoint_{i}",
            radius=0.02,
            position=to_3d(pos),
            color=(0, 200, 0),
        )

    # Obstacles (red).
    for i, (center, radius) in enumerate(obstacles):
        num_segments = 32
        angles = onp.linspace(0, 2 * onp.pi, num_segments + 1)
        circle_points = onp.array(
            [
                [
                    float(center[0]) + radius * onp.cos(angles[j]),
                    float(center[1]) + radius * onp.sin(angles[j]),
                    0.0,
                ]
                for j in range(num_segments + 1)
            ]
        )
        server.scene.add_line_segments(
            f"/obstacle_{i}/boundary",
            points=onp.stack([circle_points[:-1], circle_points[1:]], axis=1),
            colors=(200, 50, 50),
            line_width=3.0,
        )
        server.scene.add_icosphere(
            f"/obstacle_{i}/center",
            radius=0.02,
            position=(float(center[0]), float(center[1]), 0.0),
            color=(200, 50, 50),
        )

    # Start and goal markers.
    server.scene.add_icosphere(
        "/start",
        radius=0.05,
        position=to_3d(start),
        color=(0, 255, 0),
    )

    server.scene.add_icosphere(
        "/goal",
        radius=0.05,
        position=to_3d(goal),
        color=(255, 255, 0),
    )

    server.scene.add_grid("/grid", width=5.0, height=2.0, cell_size=0.5)

    print("Visualization server running at http://localhost:8080")
    server.sleep_forever()


if __name__ == "__main__":
    main()
