"""Imitation learning / inverse optimal control using differentiable solving.

This example demonstrates learning cost function weights from expert demonstrations
using bilevel optimization via `solve_differentiable()`.

Problem setup:
- A robot navigates through 5 SE2 poses in a 2D plane
- Two competing objectives:
  1. Waypoint attraction: Pull poses toward target waypoints
  2. Smoothness: Penalize large relative transforms between consecutive poses
- An additional "attractor" pulls middle poses off the straight line

The bilevel optimization structure:
- Inner problem: Given cost weights, solve the trajectory optimization
- Outer problem: Minimize difference between solved trajectory and expert trajectory

Why this requires bilevel optimization:
- The inner solver only sees the weighted cost function
- The outer loss compares to the expert trajectory (not visible to inner solver)
- Cannot fold expert matching into the inner problem without changing its semantics

This is a fundamental pattern in inverse reinforcement learning and system identification.
"""

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp
import optax
import viser

import jaxls


# =============================================================================
# Problem Setup
# =============================================================================


def main():
    print("=" * 70)
    print("Imitation Learning with Differentiable Optimization")
    print("=" * 70)
    print()

    # 5 SE2 poses in a line
    num_poses = 5
    pose_vars = [jaxls.SE2Var(i) for i in range(num_poses)]

    # Target waypoints along x-axis
    waypoints = [jaxlie.SE2.from_xy_theta(float(i), 0.0, 0.0) for i in range(num_poses)]

    # Attractor pulls middle poses off the line (makes problem interesting)
    attractor = jaxlie.SE2.from_xy_theta(2.0, 1.5, 0.0)
    attractor_indices = [1, 2, 3]  # Middle poses affected by attractor

    print("Problem setup:")
    print(f"  Poses: {num_poses} SE2 variables")
    print("  Waypoints: (0,0), (1,0), (2,0), (3,0), (4,0)")
    print("  Attractor: (2.0, 1.5) - pulls middle poses")
    print()

    # =========================================================================
    # Cost Functions with Weight Parameters
    # =========================================================================

    # We use a single cost function that takes all weights as batched parameters.
    # This allows us to avoid re-analyzing the problem structure for each weight
    # configuration, which is essential for efficient gradient computation.

    @jaxls.Cost.factory
    def waypoint_cost(
        vals: jaxls.VarValues,
        var: jaxls.SE2Var,
        target: jaxlie.SE2,
        weight: jax.Array,
    ) -> jax.Array:
        """Pull pose toward a target waypoint (weighted)."""
        return weight * (vals[var] @ target.inverse()).log()

    @jaxls.Cost.factory
    def attractor_cost(
        vals: jaxls.VarValues,
        var: jaxls.SE2Var,
        target: jaxlie.SE2,
        weight: jax.Array,
    ) -> jax.Array:
        """Pull pose toward an attractor point (weighted)."""
        return weight * (vals[var] @ target.inverse()).log()

    @jaxls.Cost.factory
    def smoothness_cost(
        vals: jaxls.VarValues,
        var0: jaxls.SE2Var,
        var1: jaxls.SE2Var,
        weight: jax.Array,
    ) -> jax.Array:
        """Penalize non-identity relative transform between consecutive poses."""
        return weight * (vals[var0].inverse() @ vals[var1]).log()

    # =========================================================================
    # Build and Solve with Weights
    # =========================================================================

    def build_and_solve(weights: jax.Array) -> jaxls.VarValues:
        """Build problem and solve with given weights.

        Args:
            weights: [waypoint_weight, attractor_weight, smoothness_weight]

        Returns:
            Solution VarValues
        """
        waypoint_weight = weights[0]
        attractor_weight = weights[1]
        smoothness_weight = weights[2]

        costs: list[jaxls.Cost] = []

        # Waypoint costs
        for var, target in zip(pose_vars, waypoints):
            costs.append(waypoint_cost(var, target, waypoint_weight))

        # Attractor costs
        for i in attractor_indices:
            costs.append(attractor_cost(pose_vars[i], attractor, attractor_weight))

        # Smoothness costs
        for i in range(num_poses - 1):
            costs.append(
                smoothness_cost(pose_vars[i], pose_vars[i + 1], smoothness_weight)
            )

        problem = jaxls.LeastSquaresProblem(costs, pose_vars).analyze()
        return problem.solve_differentiable(
            linear_solver="dense_cholesky",
            verbose=False,
        )

    # =========================================================================
    # Generate Expert Trajectory
    # =========================================================================

    # True weights (unknown to learner)
    # [waypoint_weight, attractor_weight, smoothness_weight]
    true_weights = jnp.array([1.0, 0.3, 2.0])

    print("True (hidden) weights:")
    print(f"  waypoint_weight:  {true_weights[0]:.2f} (fixed at 1.0)")
    print(f"  attractor_weight: {true_weights[1]:.2f}")
    print(f"  smoothness_weight: {true_weights[2]:.2f}")
    print()

    # Generate expert trajectory
    print("Generating expert trajectory...")
    expert_solution = build_and_solve(true_weights)

    print("Expert trajectory:")
    for i, var in enumerate(pose_vars):
        pose = expert_solution[var]
        xy = pose.translation()
        theta = jnp.arctan2(
            pose.rotation().as_matrix()[1, 0], pose.rotation().as_matrix()[0, 0]
        )
        print(f"  Pose {i}: ({xy[0]:.3f}, {xy[1]:.3f}, {theta:.3f})")
    print()

    # =========================================================================
    # Imitation Learning Loss
    # =========================================================================

    def imitation_loss(learnable_weights: jax.Array) -> jax.Array:
        """Compute loss between learned trajectory and expert trajectory.

        Args:
            learnable_weights: [attractor_weight, smoothness_weight]
                              (waypoint_weight is fixed at 1.0 for scale invariance)

        Returns:
            Scalar loss (sum of squared Lie group distances)
        """
        # Construct full weights with fixed waypoint_weight = 1.0
        full_weights = jnp.array([1.0, learnable_weights[0], learnable_weights[1]])

        # Solve with current weights
        solution = build_and_solve(full_weights)

        # Compute loss: sum of squared distances in Lie algebra
        total_loss = jnp.array(0.0)
        for var in pose_vars:
            # Lie group distance: ||log(solution @ expert^{-1})||^2
            diff = (solution[var] @ expert_solution[var].inverse()).log()
            total_loss = total_loss + jnp.sum(diff**2)

        return total_loss

    # =========================================================================
    # Training Loop
    # =========================================================================

    print("=" * 70)
    print("Training: Learning weights from expert demonstration")
    print("=" * 70)
    print()

    # Initial guess for learnable weights (different from true values)
    # [attractor_weight, smoothness_weight]
    init_learnable_weights = jnp.array([0.1, 0.5])

    # Optimizer
    learning_rate = 0.05
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(init_learnable_weights)

    # Training
    weights = init_learnable_weights
    num_iterations = 50

    print(f"Initial weights: attractor={weights[0]:.3f}, smoothness={weights[1]:.3f}")
    print(
        f"Target weights:  attractor={true_weights[1]:.3f}, smoothness={true_weights[2]:.3f}"
    )
    print(f"Optimizer: Adam, lr={learning_rate}")
    print(f"Iterations: {num_iterations}")
    print()
    print(f"{'Iter':>5} | {'Loss':>10} | {'attractor_w':>12} | {'smoothness_w':>12}")
    print("-" * 55)

    # Use JIT for faster training
    loss_and_grad_fn = jax.jit(jax.value_and_grad(imitation_loss))

    for i in range(num_iterations):
        loss, grads = loss_and_grad_fn(weights)

        updates, opt_state = optimizer.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)

        # Clamp weights to be positive
        weights = jnp.maximum(weights, 0.01)

        # Print progress
        if i % 5 == 0 or i == num_iterations - 1:
            print(f"{i:>5} | {loss:>10.6f} | {weights[0]:>12.4f} | {weights[1]:>12.4f}")

    print()

    # =========================================================================
    # Results
    # =========================================================================

    print("=" * 70)
    print("Results")
    print("=" * 70)
    print()

    final_attractor_weight = float(weights[0])
    final_smoothness_weight = float(weights[1])

    print("Weight comparison:")
    print(f"  {'Weight':<20} | {'True':>10} | {'Learned':>10} | {'Error':>10}")
    print(f"  {'-' * 20}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")
    print(
        f"  {'waypoint_weight':<20} | {float(true_weights[0]):>10.4f} | "
        f"{1.0:>10.4f} | {'(fixed)':>10}"
    )
    print(
        f"  {'attractor_weight':<20} | {float(true_weights[1]):>10.4f} | "
        f"{final_attractor_weight:>10.4f} | "
        f"{abs(float(true_weights[1]) - final_attractor_weight):>10.4f}"
    )
    print(
        f"  {'smoothness_weight':<20} | {float(true_weights[2]):>10.4f} | "
        f"{final_smoothness_weight:>10.4f} | "
        f"{abs(float(true_weights[2]) - final_smoothness_weight):>10.4f}"
    )
    print()

    # Compare trajectories
    final_weights = jnp.array([1.0, final_attractor_weight, final_smoothness_weight])
    learned_solution = build_and_solve(final_weights)

    print("Trajectory comparison:")
    print(
        f"  {'Pose':<6} | {'Expert (x,y)':>15} | {'Learned (x,y)':>15} | {'Error':>10}"
    )
    print(f"  {'-' * 6}-+-{'-' * 15}-+-{'-' * 15}-+-{'-' * 10}")

    total_traj_error = 0.0
    for i, var in enumerate(pose_vars):
        expert_xy = expert_solution[var].translation()
        learned_xy = learned_solution[var].translation()
        error = jnp.linalg.norm(expert_xy - learned_xy)
        total_traj_error += float(error)
        print(
            f"  {i:<6} | ({expert_xy[0]:>6.3f}, {expert_xy[1]:>5.3f}) | "
            f"({learned_xy[0]:>6.3f}, {learned_xy[1]:>5.3f}) | {error:>10.4f}"
        )

    print()
    print(f"Total trajectory error: {total_traj_error:.6f}")
    print()

    # =========================================================================
    # Summary
    # =========================================================================

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("This example demonstrated bilevel optimization for imitation learning:")
    print()
    print("  1. Inner problem: Trajectory optimization with weighted costs")
    print("     - Waypoint attraction, smoothness, and attractor terms")
    print("     - Solved via solve_differentiable()")
    print()
    print("  2. Outer problem: Weight learning from expert demonstration")
    print("     - Loss: Lie group distance between trajectories")
    print("     - Gradients flow through the inner solver via adjoint method")
    print()
    print("Key insight: The expert trajectory is only visible to the outer loss,")
    print("not to the inner optimization. This bilevel structure cannot be")
    print("reformulated as a single optimization problem.")
    print()

    # =========================================================================
    # Visualization
    # =========================================================================

    print("=" * 70)
    print("Visualization")
    print("=" * 70)
    print()
    print("Starting Viser visualization server...")

    server = viser.ViserServer()

    def pose_to_3d(pose: jaxlie.SE2) -> tuple[float, float, float]:
        """Convert SE2 pose to 3D position for visualization."""
        xy = pose.translation()
        return (float(xy[0]), float(xy[1]), 0.0)

    def draw_trajectory(
        name: str,
        solution: jaxls.VarValues,
        color: tuple[int, int, int],
        line_width: float = 3.0,
        sphere_radius: float = 0.03,
    ) -> None:
        """Draw a trajectory with line segments and pose markers."""
        # Get positions
        positions = onp.array(
            [onp.array(pose_to_3d(solution[var])) for var in pose_vars]
        )

        # Draw path as line segments
        server.scene.add_line_segments(
            f"/{name}/path",
            points=onp.stack([positions[:-1], positions[1:]], axis=1),
            colors=color,
            line_width=line_width,
        )

        # Draw pose markers
        for i, var in enumerate(pose_vars):
            pose = solution[var]
            pos = pose_to_3d(pose)

            # Sphere at position
            server.scene.add_icosphere(
                f"/{name}/pose_{i}",
                radius=sphere_radius,
                position=pos,
                color=color,
            )

            # Draw orientation arrow
            theta = float(
                jnp.arctan2(
                    pose.rotation().as_matrix()[1, 0], pose.rotation().as_matrix()[0, 0]
                )
            )
            arrow_length = 0.15
            arrow_end = (
                pos[0] + arrow_length * onp.cos(theta),
                pos[1] + arrow_length * onp.sin(theta),
                0.0,
            )
            server.scene.add_line_segments(
                f"/{name}/arrow_{i}",
                points=onp.array([[[pos[0], pos[1], pos[2]], list(arrow_end)]]),
                colors=color,
                line_width=2.0,
            )

    # Generate trajectory with initial (wrong) weights for comparison
    initial_weights_full = jnp.array(
        [1.0, init_learnable_weights[0], init_learnable_weights[1]]
    )
    initial_solution = build_and_solve(initial_weights_full)

    # Draw expert trajectory (blue)
    draw_trajectory("expert", expert_solution, color=(50, 100, 200), line_width=4.0)

    # Draw initial trajectory (red) - with wrong weights
    draw_trajectory("initial", initial_solution, color=(200, 50, 50), line_width=3.0)

    # Draw learned trajectory (green)
    draw_trajectory("learned", learned_solution, color=(50, 200, 50), line_width=3.0)

    # Draw target waypoints (gray spheres)
    for i, wp in enumerate(waypoints):
        xy = wp.translation()
        server.scene.add_icosphere(
            f"/waypoints/wp_{i}",
            radius=0.04,
            position=(float(xy[0]), float(xy[1]), 0.0),
            color=(150, 150, 150),
        )

    # Draw attractor (yellow sphere)
    attractor_xy = attractor.translation()
    server.scene.add_icosphere(
        "/attractor",
        radius=0.08,
        position=(float(attractor_xy[0]), float(attractor_xy[1]), 0.0),
        color=(255, 200, 50),
    )

    # Add grid for reference
    server.scene.add_grid("/grid", width=6.0, height=3.0, cell_size=0.5)

    # Add legend text
    print()
    print("Legend:")
    print("  Blue:   Expert trajectory (ground truth)")
    print("  Red:    Initial trajectory (before learning)")
    print("  Green:  Learned trajectory (after optimization)")
    print("  Gray:   Target waypoints")
    print("  Yellow: Attractor point")
    print()

    print(
        f"Visualization server running at http://{server.get_host()}:{server.get_port()}"
    )
    server.sleep_forever()


if __name__ == "__main__":
    main()
