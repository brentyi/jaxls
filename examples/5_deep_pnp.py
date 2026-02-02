"""Deep PnP: End-to-end differentiable vision + geometry optimization.

This example demonstrates learning neural network weights to handle outliers in
Perspective-n-Point (PnP) pose estimation using bilevel optimization via
`solve_differentiable()`.

Problem setup:
- Given 3D points and 2D observations (some with outliers)
- A neural network predicts per-correspondence weights in [0,1]
- jaxls LM solver estimates camera pose (SE3) using weighted reprojection costs
- Geodesic loss vs ground truth pose flows back through the solver to train the network

The bilevel optimization structure:
- Inner problem: Weighted PnP - estimate SE3 pose minimizing weighted reprojection error
- Outer problem: Minimize reprojection error (or geodesic pose error) of the estimated pose

This is a fundamental pattern in learning-based robust estimation, where we want
to learn which measurements to trust without ground truth outlier labels.
"""

import logging

# Suppress jaxls and loguru logging for cleaner output
logging.getLogger("jaxls").setLevel(logging.ERROR)
# Disable loguru (used by jaxls internally) as well
try:
    from loguru import logger
    logger.disable("jaxls")
except ImportError:
    pass

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp
import optax
import viser

import jaxls


# =============================================================================
# Synthetic Data Generation
# =============================================================================


def generate_pnp_problem(
    key: jax.Array,
    num_inliers: int,
    num_outliers: int,
    noise_std: float = 2.0,
    focal: float = 500.0,
) -> tuple[jax.Array, jax.Array, jaxlie.SE3, jax.Array]:
    """Generate a PnP problem with inliers and outliers."""
    keys = jax.random.split(key, 6)
    num_total = num_inliers + num_outliers

    # Generate random 3D points in front of camera (z in [2, 10])
    points_3d = jax.random.uniform(keys[0], (num_total, 3), minval=-3.0, maxval=3.0)
    points_3d = points_3d.at[:, 2].set(
        jax.random.uniform(keys[1], (num_total,), minval=2.0, maxval=10.0)
    )

    # Shift points to positive X region for better scene composition
    points_3d = points_3d.at[:, 0].add(3.0)

    # Generate random camera pose (larger rotation and translation for visibility)
    rotation_vec = jax.random.uniform(keys[2], (3,), minval=-0.5, maxval=0.5)
    rotation = jaxlie.SO3.exp(rotation_vec)
    translation = jax.random.uniform(keys[3], (3,), minval=-2.0, maxval=2.0)
    gt_pose = jaxlie.SE3.from_rotation_and_translation(rotation, translation)

    # Project 3D points to 2D
    points_cam = jax.vmap(lambda p: gt_pose @ p)(points_3d)
    points_2d_true = focal * points_cam[:, :2] / points_cam[:, 2:3]

    # Add noise to observations
    noise = jax.random.normal(keys[4], (num_total, 2)) * noise_std
    points_2d = points_2d_true + noise

    # Replace outlier observations with random 2D points
    inlier_mask = jnp.arange(num_total) < num_inliers
    outlier_2d = jax.random.uniform(keys[5], (num_total, 2), minval=-200.0, maxval=200.0)
    points_2d = jnp.where(inlier_mask[:, None], points_2d, outlier_2d)

    return points_3d, points_2d, gt_pose, inlier_mask


# =============================================================================
# Visualization Helpers
# =============================================================================


def exaggerate_pose_error(
    estimated: jaxlie.SE3, gt: jaxlie.SE3, scale: float = 3.0
) -> jaxlie.SE3:
    """Exaggerate the difference between estimated and GT pose for visualization.

    This makes pose differences more visually apparent by scaling the error
    in tangent space before applying it.
    """
    # Compute relative transform (error)
    error = estimated @ gt.inverse()
    # Scale the error in tangent space
    error_scaled = jaxlie.SE3.exp(error.log() * scale)
    # Apply scaled error to GT
    return error_scaled @ gt


# =============================================================================
# Pure JAX MLP
# =============================================================================


def init_mlp(key: jax.Array, input_dim: int = 7, hidden_dim: int = 32) -> dict:
    """Initialize MLP parameters."""
    keys = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(keys[0], (input_dim, hidden_dim)) * 0.1,
        "b1": jnp.zeros(hidden_dim),
        "w2": jax.random.normal(keys[1], (hidden_dim, hidden_dim)) * 0.1,
        "b2": jnp.zeros(hidden_dim),
        "w3": jax.random.normal(keys[2], (hidden_dim, 1)) * 0.1,
        "b3": jnp.zeros(1),
    }


def apply_mlp(params: dict, features: jax.Array) -> jax.Array:
    """Forward pass: features (N, D) -> weights (N,) in [0, 1]."""
    x = jax.nn.relu(features @ params["w1"] + params["b1"])
    x = jax.nn.relu(x @ params["w2"] + params["b2"])
    x = x @ params["w3"] + params["b3"]
    return jax.nn.sigmoid(x.squeeze(-1))


def build_features(points_3d: jax.Array, points_2d: jax.Array, focal: float) -> jax.Array:
    """Build input features for the MLP."""
    points_3d_norm = points_3d / 10.0
    points_2d_norm = points_2d / 500.0
    reproj_scale = (focal / jnp.maximum(points_3d[:, 2], 0.1))[:, None] / 500.0
    obs_magnitude = jnp.linalg.norm(points_2d, axis=1, keepdims=True) / 500.0
    return jnp.concatenate([points_3d_norm, points_2d_norm, reproj_scale, obs_magnitude], axis=1)


# =============================================================================
# Weighted Reprojection Cost
# =============================================================================


@jaxls.Cost.factory
def weighted_reprojection_cost(
    vals: jaxls.VarValues,
    pose_var: jaxls.SE3Var,
    point_3d: jax.Array,
    observed_2d: jax.Array,
    weight: jax.Array,
    focal: float,
) -> jax.Array:
    """Weighted reprojection error cost."""
    pose = vals[pose_var]
    point_cam = pose @ point_3d
    projected = focal * point_cam[:2] / point_cam[2]
    return weight * (projected - observed_2d)


# =============================================================================
# Reprojection Error Utilities
# =============================================================================


def compute_reprojection_errors(
    pose: jaxlie.SE3,
    points_3d: jax.Array,
    points_2d: jax.Array,
    focal: float,
) -> jax.Array:
    """Compute per-point reprojection errors in pixels."""
    points_cam = jax.vmap(lambda p: pose @ p)(points_3d)
    projected = focal * points_cam[:, :2] / points_cam[:, 2:3]
    return jnp.linalg.norm(projected - points_2d, axis=1)


def print_error_histogram(
    label: str, errors: jax.Array, bins: int = 10, width: int = 24
) -> None:
    """Print a simple ASCII histogram of reprojection errors."""
    errors_np = onp.asarray(errors)
    hist, edges = onp.histogram(errors_np, bins=bins)
    max_count = int(hist.max()) if hist.size > 0 else 1
    print(f"  {label}:")
    for i, count in enumerate(hist):
        bar_len = int(width * (count / max_count)) if max_count > 0 else 0
        bar = "#" * bar_len
        left = edges[i]
        right = edges[i + 1]
        print(f"    {left:6.1f}-{right:6.1f}px | {bar} ({count})")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Deep PnP: Learning Robust Pose Estimation with Differentiable Solving")
    print("=" * 70)
    print()

    # Parameters
    focal = 500.0
    num_inliers = 30
    num_outliers = 10
    num_correspondences = num_inliers + num_outliers
    num_iterations = 50
    learning_rate = 0.02
    outer_loss = "reprojection"  # "reprojection" or "pose"

    print("Problem setup:")
    print(f"  Correspondences: {num_inliers} inliers + {num_outliers} outliers")
    print(f"  Focal length: {focal}")
    print(f"  Training iterations: {num_iterations}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Outer loss: {outer_loss}")
    print()

    # Initialize
    key = jax.random.key(42)
    key, init_key = jax.random.split(key)
    params = init_mlp(init_key)

    print("Network: 7 -> 32 -> 32 -> 1 (sigmoid)")
    print()

    # =========================================================================
    # Generate FIXED training problem (like example 4 pattern)
    # =========================================================================

    key, data_key = jax.random.split(key)
    train_3d, train_2d, train_gt, train_mask = generate_pnp_problem(
        data_key, num_inliers, num_outliers, focal=focal
    )

    print("Training problem generated (fixed for all iterations)")
    print()

    # =========================================================================
    # Build and Solve Function
    # =========================================================================

    # Define variables outside the function (like example 4)
    pose_var = jaxls.SE3Var(0)

    def build_and_solve(weights: jax.Array) -> jaxlie.SE3:
        """Build weighted PnP problem and solve differentiably.

        Following example 4 pattern: fixed problem structure, only weights vary.
        """
        cost = weighted_reprojection_cost(
            jaxls.SE3Var(id=jnp.zeros(num_correspondences, dtype=jnp.int32)),
            train_3d,
            train_2d,
            weights,
            focal,
        )
        problem = jaxls.LeastSquaresProblem([cost], [pose_var]).analyze()
        return problem.solve_differentiable(
            linear_solver="dense_cholesky",
            verbose=False,
        )[pose_var]

    # =========================================================================
    # Loss Function
    # =========================================================================

    def loss_fn(params: dict) -> jax.Array:
        """Compute pose loss for the training problem."""
        features = build_features(train_3d, train_2d, focal)
        weights = apply_mlp(params, features)
        weights = jnp.clip(weights, 0.01, 1.0)
        estimated = build_and_solve(weights)
        if outer_loss == "pose":
            return jnp.sum((estimated @ train_gt.inverse()).log() ** 2)
        if outer_loss == "reprojection":
            errors = compute_reprojection_errors(estimated, train_3d, train_2d, focal)
            return jnp.mean(errors**2)
        raise ValueError(f"Unknown outer loss: {outer_loss}")

    # JIT compile
    loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # =========================================================================
    # Training Loop
    # =========================================================================

    print("=" * 70)
    print("Training")
    print("=" * 70)
    print()

    loss_label = "Pose Loss" if outer_loss == "pose" else "Reproj MSE"
    print(f"{'Iter':>5} | {loss_label:>12}")
    print("-" * 25)

    for i in range(num_iterations):
        loss, grads = loss_and_grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if i % 10 == 0 or i == num_iterations - 1:
            print(f"{i:>5} | {float(loss):>12.4f}")

    print()

    # =========================================================================
    # Evaluation
    # =========================================================================

    print("=" * 70)
    print("Evaluation")
    print("=" * 70)
    print()

    # Evaluate on the training problem
    features = build_features(train_3d, train_2d, focal)
    learned_weights = apply_mlp(params, features)

    def pose_geodesic_loss(pose: jaxlie.SE3, gt: jaxlie.SE3) -> jax.Array:
        return jnp.sum((pose @ gt.inverse()).log() ** 2)

    def reprojection_rmse(
        pose: jaxlie.SE3, points_3d: jax.Array, points_2d: jax.Array, focal_len: float
    ) -> jax.Array:
        errors = compute_reprojection_errors(pose, points_3d, points_2d, focal_len)
        return jnp.sqrt(jnp.mean(errors**2))

    # Learned weights
    learned_pose = build_and_solve(jnp.clip(learned_weights, 0.01, 1.0))
    learned_pose_loss = float(pose_geodesic_loss(learned_pose, train_gt))
    learned_reproj_rmse = float(
        reprojection_rmse(learned_pose, train_3d, train_2d, focal)
    )

    # Uniform weights
    uniform_pose = build_and_solve(jnp.ones(num_correspondences))
    uniform_pose_loss = float(pose_geodesic_loss(uniform_pose, train_gt))
    uniform_reproj_rmse = float(
        reprojection_rmse(uniform_pose, train_3d, train_2d, focal)
    )

    # Oracle weights
    oracle_pose = build_and_solve(train_mask.astype(jnp.float32))
    oracle_pose_loss = float(pose_geodesic_loss(oracle_pose, train_gt))
    oracle_reproj_rmse = float(
        reprojection_rmse(oracle_pose, train_3d, train_2d, focal)
    )

    print("Pose loss (geodesic SE3 distance, lower is better):")
    print(f"  {'Method':<20} | {'Loss':>10}")
    print(f"  {'-' * 20}-+-{'-' * 10}")
    print(f"  {'Learned weights':<20} | {learned_pose_loss:>10.4f}")
    print(f"  {'Uniform weights':<20} | {uniform_pose_loss:>10.4f}")
    print(f"  {'Oracle weights':<20} | {oracle_pose_loss:>10.4f}")
    print()

    print("Reprojection error (RMSE in px, lower is better):")
    print(f"  {'Method':<20} | {'RMSE':>10}")
    print(f"  {'-' * 20}-+-{'-' * 10}")
    print(f"  {'Learned weights':<20} | {learned_reproj_rmse:>10.3f}")
    print(f"  {'Uniform weights':<20} | {uniform_reproj_rmse:>10.3f}")
    print(f"  {'Oracle weights':<20} | {oracle_reproj_rmse:>10.3f}")
    print()

    # Evaluate on test problems
    print("Generalization to new problems:")
    key, test_key = jax.random.split(key)
    test_keys = jax.random.split(test_key, 10)

    learned_losses = []
    uniform_losses = []

    for tk in test_keys:
        test_3d, test_2d, test_gt, test_mask = generate_pnp_problem(
            tk, num_inliers, num_outliers, focal=focal
        )

        # Apply learned network to new problem
        test_features = build_features(test_3d, test_2d, focal)
        test_weights = apply_mlp(params, test_features)
        test_weights = jnp.clip(test_weights, 0.01, 1.0)

        # Solve with learned weights
        test_cost = weighted_reprojection_cost(
            jaxls.SE3Var(id=jnp.zeros(num_correspondences, dtype=jnp.int32)),
            test_3d, test_2d, test_weights, focal,
        )
        test_problem = jaxls.LeastSquaresProblem([test_cost], [pose_var]).analyze()
        test_learned_pose = test_problem.solve(verbose=False)[pose_var]
        learned_losses.append(
            float(reprojection_rmse(test_learned_pose, test_3d, test_2d, focal))
        )

        # Solve with uniform weights
        uniform_cost = weighted_reprojection_cost(
            jaxls.SE3Var(id=jnp.zeros(num_correspondences, dtype=jnp.int32)),
            test_3d, test_2d, jnp.ones(num_correspondences), focal,
        )
        uniform_problem = jaxls.LeastSquaresProblem([uniform_cost], [pose_var]).analyze()
        test_uniform_pose = uniform_problem.solve(verbose=False)[pose_var]
        uniform_losses.append(
            float(reprojection_rmse(test_uniform_pose, test_3d, test_2d, focal))
        )

    print("Generalization reprojection error (RMSE in px):")
    print(f"  {'Method':<20} | {'Mean':>10} | {'Std':>10}")
    print(f"  {'-' * 20}-+-{'-' * 10}-+-{'-' * 10}")
    print(
        f"  {'Learned weights':<20} | {onp.mean(learned_losses):>10.3f} | {onp.std(learned_losses):>10.3f}"
    )
    print(
        f"  {'Uniform weights':<20} | {onp.mean(uniform_losses):>10.3f} | {onp.std(uniform_losses):>10.3f}"
    )
    print()

    # Weight analysis
    print("Weight prediction analysis (training problem):")
    inlier_w = learned_weights[train_mask]
    outlier_w = learned_weights[~train_mask]
    print(f"  Inlier weights:  mean={float(inlier_w.mean()):.3f}, std={float(inlier_w.std()):.3f}")
    print(f"  Outlier weights: mean={float(outlier_w.mean()):.3f}, std={float(outlier_w.std()):.3f}")
    print()
    print("Reprojection error histograms (training problem):")
    print_error_histogram(
        "Learned weights",
        compute_reprojection_errors(learned_pose, train_3d, train_2d, focal),
    )
    print_error_histogram(
        "Uniform weights",
        compute_reprojection_errors(uniform_pose, train_3d, train_2d, focal),
    )
    print()

    # =========================================================================
    # Summary
    # =========================================================================

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("This example demonstrated bilevel optimization for robust pose estimation:")
    print()
    print("  1. Inner problem: Weighted PnP pose estimation")
    print("     - Minimize weighted reprojection error")
    print("     - SE3 pose solved via jaxls Levenberg-Marquardt")
    print()
    print("  2. Outer problem: Learn correspondence weights from supervision")
    if outer_loss == "pose":
        print("     - Loss: Geodesic SE3 distance to ground truth pose")
    else:
        print("     - Loss: Reprojection error of the estimated pose")
    print("     - Gradients flow through solver via adjoint method")
    print()
    if outer_loss == "pose":
        print("Key insight: The network learns to downweight outliers without explicit")
        print("outlier labels - only pose supervision is needed.")
    else:
        print("Key insight: Reprojection supervision can replace pose supervision,")
        print("but it may be more sensitive to outliers.")
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

    # Draw 3D points colored by inlier/outlier status with weight indication
    for i in range(num_correspondences):
        point = train_3d[i]
        weight = float(learned_weights[i])
        is_inlier = bool(train_mask[i])

        # Color by ground truth status: blue=inlier, red=outlier
        # Brightness indicates learned weight (brighter = higher weight)
        if is_inlier:
            # Inliers: blue, brightness by weight
            brightness = 0.3 + 0.7 * weight
            r, g, b = int(50 * brightness), int(100 * brightness), int(255 * brightness)
        else:
            # Outliers: red/orange
            brightness = 0.3 + 0.7 * weight
            r, g, b = int(255 * brightness), int(50 * brightness), int(50 * brightness)

        radius = 0.15  # Larger for visibility

        server.scene.add_icosphere(
            f"/points/pt_{i}",
            radius=radius,
            position=(float(point[0]), float(point[1]), float(point[2])),
            color=(r, g, b),
        )

    # Draw camera frustums with image planes showing reprojection errors
    def draw_frustum(
        name: str,
        pose: jaxlie.SE3,
        color: tuple[int, int, int],
        scale: float = 0.5,
    ):
        """Draw camera frustum wireframe."""
        T = pose.inverse()  # camera-to-world transform
        corners = onp.array([
            [0, 0, 0],
            [-scale, -scale, scale * 2],
            [scale, -scale, scale * 2],
            [scale, scale, scale * 2],
            [-scale, scale, scale * 2],
        ])
        corners_w = onp.array([onp.array(T @ c) for c in corners])
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
        for j, (a, b) in enumerate(edges):
            server.scene.add_line_segments(
                f"/{name}/e{j}",
                points=onp.array([[corners_w[a], corners_w[b]]]),
                colors=color,
                line_width=3.0,
            )

    def draw_image_plane(
        name: str,
        pose: jaxlie.SE3,
        points_3d: jax.Array,
        points_2d_obs: jax.Array,
        weights: jax.Array,
        focal_len: float,
        scale: float = 0.5,
        img_scale: float = 0.003,  # Larger scale for visibility
    ):
        """Draw image plane with 2D observations and reprojections inside frustum.

        Shows observed 2D points and their reprojection errors as connecting lines.
        """
        T = pose.inverse()  # camera-to-world
        plane_z = scale * 2  # Image plane at frustum base

        # Draw image frame boundary
        frame_size = scale * 0.8
        frame_corners_cam = onp.array([
            [-frame_size, -frame_size, plane_z],
            [frame_size, -frame_size, plane_z],
            [frame_size, frame_size, plane_z],
            [-frame_size, frame_size, plane_z],
        ])
        frame_corners_w = onp.array([onp.array(T @ c) for c in frame_corners_cam])
        frame_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for j, (a, b) in enumerate(frame_edges):
            server.scene.add_line_segments(
                f"/{name}/img/frame_{j}",
                points=onp.array([[frame_corners_w[a], frame_corners_w[b]]]),
                colors=(100, 100, 100),
                line_width=2.0,
            )

        for i in range(len(points_3d)):
            # Project 3D point using this pose
            p_cam = pose @ points_3d[i]
            if float(p_cam[2]) <= 0:  # Behind camera
                continue

            proj_2d = focal_len * p_cam[:2] / p_cam[2]

            # Convert 2D coords to 3D position on image plane
            obs_3d_cam = jnp.array([
                float(points_2d_obs[i, 0]) * img_scale,
                float(points_2d_obs[i, 1]) * img_scale,
                plane_z,
            ])
            proj_3d_cam = jnp.array([
                float(proj_2d[0]) * img_scale,
                float(proj_2d[1]) * img_scale,
                plane_z,
            ])

            # Transform to world frame
            obs_3d_world = onp.array(T @ obs_3d_cam)
            proj_3d_world = onp.array(T @ proj_3d_cam)

            # Draw observation point (colored by weight: green=high, red=low)
            weight = float(weights[i])
            obs_color = (int(255 * (1 - weight)), int(255 * weight), 0)
            server.scene.add_icosphere(
                f"/{name}/img/obs_{i}",
                radius=0.04,  # Larger for visibility
                position=tuple(obs_3d_world),
                color=obs_color,
            )

            # Draw projected point (cyan for contrast)
            server.scene.add_icosphere(
                f"/{name}/img/proj_{i}",
                radius=0.03,
                position=tuple(proj_3d_world),
                color=(0, 255, 255),
            )

            # Draw reprojection error line (yellow for visibility)
            server.scene.add_line_segments(
                f"/{name}/img/err_{i}",
                points=onp.array([[obs_3d_world, proj_3d_world]]),
                colors=(255, 255, 0),
                line_width=2.0,
            )

    # Exaggerate pose differences for visibility (set > 1.0 to amplify).
    exaggeration_scale = 1.0
    learned_pose_vis = exaggerate_pose_error(learned_pose, train_gt, exaggeration_scale)
    uniform_pose_vis = exaggerate_pose_error(uniform_pose, train_gt, exaggeration_scale)

    # Draw frustums (GT and exaggerated estimated poses)
    frustum_scale = 0.5
    draw_frustum("gt", train_gt, (50, 100, 200), scale=frustum_scale)
    draw_frustum("learned", learned_pose_vis, (50, 200, 50), scale=frustum_scale)
    draw_frustum("uniform", uniform_pose_vis, (200, 50, 50), scale=frustum_scale)

    # Draw image planes showing reprojection errors for all three cameras
    # Use actual poses (not exaggerated) for correct reprojection computation
    draw_image_plane(
        "gt", train_gt, train_3d, train_2d, learned_weights, focal, scale=frustum_scale
    )
    draw_image_plane(
        "learned",
        learned_pose,
        train_3d,
        train_2d,
        learned_weights,
        focal,
        scale=frustum_scale,
    )
    draw_image_plane(
        "uniform",
        uniform_pose,
        train_3d,
        train_2d,
        jnp.ones(num_correspondences),
        focal,
        scale=frustum_scale,
    )

    # Add labels
    label_offset = onp.array([0, 0, 0.4])
    server.scene.add_label(
        "/labels/gt",
        "GT (Blue)",
        position=tuple(onp.array(train_gt.inverse().translation()) + label_offset),
    )
    server.scene.add_label(
        "/labels/learned",
        "Learned (Green)",
        position=tuple(
            onp.array(learned_pose_vis.inverse().translation()) + label_offset
        ),
    )
    server.scene.add_label(
        "/labels/uniform",
        "Uniform (Red)",
        position=tuple(
            onp.array(uniform_pose_vis.inverse().translation()) + label_offset
        ),
    )

    # Add coordinate frame at origin for reference
    server.scene.add_frame("/origin", axes_length=0.5, axes_radius=0.02)

    # Add grid
    server.scene.add_grid("/grid", width=15.0, height=15.0, cell_size=1.0)

    print()
    print("Legend:")
    print("  3D Points: Blue=inlier, Red=outlier (brightness=learned weight)")
    print("  Cameras: Blue=GT, Green=Learned, Red=Uniform")
    print("  Image planes: Colored dots=observations, Cyan dots=projections")
    print("  Yellow lines = reprojection errors")
    print(f"  Pose error exaggeration: {exaggeration_scale}x for visibility")
    print()
    print(f"Visualization: http://{server.get_host()}:{server.get_port()}")
    server.sleep_forever()


if __name__ == "__main__":
    main()
