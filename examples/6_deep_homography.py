"""Deep Homography: End-to-end differentiable image alignment via learned features.

This example demonstrates learning CNN feature representations that make homography
estimation easier using bilevel optimization via `solve_differentiable()`.

Problem setup:
- Given two images (source and target, where target is a warped version of source)
- A CNN extracts feature maps from both images
- jaxls LM solver estimates the 8-parameter homography aligning the feature maps
- 4-corner error loss flows back through the solver to train the CNN

The bilevel optimization structure:
- Inner problem: Estimate homography by minimizing feature alignment error
- Outer problem: Minimize 4-corner reprojection error of estimated homography

This demonstrates that learned feature representations can be optimized end-to-end
to produce features that are easier for the optimization to align correctly.
"""

import logging

# Suppress jaxls and loguru logging for cleaner output
logging.getLogger("jaxls").setLevel(logging.ERROR)
try:
    from loguru import logger

    logger.disable("jaxls")
except ImportError:
    pass

import jax
import jax.numpy as jnp
import numpy as onp
import optax
import viser

import jaxls


# =============================================================================
# Image Loading and Data Generation
# =============================================================================


def load_sample_images() -> list[jax.Array]:
    """Load sample images from skimage.data.

    Returns a list of grayscale images normalized to [0, 1].
    """
    from skimage import data
    from skimage.color import rgb2gray
    from skimage.transform import resize

    images = []
    target_size = (64, 64)

    # Load built-in images
    image_funcs = [data.camera, data.astronaut, data.coffee, data.chelsea, data.coins]

    for func in image_funcs:
        img = func()
        # Convert to grayscale if needed
        if img.ndim == 3:
            img = rgb2gray(img)
        # Resize to target size
        img = resize(img, target_size, anti_aliasing=True)
        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        images.append(jnp.array(img, dtype=jnp.float32))

    return images


def generate_random_homography(
    key: jax.Array,
    image_size: tuple[int, int] = (64, 64),
    rotation_range: float = 0.2,
    translation_range: float = 5.0,
    scale_range: tuple[float, float] = (0.9, 1.1),
    perspective_range: float = 1e-4,
) -> jax.Array:
    """Generate a random homography matrix.

    Args:
        key: JAX random key
        image_size: (height, width) of the image
        rotation_range: Max rotation in radians
        translation_range: Max translation in pixels
        scale_range: (min_scale, max_scale)
        perspective_range: Max perspective distortion

    Returns:
        3x3 homography matrix
    """
    keys = jax.random.split(key, 5)
    h, w = image_size

    # Center of the image
    cx, cy = w / 2, h / 2

    # Random rotation
    theta = jax.random.uniform(keys[0], (), minval=-rotation_range, maxval=rotation_range)
    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
    R = jnp.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])

    # Random translation
    tx = jax.random.uniform(keys[1], (), minval=-translation_range, maxval=translation_range)
    ty = jax.random.uniform(keys[2], (), minval=-translation_range, maxval=translation_range)
    T = jnp.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    # Random scale
    s = jax.random.uniform(keys[3], (), minval=scale_range[0], maxval=scale_range[1])
    S = jnp.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])

    # Random perspective (small)
    p1 = jax.random.uniform(keys[4], (), minval=-perspective_range, maxval=perspective_range)
    p2 = jax.random.uniform(keys[4], (), minval=-perspective_range, maxval=perspective_range)
    P = jnp.array([[1, 0, 0], [0, 1, 0], [p1, p2, 1]])

    # Center transform matrices
    T_center = jnp.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    T_uncenter = jnp.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])

    # Compose: translate to center, apply transforms, translate back
    H = T_uncenter @ P @ S @ R @ T_center @ T

    # Normalize so H[2,2] = 1
    H = H / H[2, 2]

    return H


# =============================================================================
# Differentiable Image Warping
# =============================================================================


def bilinear_sample(img: jax.Array, coords: jax.Array) -> jax.Array:
    """Sample image at subpixel coordinates using bilinear interpolation.

    Args:
        img: Image of shape (H, W)
        coords: Coordinates of shape (..., 2) where coords[..., 0] is x and coords[..., 1] is y

    Returns:
        Sampled values of shape (...)
    """
    h, w = img.shape
    x = coords[..., 0]
    y = coords[..., 1]

    # Get integer coordinates
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp to valid range
    x0_c = jnp.clip(x0, 0, w - 1)
    x1_c = jnp.clip(x1, 0, w - 1)
    y0_c = jnp.clip(y0, 0, h - 1)
    y1_c = jnp.clip(y1, 0, h - 1)

    # Get pixel values at corners
    Ia = img[y0_c, x0_c]
    Ib = img[y1_c, x0_c]
    Ic = img[y0_c, x1_c]
    Id = img[y1_c, x1_c]

    # Bilinear weights
    wa = (x1.astype(jnp.float32) - x) * (y1.astype(jnp.float32) - y)
    wb = (x1.astype(jnp.float32) - x) * (y - y0.astype(jnp.float32))
    wc = (x - x0.astype(jnp.float32)) * (y1.astype(jnp.float32) - y)
    wd = (x - x0.astype(jnp.float32)) * (y - y0.astype(jnp.float32))

    # Interpolate
    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def create_grid(height: int, width: int) -> jax.Array:
    """Create a regular grid of pixel coordinates.

    Returns:
        Coordinates of shape (H, W, 2) where grid[y, x] = [x, y]
    """
    xs = jnp.arange(width, dtype=jnp.float32)
    ys = jnp.arange(height, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(xs, ys)
    return jnp.stack([xx, yy], axis=-1)


def apply_homography_to_points(H: jax.Array, points: jax.Array) -> jax.Array:
    """Apply homography to 2D points.

    Args:
        H: 3x3 homography matrix
        points: Points of shape (..., 2)

    Returns:
        Transformed points of shape (..., 2)
    """
    # Convert to homogeneous coordinates
    ones = jnp.ones(points.shape[:-1] + (1,))
    points_h = jnp.concatenate([points, ones], axis=-1)

    # Apply homography
    transformed = jnp.einsum("ij,...j->...i", H, points_h)

    # Convert back to Cartesian
    return transformed[..., :2] / transformed[..., 2:3]


def warp_perspective(img: jax.Array, H: jax.Array) -> jax.Array:
    """Warp image using homography via inverse warping.

    Args:
        img: Image of shape (H, W)
        H: 3x3 homography matrix

    Returns:
        Warped image of shape (H, W)
    """
    h, w = img.shape

    # Create output grid
    grid = create_grid(h, w)

    # Apply inverse homography to get source coordinates
    H_inv = jnp.linalg.inv(H)
    src_coords = apply_homography_to_points(H_inv, grid)

    # Sample from source image
    return bilinear_sample(img, src_coords)


def get_four_corners(height: int, width: int) -> jax.Array:
    """Get the 4 corner points of an image.

    Returns:
        Array of shape (4, 2) with corners [TL, TR, BR, BL]
    """
    return jnp.array(
        [
            [0.0, 0.0],  # Top-left
            [width - 1.0, 0.0],  # Top-right
            [width - 1.0, height - 1.0],  # Bottom-right
            [0.0, height - 1.0],  # Bottom-left
        ]
    )


def compute_corner_error(H_est: jax.Array, H_gt: jax.Array, image_size: tuple[int, int]) -> jax.Array:
    """Compute mean corner error between estimated and ground truth homographies.

    Args:
        H_est: Estimated 3x3 homography
        H_gt: Ground truth 3x3 homography
        image_size: (height, width)

    Returns:
        Mean corner error in pixels (MACE - Mean Average Corner Error)
    """
    h, w = image_size
    corners = get_four_corners(h, w)

    # Transform corners
    corners_est = apply_homography_to_points(H_est, corners)
    corners_gt = apply_homography_to_points(H_gt, corners)

    # Compute L2 distances
    errors = jnp.linalg.norm(corners_est - corners_gt, axis=-1)
    return jnp.mean(errors)


# =============================================================================
# Homography Variable and Cost
# =============================================================================


class HomographyVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.zeros(8),
):
    """8-parameter homography variable (Euclidean parameterization).

    The homography is parameterized as:
        H = [[1+h0, h1, h2],
             [h3, 1+h4, h5],
             [h6, h7, 1]]

    This ensures H starts as identity when parameters are zero.
    """

    pass


def params_to_homography(params: jax.Array) -> jax.Array:
    """Convert 8 parameters to a 3x3 homography matrix."""
    h = params
    H = jnp.array(
        [
            [1.0 + h[0], h[1], h[2]],
            [h[3], 1.0 + h[4], h[5]],
            [h[6], h[7], 1.0],
        ]
    )
    return H


def homography_to_params(H: jax.Array) -> jax.Array:
    """Convert a 3x3 homography matrix to 8 parameters."""
    H = H / H[2, 2]  # Normalize
    return jnp.array(
        [
            H[0, 0] - 1.0,
            H[0, 1],
            H[0, 2],
            H[1, 0],
            H[1, 1] - 1.0,
            H[1, 2],
            H[2, 0],
            H[2, 1],
        ]
    )


@jaxls.Cost.factory
def feature_alignment_cost(
    vals: jaxls.VarValues,
    h_var: HomographyVar,
    sample_points: jax.Array,
    feat1: jax.Array,
    feat2: jax.Array,
) -> jax.Array:
    """Cost for aligning feature maps via homography.

    This computes the full residual vector over all sample points in a single cost.

    Args:
        vals: Variable values
        h_var: Homography variable (single, not batched)
        sample_points: Points at which to sample features, shape (N, 2)
        feat1: Source feature map, shape (H, W)
        feat2: Target feature map, shape (H, W)

    Returns:
        Residual vector of shape (N,)
    """
    params = vals[h_var]
    H = params_to_homography(params)

    # Warp all sample points from source to target frame
    warped_points = apply_homography_to_points(H, sample_points)

    # Sample features at original and warped locations (vectorized)
    f1_samples = jax.vmap(lambda p: bilinear_sample(feat1, p))(sample_points)
    f2_samples = jax.vmap(lambda p: bilinear_sample(feat2, p))(warped_points)

    # Feature difference
    return f1_samples - f2_samples


# =============================================================================
# Feature CNN
# =============================================================================


def init_feature_cnn(key: jax.Array, num_filters: int = 16) -> dict:
    """Initialize a simple 2-layer CNN for feature extraction.

    Architecture:
        Conv 3x3 -> ReLU -> Conv 3x3 -> (output)

    Args:
        key: Random key for initialization
        num_filters: Number of filters per layer

    Returns:
        Dictionary of CNN parameters
    """
    keys = jax.random.split(key, 2)

    # He initialization for ReLU
    scale1 = jnp.sqrt(2.0 / 9.0)  # 3x3 kernel, 1 input channel
    scale2 = jnp.sqrt(2.0 / (9.0 * num_filters))

    return {
        "conv1_w": jax.random.normal(keys[0], (3, 3, 1, num_filters)) * scale1,
        "conv1_b": jnp.zeros(num_filters),
        "conv2_w": jax.random.normal(keys[1], (3, 3, num_filters, 1)) * scale2,
        "conv2_b": jnp.zeros(1),
    }


def apply_feature_cnn(params: dict, img: jax.Array) -> jax.Array:
    """Apply feature CNN to extract feature map.

    Args:
        params: CNN parameters
        img: Input image of shape (H, W)

    Returns:
        Feature map of shape (H, W)
    """
    # Add batch and channel dimensions: (H, W) -> (1, H, W, 1)
    x = img[None, :, :, None]

    # Conv1 + ReLU
    x = jax.lax.conv_general_dilated(
        x,
        params["conv1_w"],
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    x = x + params["conv1_b"]
    x = jax.nn.relu(x)

    # Conv2 (no final activation - let the values be positive or negative)
    x = jax.lax.conv_general_dilated(
        x,
        params["conv2_w"],
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    x = x + params["conv2_b"]

    # Remove batch and channel dimensions: (1, H, W, 1) -> (H, W)
    return x[0, :, :, 0]


# =============================================================================
# Homography Estimation via LM Solver
# =============================================================================


def generate_sample_points(
    height: int,
    width: int,
    num_points: int,
    key: jax.Array,
    margin: int = 4,
) -> jax.Array:
    """Generate random sample points for feature matching.

    Args:
        height, width: Image dimensions
        num_points: Number of points to sample
        key: Random key
        margin: Margin from image edges

    Returns:
        Sample points of shape (num_points, 2)
    """
    k1, k2 = jax.random.split(key)
    xs = jax.random.uniform(k1, (num_points,), minval=margin, maxval=width - margin)
    ys = jax.random.uniform(k2, (num_points,), minval=margin, maxval=height - margin)
    return jnp.stack([xs, ys], axis=-1)


def build_and_solve_homography(
    feat1: jax.Array,
    feat2: jax.Array,
    sample_points: jax.Array,
    initial_params: jax.Array | None = None,
) -> jax.Array:
    """Build homography estimation problem and solve differentiably.

    Args:
        feat1: Source feature map
        feat2: Target feature map (warped)
        sample_points: Points at which to match features, shape (N, 2)
        initial_params: Optional initial guess for homography parameters

    Returns:
        Estimated 3x3 homography matrix
    """
    h_var = HomographyVar(0)

    # Create single cost that computes all residuals
    cost = feature_alignment_cost(
        h_var,
        sample_points,
        feat1,
        feat2,
    )

    # Build and solve
    initial_vals = None
    if initial_params is not None:
        initial_vals = jaxls.VarValues.make([h_var.with_value(initial_params)])

    problem = jaxls.LeastSquaresProblem([cost], [h_var]).analyze()
    solution = problem.solve_differentiable(
        linear_solver="dense_cholesky",
        initial_vals=initial_vals,
        verbose=False,
    )

    # Convert parameters to homography matrix
    return params_to_homography(solution[h_var])


# =============================================================================
# Training Loop
# =============================================================================


def create_training_batch(
    images: list[jax.Array],
    key: jax.Array,
    num_samples: int = 256,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Create a training batch from source images.

    Args:
        images: List of source images
        key: Random key
        num_samples: Number of sample points for feature matching

    Returns:
        (source_img, target_img, H_gt, sample_points)
    """
    keys = jax.random.split(key, 3)

    # Select random image
    img_idx = jax.random.randint(keys[0], (), 0, len(images))
    source_img = images[img_idx]

    # Generate random homography
    h, w = source_img.shape
    H_gt = generate_random_homography(keys[1], (h, w))

    # Warp source to get target
    target_img = warp_perspective(source_img, H_gt)

    # Generate sample points
    sample_points = generate_sample_points(h, w, num_samples, keys[2])

    return source_img, target_img, H_gt, sample_points


def main():
    print("=" * 70)
    print("Deep Homography: Learning Features for Differentiable Image Alignment")
    print("=" * 70)
    print()

    # Parameters
    num_training_iters = 50
    num_sample_points = 128
    learning_rate = 0.01
    image_size = (64, 64)

    print("Configuration:")
    print(f"  Image size: {image_size}")
    print(f"  Sample points: {num_sample_points}")
    print(f"  Training iterations: {num_training_iters}")
    print(f"  Learning rate: {learning_rate}")
    print()

    # Load images
    print("Loading sample images...")
    images = load_sample_images()
    print(f"  Loaded {len(images)} images of size {images[0].shape}")
    print()

    # Initialize CNN
    key = jax.random.key(42)
    key, cnn_key = jax.random.split(key)
    cnn_params = init_feature_cnn(cnn_key)
    print("Initialized feature CNN:")
    print(f"  conv1_w: {cnn_params['conv1_w'].shape}")
    print(f"  conv2_w: {cnn_params['conv2_w'].shape}")
    print()

    # Generate fixed training data for consistent evaluation
    key, data_key = jax.random.split(key)
    source_img, target_img, H_gt, sample_points = create_training_batch(
        images, data_key, num_sample_points
    )

    print(f"Training data:")
    print(f"  Source image: {source_img.shape}, range [{float(source_img.min()):.2f}, {float(source_img.max()):.2f}]")
    print(f"  GT homography corner displacement: {float(compute_corner_error(jnp.eye(3), H_gt, image_size)):.2f} px")
    print()

    # =========================================================================
    # Define Loss Function
    # =========================================================================

    def loss_fn(params: dict) -> jax.Array:
        """Compute corner error loss for the training example.

        The CNN extracts features, LM solver estimates homography from features,
        and we measure how close the estimated homography is to ground truth.
        """
        # Extract features using CNN
        feat1 = apply_feature_cnn(params, source_img)
        feat2 = apply_feature_cnn(params, target_img)

        # Solve for homography
        H_est = build_and_solve_homography(feat1, feat2, sample_points)

        # Compute corner error as loss
        return compute_corner_error(H_est, H_gt, image_size)

    # JIT compile
    loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(cnn_params)

    # =========================================================================
    # Training Loop
    # =========================================================================

    print("=" * 70)
    print("Training")
    print("=" * 70)
    print()

    print(f"{'Iter':>5} | {'Corner Error (px)':>18} | {'Grad Norm':>12}")
    print("-" * 45)

    losses = []
    for i in range(num_training_iters):
        loss, grads = loss_and_grad_fn(cnn_params)
        updates, opt_state = optimizer.update(grads, opt_state, cnn_params)
        cnn_params = optax.apply_updates(cnn_params, updates)

        losses.append(float(loss))

        # Compute gradient norm
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree.leaves(grads))
        )

        if i % 10 == 0 or i == num_training_iters - 1:
            print(f"{i:>5} | {float(loss):>18.4f} | {float(grad_norm):>12.6f}")

    print()

    # =========================================================================
    # Evaluation
    # =========================================================================

    print("=" * 70)
    print("Evaluation")
    print("=" * 70)
    print()

    # Evaluate learned features
    feat1_learned = apply_feature_cnn(cnn_params, source_img)
    feat2_learned = apply_feature_cnn(cnn_params, target_img)
    H_est_learned = build_and_solve_homography(feat1_learned, feat2_learned, sample_points)
    error_learned = float(compute_corner_error(H_est_learned, H_gt, image_size))

    # Compare with raw pixel matching
    H_est_raw = build_and_solve_homography(source_img, target_img, sample_points)
    error_raw = float(compute_corner_error(H_est_raw, H_gt, image_size))

    # Random CNN (before training)
    key, rand_key = jax.random.split(key)
    random_cnn_params = init_feature_cnn(rand_key)
    feat1_rand = apply_feature_cnn(random_cnn_params, source_img)
    feat2_rand = apply_feature_cnn(random_cnn_params, target_img)
    H_est_rand = build_and_solve_homography(feat1_rand, feat2_rand, sample_points)
    error_rand = float(compute_corner_error(H_est_rand, H_gt, image_size))

    print("Mean Average Corner Error (MACE) in pixels:")
    print(f"  {'Method':<25} | {'MACE (px)':>12}")
    print(f"  {'-' * 25}-+-{'-' * 12}")
    print(f"  {'Learned CNN features':<25} | {error_learned:>12.4f}")
    print(f"  {'Raw pixel matching':<25} | {error_raw:>12.4f}")
    print(f"  {'Random CNN features':<25} | {error_rand:>12.4f}")
    print()

    # Test on new examples
    print("Generalization to new homographies:")
    key, test_key = jax.random.split(key)
    test_keys = jax.random.split(test_key, 5)

    learned_errors = []
    raw_errors = []

    for tk in test_keys:
        _, target_test, H_test, points_test = create_training_batch(images, tk, num_sample_points)
        # Use same source image
        feat1_test = apply_feature_cnn(cnn_params, source_img)
        feat2_test = apply_feature_cnn(cnn_params, target_test)

        H_est_test = build_and_solve_homography(feat1_test, feat2_test, points_test)
        learned_errors.append(float(compute_corner_error(H_est_test, H_test, image_size)))

        H_est_raw_test = build_and_solve_homography(source_img, target_test, points_test)
        raw_errors.append(float(compute_corner_error(H_est_raw_test, H_test, image_size)))

    print(f"  {'Method':<25} | {'Mean MACE':>12} | {'Std':>12}")
    print(f"  {'-' * 25}-+-{'-' * 12}-+-{'-' * 12}")
    print(f"  {'Learned CNN features':<25} | {onp.mean(learned_errors):>12.4f} | {onp.std(learned_errors):>12.4f}")
    print(f"  {'Raw pixel matching':<25} | {onp.mean(raw_errors):>12.4f} | {onp.std(raw_errors):>12.4f}")
    print()

    # =========================================================================
    # Summary
    # =========================================================================

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("This example demonstrated bilevel optimization for learning feature")
    print("representations that improve homography estimation:")
    print()
    print("  1. Inner problem: Homography estimation from feature alignment")
    print("     - Minimize feature MSE between warped source and target")
    print("     - 8 DOF homography solved via jaxls Levenberg-Marquardt")
    print()
    print("  2. Outer problem: Learn CNN that produces easy-to-align features")
    print("     - Loss: 4-corner reprojection error of estimated homography")
    print("     - Gradients flow through solver via adjoint method")
    print()
    print("Key insight: The CNN learns to extract features that make the")
    print("homography estimation optimization problem easier to solve correctly.")
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

    # Create image planes
    plane_size = 2.0
    plane_spacing = 2.5

    # Source image
    source_rgb = onp.stack([onp.array(source_img)] * 3, axis=-1)
    server.scene.add_image(
        "/images/source",
        image=source_rgb,
        render_width=plane_size,
        render_height=plane_size,
        position=(-plane_spacing, 0.0, 0.0),
    )
    server.scene.add_label(
        "/labels/source",
        "Source",
        position=(-plane_spacing, -1.3, 0.0),
    )

    # Target image (GT warped)
    target_rgb = onp.stack([onp.array(target_img)] * 3, axis=-1)
    server.scene.add_image(
        "/images/target",
        image=target_rgb,
        render_width=plane_size,
        render_height=plane_size,
        position=(0.0, 0.0, 0.0),
    )
    server.scene.add_label(
        "/labels/target",
        "Target (GT Warp)",
        position=(0.0, -1.3, 0.0),
    )

    # Estimated warp result
    warped_est = warp_perspective(source_img, H_est_learned)
    warped_rgb = onp.stack([onp.array(warped_est)] * 3, axis=-1)
    server.scene.add_image(
        "/images/warped_est",
        image=warped_rgb,
        render_width=plane_size,
        render_height=plane_size,
        position=(plane_spacing, 0.0, 0.0),
    )
    server.scene.add_label(
        "/labels/warped_est",
        "Estimated Warp",
        position=(plane_spacing, -1.3, 0.0),
    )

    # Feature maps
    feat1_vis = (feat1_learned - feat1_learned.min()) / (feat1_learned.max() - feat1_learned.min() + 1e-8)
    feat1_rgb = onp.stack([onp.array(feat1_vis)] * 3, axis=-1)
    server.scene.add_image(
        "/images/feat1",
        image=feat1_rgb,
        render_width=plane_size,
        render_height=plane_size,
        position=(-plane_spacing, 2.5, 0.0),
    )
    server.scene.add_label(
        "/labels/feat1",
        "Source Features",
        position=(-plane_spacing, 1.2, 0.0),
    )

    feat2_vis = (feat2_learned - feat2_learned.min()) / (feat2_learned.max() - feat2_learned.min() + 1e-8)
    feat2_rgb = onp.stack([onp.array(feat2_vis)] * 3, axis=-1)
    server.scene.add_image(
        "/images/feat2",
        image=feat2_rgb,
        render_width=plane_size,
        render_height=plane_size,
        position=(0.0, 2.5, 0.0),
    )
    server.scene.add_label(
        "/labels/feat2",
        "Target Features",
        position=(0.0, 1.2, 0.0),
    )

    # Draw corner correspondences
    h, w = image_size
    corners = get_four_corners(h, w)
    corners_gt = apply_homography_to_points(H_gt, corners)
    corners_est = apply_homography_to_points(H_est_learned, corners)

    # Scale corners to visualization space
    def corner_to_3d(corner: jax.Array, x_offset: float) -> onp.ndarray:
        """Convert corner coordinates to 3D visualization position."""
        x = float(corner[0]) / w * plane_size - plane_size / 2 + x_offset
        y = float(corner[1]) / h * plane_size - plane_size / 2
        return onp.array([x, -y, 0.1])

    corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for i, (c_src, c_gt, c_est) in enumerate(zip(corners, corners_gt, corners_est)):
        color = corner_colors[i]

        # Source corner
        pos_src = corner_to_3d(c_src, -plane_spacing)
        server.scene.add_icosphere(
            f"/corners/source_{i}",
            radius=0.05,
            position=tuple(pos_src),
            color=color,
        )

        # GT target corner
        pos_gt = corner_to_3d(c_gt, 0.0)
        server.scene.add_icosphere(
            f"/corners/target_gt_{i}",
            radius=0.05,
            position=tuple(pos_gt),
            color=color,
        )

        # Estimated target corner
        pos_est = corner_to_3d(c_est, plane_spacing)
        server.scene.add_icosphere(
            f"/corners/target_est_{i}",
            radius=0.05,
            position=tuple(pos_est),
            color=color,
        )

        # Draw error line between GT and estimated corners on middle image
        pos_gt_middle = corner_to_3d(c_gt, 0.0)
        pos_est_middle = corner_to_3d(c_est, 0.0)
        server.scene.add_line_segments(
            f"/corners/error_{i}",
            points=onp.array([[pos_gt_middle, pos_est_middle]]),
            colors=(255, 100, 100),
            line_width=2.0,
        )

    # Add grid
    server.scene.add_grid("/grid", width=10.0, height=6.0, cell_size=0.5)

    print()
    print("Legend:")
    print("  Top row:    Source features | Target features")
    print("  Bottom row: Source image    | Target (GT)     | Estimated warp")
    print("  Colored spheres: 4 corner correspondences")
    print("  Red lines: Corner estimation errors")
    print()
    print(f"Final MACE: {error_learned:.4f} px")
    print()
    print(f"Visualization: http://{server.get_host()}:{server.get_port()}")
    server.sleep_forever()


if __name__ == "__main__":
    main()
