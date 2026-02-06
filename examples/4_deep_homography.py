"""Deep Homography: Differentiable image alignment via learned features.

Demonstrates bilevel optimization with `solve_differentiable()`:
- Inner problem: Estimate homography by minimizing feature alignment error
- Outer problem: Minimize 4-corner reprojection error to train a CNN

The CNN learns to extract features that make homography estimation easier.

Uses real images with synthetic homographies (same methodology as HPatches benchmark).
"""

import logging

logging.getLogger("jaxls").setLevel(logging.ERROR)
try:
    from loguru import logger

    logger.disable("jaxls")
except ImportError:
    pass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize

import jaxls


def main():
    print("Deep Homography: Learning Features for Differentiable Image Alignment")
    print("=" * 65)

    # Setup
    key = jax.random.key(42)
    size, num_points, num_iters, lr = 64, 128, 50, 0.002

    # Load real image and create warped pair with synthetic homography
    # This mirrors HPatches methodology: real images + synthetic warps
    source = load_image(size)
    key, k_h, k_pts = jax.random.split(key, 3)
    H_gt = random_homography(k_h, size)
    target = warp_image(source, H_gt)

    # Sample points for feature matching
    xs = jax.random.uniform(k_pts, (num_points,), minval=4, maxval=size - 4)
    ys = jax.random.uniform(
        jax.random.split(k_pts)[0], (num_points,), minval=4, maxval=size - 4
    )
    sample_points = jnp.stack([xs, ys], axis=-1)

    # Initialize CNN
    key, cnn_key = jax.random.split(key)
    cnn_params = init_cnn(cnn_key)

    print(f"Image: astronaut (skimage), Size: {size}x{size}")
    print(f"Sample points: {num_points}, Iterations: {num_iters}")
    print(
        f"GT corner displacement: {float(corner_error(jnp.eye(3), H_gt, size)):.2f} px"
    )
    print()

    # Loss function that returns both loss and intermediate results for snapshots
    def loss_fn(params):
        feat1 = apply_cnn(params, source)
        feat2 = apply_cnn(params, target)
        H_est = solve_homography(feat1, feat2, sample_points)
        return corner_error(H_est, H_gt, size)

    def forward_with_intermediates(params):
        """Forward pass returning features and estimated homography."""
        feat1 = apply_cnn(params, source)
        feat2 = apply_cnn(params, target)
        H_est = solve_homography(feat1, feat2, sample_points)
        err = corner_error(H_est, H_gt, size)
        return feat1, feat2, H_est, err

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(cnn_params)

    # Iterations at which to capture snapshots for visualization
    snapshot_iters = [0, 15, 30, num_iters - 1]
    snapshots = []  # List of (iter, feat1, feat2, H_est, error) tuples

    # Training loop
    print("Training:")
    losses = []
    for i in range(num_iters):
        # Capture snapshot before update at specified iterations
        if i in snapshot_iters:
            feat1, feat2, H_est, err = forward_with_intermediates(cnn_params)
            snapshots.append((i, feat1, feat2, H_est, float(err)))

        loss, grads = loss_and_grad(cnn_params)
        updates, opt_state = optimizer.update(grads, opt_state, cnn_params)
        cnn_params = optax.apply_updates(cnn_params, updates)
        losses.append(float(loss))
        if i % 5 == 0 or i == num_iters - 1:
            print(f"  Iter {i:3d}: corner error = {float(loss):.4f} px")

    # Capture final snapshot after last update
    feat1, feat2, H_est, err = forward_with_intermediates(cnn_params)
    # Replace the last snapshot with post-training results
    snapshots[-1] = (num_iters - 1, feat1, feat2, H_est, float(err))

    # Evaluation
    print()
    print("Comparison:")
    H_learned = H_est
    H_raw = solve_homography(source, target, sample_points)

    err_learned = float(corner_error(H_learned, H_gt, size))
    err_raw = float(corner_error(H_raw, H_gt, size))
    print(f"  Learned CNN features: {err_learned:.4f} px")
    print(f"  Raw pixel matching:   {err_raw:.4f} px")

    # Evolution visualization: 4 columns (iterations) x 4 rows
    # Row 0: Source features
    # Row 1: Target features
    # Row 2: Estimated warp result
    # Row 3: Training loss curve (spans all columns)
    n_snapshots = len(snapshots)
    fig = plt.figure(figsize=(12, 10))

    # Create grid: 4 rows, n_snapshots columns
    # Last row spans all columns for the loss plot
    gs = fig.add_gridspec(
        4, n_snapshots, height_ratios=[1, 1, 1, 0.8], hspace=0.3, wspace=0.15
    )

    for col, (iter_num, feat1, feat2, H_est, err) in enumerate(snapshots):
        # Row 0: Source features
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(feat1, cmap="viridis")
        ax0.set_title(f"Iter {iter_num}", fontsize=10)
        ax0.axis("off")
        if col == 0:
            ax0.set_ylabel("Src Features", fontsize=9)

        # Row 1: Target features
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(feat2, cmap="viridis")
        ax1.axis("off")
        if col == 0:
            ax1.set_ylabel("Tgt Features", fontsize=9)

        # Row 2: Estimated warp result
        ax2 = fig.add_subplot(gs[2, col])
        warped = warp_image(source, H_est)
        ax2.imshow(warped, cmap="gray")
        ax2.set_title(f"err={err:.2f}px", fontsize=9)
        ax2.axis("off")
        if col == 0:
            ax2.set_ylabel("Est. Warp", fontsize=9)

    # Row 3: Loss curve spanning all columns
    ax_loss = fig.add_subplot(gs[3, :])
    ax_loss.plot(losses, "b-", linewidth=2, label="Corner Error")
    # Mark snapshot iterations
    for iter_num, _, _, _, err in snapshots:
        ax_loss.axvline(x=iter_num, color="r", linestyle="--", alpha=0.5)
        ax_loss.plot(iter_num, err, "ro", markersize=8)
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Corner Error (px)")
    ax_loss.set_title("Training Progress (red dots = snapshots above)")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    plt.suptitle(
        "Deep Homography: Feature Evolution During Training", fontsize=12, y=0.98
    )
    output_path = "/tmp/claude/deep_homography_evolution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()

# === Real Image Loading ===


def load_image(size: int = 64) -> jax.Array:
    """Load a real photograph and resize for the example.

    Uses skimage's astronaut image - a real photograph that provides
    natural texture and edges for feature learning.
    """
    img = rgb2gray(data.astronaut())
    img = resize(img, (size, size), anti_aliasing=True)
    return jnp.array(img, dtype=jnp.float32)


def random_homography(key: jax.Array, size: int = 64) -> jax.Array:
    """Generate a random homography with rotation, translation, and scale."""
    keys = jax.random.split(key, 4)
    cx, cy = size / 2, size / 2

    # Moderate perturbations - challenging but tractable
    theta = jax.random.uniform(keys[0], (), minval=-0.25, maxval=0.25)
    tx = jax.random.uniform(keys[1], (), minval=-6.0, maxval=6.0)
    ty = jax.random.uniform(keys[2], (), minval=-6.0, maxval=6.0)
    s = jax.random.uniform(keys[3], (), minval=0.9, maxval=1.1)

    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
    R = jnp.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
    T = jnp.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    S = jnp.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    T_center = jnp.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    T_uncenter = jnp.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])

    H = T_uncenter @ S @ R @ T_center @ T
    return H / H[2, 2]


# === Image Warping ===


def bilinear_sample(img: jax.Array, coords: jax.Array) -> jax.Array:
    """Sample image at subpixel coordinates using bilinear interpolation."""
    h, w = img.shape
    x, y = coords[..., 0], coords[..., 1]
    x0, y0 = jnp.floor(x).astype(jnp.int32), jnp.floor(y).astype(jnp.int32)
    x1, y1 = x0 + 1, y0 + 1

    x0c, x1c = jnp.clip(x0, 0, w - 1), jnp.clip(x1, 0, w - 1)
    y0c, y1c = jnp.clip(y0, 0, h - 1), jnp.clip(y1, 0, h - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (
        wa * img[y0c, x0c]
        + wb * img[y1c, x0c]
        + wc * img[y0c, x1c]
        + wd * img[y1c, x1c]
    )


def apply_homography(H: jax.Array, points: jax.Array) -> jax.Array:
    """Apply homography to 2D points."""
    ones = jnp.ones(points.shape[:-1] + (1,))
    pts_h = jnp.concatenate([points, ones], axis=-1)
    transformed = jnp.einsum("ij,...j->...i", H, pts_h)
    return transformed[..., :2] / transformed[..., 2:3]


def warp_image(img: jax.Array, H: jax.Array) -> jax.Array:
    """Warp image using homography via inverse warping."""
    h, w = img.shape
    xs, ys = jnp.meshgrid(
        jnp.arange(w, dtype=jnp.float32), jnp.arange(h, dtype=jnp.float32)
    )
    grid = jnp.stack([xs, ys], axis=-1)
    src_coords = apply_homography(jnp.linalg.inv(H), grid)
    return bilinear_sample(img, src_coords)


def corner_error(H_est: jax.Array, H_gt: jax.Array, size: int) -> jax.Array:
    """Compute mean corner error between estimated and ground truth homographies."""
    corners = jnp.array(
        [[0.0, 0.0], [size - 1.0, 0.0], [size - 1.0, size - 1.0], [0.0, size - 1.0]]
    )
    corners_est = apply_homography(H_est, corners)
    corners_gt = apply_homography(H_gt, corners)
    return jnp.mean(jnp.linalg.norm(corners_est - corners_gt, axis=-1))


# === Homography Variable & Cost ===


class HomographyVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(8)):
    """8-parameter homography: H = [[1+h0,h1,h2],[h3,1+h4,h5],[h6,h7,1]]."""


def params_to_H(params: jax.Array) -> jax.Array:
    """Convert 8 parameters to 3x3 homography matrix."""
    h = params
    return jnp.array(
        [[1.0 + h[0], h[1], h[2]], [h[3], 1.0 + h[4], h[5]], [h[6], h[7], 1.0]]
    )


@jaxls.Cost.factory
def feature_alignment_cost(
    vals: jaxls.VarValues,
    h_var: HomographyVar,
    sample_points: jax.Array,
    feat1: jax.Array,
    feat2: jax.Array,
) -> jax.Array:
    """Cost for aligning feature maps via homography."""
    H = params_to_H(vals[h_var])
    warped_pts = apply_homography(H, sample_points)
    f1 = jax.vmap(lambda p: bilinear_sample(feat1, p))(sample_points)
    f2 = jax.vmap(lambda p: bilinear_sample(feat2, p))(warped_pts)
    return f1 - f2


# === Simple CNN ===


def init_cnn(key: jax.Array, num_filters: int = 16) -> dict:
    """Initialize a simple 1-layer CNN for feature extraction."""
    return {
        "conv_w": jax.random.normal(key, (3, 3, 1, num_filters)) * jnp.sqrt(2.0 / 9.0),
        "conv_b": jnp.zeros(num_filters),
    }


def apply_cnn(params: dict, img: jax.Array) -> jax.Array:
    """Apply CNN: Conv3x3 -> ReLU -> mean over channels."""
    x = img[None, :, :, None]  # (1, H, W, 1)
    x = jax.lax.conv_general_dilated(
        x, params["conv_w"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    x = jax.nn.relu(x + params["conv_b"])
    return x[0].mean(axis=-1)  # (H, W)


# === Solver ===


def solve_homography(
    feat1: jax.Array, feat2: jax.Array, sample_points: jax.Array
) -> jax.Array:
    """Estimate homography aligning feat1 to feat2 using jaxls."""
    h_var = HomographyVar(0)
    cost = feature_alignment_cost(h_var, sample_points, feat1, feat2)
    problem = jaxls.LeastSquaresProblem([cost], [h_var]).analyze()
    solution = problem.solve_differentiable(
        linear_solver="dense_cholesky", verbose=False
    )
    return params_to_H(solution[h_var])


# === Main ===
