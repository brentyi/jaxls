import dataclasses
import pathlib
from typing import List, Optional

import fannypack
import numpy as onp
from jax import numpy as jnp
from PIL import Image
from tqdm.auto import tqdm

import jaxfg

# Dataset structs


@dataclasses.dataclass(frozen=True)
class _KittiStruct:
    image: Optional[jnp.ndarray] = None
    image_diff: Optional[jnp.ndarray] = None
    x: Optional[jnp.ndarray] = None
    y: Optional[jnp.ndarray] = None
    theta: Optional[jnp.ndarray] = None
    linear_vel: Optional[jnp.ndarray] = None
    angular_vel: Optional[jnp.ndarray] = None


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)  # Doesn't do anything, just for jedi
class KittiStructNormalized(_KittiStruct):
    pass


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)  # Doesn't do anything, just for jedi
class KittiStructRaw(_KittiStruct):
    pass


# Helpers


def wrap_angle(thetas: onp.ndarray) -> onp.ndarray:
    return ((thetas - onp.pi) % (2 * onp.pi)) - onp.pi


# Actual script

fannypack.utils.pdb_safety_net()

path: pathlib.Path
directories = sorted(
    filter(
        lambda path: path.is_dir(),
        (pathlib.Path.cwd() / "data" / "kitti").iterdir(),
    )
)
assert len(directories) == 11


def load_data(pose_txt: pathlib.Path, image_dir: pathlib.Path) -> KittiStructRaw:
    """Load and pre-process KITTI data from a set of paths."""
    poses = onp.loadtxt(pose_txt)
    N = poses.shape[0]
    assert poses.shape == (N, 12)

    # Reshape poses to standard 3x4 pose matrix
    poses = poses.reshape((N, 3, 4))

    # Extract 2D poses
    # Note that we treat the XZ plane as the XY plane
    xs = poses[:, 2, 3]
    ys = -poses[:, 0, 3]

    # Extra y-axis rotation
    thetas = -onp.arctan2(-poses[:, 2, 0], poses[:, 2, 2])

    # Validate shapes
    assert xs.shape == ys.shape == thetas.shape == (N,)

    # ## Uncomment to validate thetas against jaxlie
    # import jax
    # import jaxlie
    # from jax import numpy as jnp
    # def y_radians_from_rotation_matrix(R: jnp.ndarray) -> float:
    #     """Computes a rotation about the global y-axis."""
    #     return jaxlie.SO3.from_matrix(R).log()[1]
    #
    # thetas_from_matrix_log = -jax.vmap(y_radians_from_rotation_matrix)(poses[:, :3, :3])
    # onp.testing.assert_allclose(
    #     jnp.cos(thetas_from_matrix_log), jnp.cos(thetas), atol=1e-2, rtol=1e-5
    # )
    # onp.testing.assert_allclose(
    #     jnp.sin(thetas_from_matrix_log), jnp.sin(thetas), atol=1e-2, rtol=1e-5
    # )

    # Load images
    images = onp.array(
        [
            onp.array(Image.open(image_path))
            for image_path in tqdm(tuple(image_dir.iterdir()))
        ]
    )
    assert images.shape == (N, 50, 150, 3)
    assert images.dtype == onp.uint8

    # Cast to prevent overflow when computing difference images
    images_int16 = images.astype(onp.int16)

    # Consolidate all data, matching conventions from Kloss et al:
    # > How to Train Your Differentiable Filter
    # > https://arxiv.org/pdf/2012.14313.pdf

    # time between frames is really 0.103, but life feels easier if we just don't divide
    time_delta = 1.0

    data = KittiStructRaw(
        image=images[1:-1],
        image_diff=(
            # image_diff[i] = image[i] - image[i - 1]
            # => after subtracting, we're missing the first timestep
            # => to align with velocities, we need to chop off the last timestep
            images_int16[1:]
            - images_int16[:-1]
        )[:-1],
        x=xs[1:-1],
        y=ys[1:-1],
        theta=thetas[1:-1],
        linear_vel=(
            # Note that we want: positions[i + 1] = positions[i] + velocity[i]
            #
            # velocity[i] = positions[i + 1] - positions[i]
            # => after subtracting, we're missing the last timestep
            # => to align with image differences, we need to chop off the first timestep
            onp.sqrt((xs[1:] - xs[:-1]) ** 2 + (ys[1:] - ys[:-1]) ** 2)
            / time_delta
        )[1:],
        angular_vel=(
            # Same alignment logic as linear velocity
            wrap_angle(thetas[1:] - thetas[:-1])
            / time_delta
        )[1:],
    )

    # Validate alignment
    assert onp.all(
        data.image_diff[1]
        == data.image[1].astype(onp.int16) - data.image[0].astype(onp.int16)
    )
    assert data.angular_vel[0] == wrap_angle(data.theta[1] - data.theta[0])

    return data


## Was originally going to to mirroring here, but it makes more sense to do it when the data is loaded
#
# def mirror_data(data: KittiStructRaw) -> KittiStructRaw:
#     """Data augmentation: mirror a sequence."""
#     return KittiStructRaw(
#         image=data.image[:, :, ::-1, :],  # (N, rows, columns, channels)
#         image_diff=data.image_diff[:, :, ::-1, :],  # (N, rows, columns, channels)
#         x=data.x,
#         y=-data.y,
#         theta=-data.theta,
#         linear_vel=data.linear_vel,
#         angular_vel=-data.angular_vel,
#     )


for directory in directories:

    dataset_id: str = directory.stem
    assert (
        dataset_id.isdigit() and len(dataset_id) == 2
    ), "Dataset subdirectories should be two digit numbers!"

    print("Handling", directory.stem)

    data = load_data(
        pose_txt=directory.parent / f"{dataset_id}_image1.txt",
        image_dir=directory / "image_2",
    )

    with fannypack.data.TrajectoriesFile(
        str(pathlib.Path.cwd() / "data_out" / f"kitti_{dataset_id}.hdf5"), read_only=False
    ) as traj_file:
        traj_file.resize(1)
        traj_file[0] = vars(data)
        # traj_file[1] = vars(mirror_data(data))

    # PIL.Image.open()
