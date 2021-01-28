import dataclasses
from typing import List, Optional, Type, TypeVar

import fannypack
from jax import numpy as jnp

import jaxfg

T = TypeVar("T", bound="_KittiStruct")

DATASET_URLS = {
    "kitti_00.hdf5": "https://drive.google.com/file/d/1DSbwoYPpD9sAKnazBHa5EY2MxI4dCEc0/view?usp=sharing",
    "kitti_01.hdf5": "https://drive.google.com/file/d/1zDqi1oTcIrUhwcUe-xWphVNcBu-Y00Og/view?usp=sharing",
    "kitti_02.hdf5": "https://drive.google.com/file/d/1h1nqyXLP-TuUHJe7q97Plt90vvP3p8yS/view?usp=sharing",
    "kitti_03.hdf5": "https://drive.google.com/file/d/1ls0lIT1nN7eXmOI-1ZQDcD0tBZ9foCgr/view?usp=sharing",
    "kitti_04.hdf5": "https://drive.google.com/file/d/1YcRVSD9FCL6ZP_bt1Q0BVyqWTkaQxTK7/view?usp=sharing",
    "kitti_05.hdf5": "https://drive.google.com/file/d/1xFRJKS8k56UhMYrKVWw6eg7GKPtoJ9df/view?usp=sharing",
    "kitti_06.hdf5": "https://drive.google.com/file/d/1GBGh39gcsLofaT63UgXjCOgPL1n8Rpii/view?usp=sharing",
    "kitti_07.hdf5": "https://drive.google.com/file/d/1Dmr7gXFXX4Iiec3JWRrNGrV1FXoBWPEy/view?usp=sharing",
    "kitti_08.hdf5": "https://drive.google.com/file/d/1TTIlFjxXf-YpyRodB88rS49ncHGJ6C5o/view?usp=sharing",
    "kitti_09.hdf5": "https://drive.google.com/file/d/1GKJHCMj6q5hZol_gAZX9Iw5oLSXQpYIc/view?usp=sharing",
    "kitti_10.hdf5": "https://drive.google.com/file/d/1HCKczAcknVZFSfbT4W5138EzLz3EaNeD/view?usp=sharing",
}


@dataclasses.dataclass(frozen=True)
class _KittiStruct:
    image: Optional[jnp.ndarray] = None
    image_diff: Optional[jnp.ndarray] = None
    x: Optional[jnp.ndarray] = None
    y: Optional[jnp.ndarray] = None
    theta: Optional[jnp.ndarray] = None
    linear_vel: Optional[jnp.ndarray] = None
    angular_vel: Optional[jnp.ndarray] = None

    @classmethod
    def mirror_data(cls: Type[T], self: T) -> T:
        """Data augmentation helper: mirror a sequence."""

        assert self.image is not None
        assert self.image_diff is not None
        assert self.x is not None
        assert self.y is not None
        assert self.theta is not None
        assert self.linear_vel is not None
        assert self.angular_vel is not None

        return cls(
            image=self.image[:, :, ::-1, :],  # (N, rows, columns, channels)
            image_diff=self.image_diff[:, :, ::-1, :],  # (N, rows, columns, channels)
            x=self.x,
            y=-self.y,
            theta=-self.theta,
            linear_vel=self.linear_vel,
            angular_vel=-self.angular_vel,
        )


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)  # Doesn't do anything, just for jedi
class KittiStructNormalized(_KittiStruct):
    pass


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)  # Doesn't do anything, just for jedi
class KittiStructRaw(_KittiStruct):
    def normalize(self) -> KittiStructNormalized:
        return self


def load_trajectories(train: bool) -> List[KittiStructNormalized]:

    # We intentionally exclude 01 from all datasets, because it's very different
    # (highway driving)
    files: List[str]
    if train:
        files = [
            "kitti_00.hdf5",
            "kitti_02.hdf5",
            "kitti_03.hdf5",
            "kitti_04.hdf5",
            "kitti_05.hdf5",
            "kitti_06.hdf5",
            "kitti_07.hdf5",
            "kitti_08.hdf5",
            "kitti_09.hdf5",
        ]
    else:
        files = [
            "kitti_10.hdf5",
        ]

    assert len(set(files) - set(DATASET_URLS.keys())) == 0

    trajectories: List[KittiStructNormalized] = []
    for filename in files:
        with fannypack.data.TrajectoriesFile(
            fannypack.data.cached_drive_file(filename, DATASET_URLS[filename])
        ) as traj_file:
            for trajectory in traj_file:
                assert len(trajectory.keys()) == len(dataclasses.fields(KittiStructRaw))
                trajectories.append(KittiStructRaw(**trajectory).normalize())

    print("Concatenating trajectories...")
    concat: KittiStructNormalized = jaxfg.utils.pytree_concatenate(*trajectories)
    print(
        "Image moments:",
        jnp.mean(concat.image.reshape((-1, 3)), axis=0),
        jnp.std(concat.image.reshape((-1, 3)), axis=0),
    )
    print(
        "Image diff moments:",
        jnp.mean(concat.image_diff.reshape((-1, 3)), axis=0),
        jnp.std(concat.image_diff.reshape((-1, 3)), axis=0),
    )
    print(
        "Linear vel moments",
        jnp.mean(concat.linear_vel),
        jnp.std(concat.linear_vel),
    )
    print(
        "Angular vel moments",
        jnp.mean(concat.angular_vel),
        jnp.std(concat.angular_vel),
    )

    # image: Optional[jnp.ndarray] = None
    # image_diff: Optional[jnp.ndarray] = None
    # x: Optional[jnp.ndarray] = None
    # y: Optional[jnp.ndarray] = None
    # theta: Optional[jnp.ndarray] = None
    # linear_vel: Optional[jnp.ndarray] = None
    # angular_vel: Optional[jnp.ndarray] = None
