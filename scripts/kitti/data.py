import dataclasses
from typing import List, Optional, TypeVar

import fannypack
import jax
import numpy as onp
import torch
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

DATASET_MEANS = {
    "image": onp.array([88.96255, 94.19366, 92.71602]),
    "image_diff": onp.array([0.00186025, 0.00170155, 0.00212632]),
    "linear_vel": 0.89916223,
    "angular_vel": 2.0997644e-05,
}

DATASET_STD_DEVS = {
    "image": onp.array([74.88909, 76.648285, 77.99241]),
    "image_diff": onp.array([97.935776, 99.096855, 98.973915]),
    "linear_vel": 0.3150884,
    "angular_vel": 0.017592415,
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

    def get_stacked_image(self) -> jnp.ndarray:
        """Return 6-channel image for CNN."""
        return jnp.concatenate([self.image, self.image_diff], axis=-1)

    def get_stacked_velocity(self) -> jnp.ndarray:
        """Return 2-channel velocity."""
        out = jnp.stack([self.linear_vel, self.angular_vel], axis=-1)
        assert out.shape[-1] == 2
        return out

    def mirror(self: T) -> T:
        """Data augmentation helper: mirror a sequence."""

        assert self.image is not None
        assert self.image_diff is not None
        assert self.x is not None
        assert self.y is not None
        assert self.theta is not None
        assert self.linear_vel is not None
        assert self.angular_vel is not None

        return type(self)(
            image=self.image[..., :, ::-1, :],  # (N?, rows, columns, channels)
            image_diff=self.image_diff[
                ..., :, ::-1, :
            ],  # (N?, rows, columns, channels)
            x=self.x,
            y=-self.y,
            theta=-self.theta,
            linear_vel=self.linear_vel,
            angular_vel=-self.angular_vel,
        )


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)  # Doesn't do anything, just for jedi
class KittiStructNormalized(_KittiStruct):
    def unnormalize(self) -> "KittiStructRaw":
        return KittiStructRaw(
            **vars(
                dataclasses.replace(
                    self,
                    **{
                        k: (self.__getattribute__(k) * DATASET_STD_DEVS[k])
                        + DATASET_MEANS[k]
                        for k in DATASET_MEANS.keys()
                        if self.__getattribute__(k) is not None
                    },
                )
            )
        )


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)  # Doesn't do anything, just for jedi
class KittiStructRaw(_KittiStruct):
    def normalize(self) -> KittiStructNormalized:
        return KittiStructNormalized(
            **vars(
                dataclasses.replace(
                    self,
                    **{
                        k: (self.__getattribute__(k) - DATASET_MEANS[k])
                        / DATASET_STD_DEVS[k]
                        for k in DATASET_MEANS.keys()
                        if self.__getattribute__(k) is not None
                    },
                )
            )
        )


def load_trajectories(train: bool) -> List[KittiStructNormalized]:

    # We intentionally exclude 01 from all datasets, because it's very different
    # (highway driving)
    files: List[str]
    if train:
        files = [
            "kitti_00.hdf5",
            # "kitti_02.hdf5",
            # "kitti_03.hdf5",
            # "kitti_04.hdf5",
            # "kitti_05.hdf5",
            # "kitti_06.hdf5",
            # "kitti_07.hdf5",
            # "kitti_08.hdf5",
            # "kitti_09.hdf5",
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

    # # Uncomment to print statistics
    # print("Concatenating trajectories...")
    # concat: KittiStructNormalized = jaxfg.utils.pytree_concatenate(*trajectories)
    # print(
    #     "Image moments:",
    #     jnp.mean(concat.image.reshape((-1, 3)), axis=0),
    #     jnp.std(concat.image.reshape((-1, 3)), axis=0),
    # )
    # print(
    #     "Image diff moments:",
    #     jnp.mean(concat.image_diff.reshape((-1, 3)), axis=0),
    #     jnp.std(concat.image_diff.reshape((-1, 3)), axis=0),
    # )
    # print(
    #     "Linear vel moments",
    #     jnp.mean(concat.linear_vel),
    #     jnp.std(concat.linear_vel),
    # )
    # print(
    #     "Angular vel moments",
    #     jnp.mean(concat.angular_vel),
    #     jnp.std(concat.angular_vel),
    # )

    return trajectories


class KittiSubsequenceDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool, subsequence_length: int = 5):
        self.samples: List[KittiStructNormalized] = []

        for trajectory in load_trajectories(train=train):
            assert trajectory.image is not None
            timesteps = len(trajectory.image)
            index = 0
            while index + subsequence_length <= timesteps:
                self.samples.append(
                    jax.tree_util.tree_multimap(
                        lambda x: x[index : index + subsequence_length], trajectory
                    )
                )
                index += subsequence_length // 2

    def __getitem__(self, index: int) -> KittiStructNormalized:
        if index < len(self.samples):
            return self.samples[index]
        else:
            return self.samples[index - len(self.samples)].mirror()

    def __len__(self) -> int:
        return len(self.samples) * 2


class KittiSingleStepDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool):
        self.samples: List[KittiStructNormalized] = []

        for trajectory in load_trajectories(train=train):
            assert trajectory.image is not None
            timesteps = len(trajectory.image)
            for t in range(timesteps):
                self.samples.append(
                    jax.tree_util.tree_multimap(lambda x: x[t], trajectory)
                )

    def __getitem__(self, index: int) -> KittiStructNormalized:
        if index < len(self.samples):
            return self.samples[index]
        else:
            return self.samples[index - len(self.samples)].mirror()

    def __len__(self) -> int:
        return len(self.samples) * 2


#  @jax.jit
def collate_fn(batch, axis=0):
    return jaxfg.utils.pytree_stack(*batch, axis=axis)
