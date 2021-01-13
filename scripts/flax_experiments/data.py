import dataclasses
import pathlib
from typing import List, Optional

import fannypack
import jax
import numpy as onp
import torch
from jax import numpy as jnp
from tqdm.auto import tqdm

import jaxfg

# Download Google Drive files to same directory as this file
fannypack.data.set_cache_path(pathlib.Path(__file__).parent.absolute() / ".cache/")

DATASET_URLS = {
    "toy_tracking_train.hdf5": "https://drive.google.com/file/d/175y7rlpVLcX6WJk5rHqcG5yT79CnOLab/view?usp=sharing",
    "toy_tracking_val.hdf5": "https://drive.google.com/file/d/1hPujtHgYWWHyMikzGTvv1UpL3QrZfN1i/view?usp=sharing",
}

DATASET_MEANS = {
    "image": onp.array([24.30598765, 29.76503314, 29.86749727], dtype=onp.float32),
    "position": onp.array([-0.08499543, 0.07917813], dtype=onp.float32),
    "velocity": onp.array([0.02876372, 0.06096543], dtype=onp.float32),
}
DATASET_STD_DEVS = {
    "image": onp.array([74.88154621, 81.87872827, 82.00088091], dtype=onp.float32),
    "position": onp.array([30.53421, 30.84835], dtype=onp.float32),
    "velocity": onp.array([6.636913, 6.647381], dtype=onp.float32),
}


@jax.partial(jaxfg.utils.register_dataclass_pytree, static_fields=("normalized",))
@dataclasses.dataclass(frozen=True)
class ToyDatasetStruct:
    """Fields in our toy dataset. Holds an array or timestep."""

    normalized: bool
    image: Optional[jnp.ndarray] = None
    visible_pixels_count: Optional[jnp.ndarray] = None
    position: Optional[jnp.ndarray] = None
    velocity: Optional[jnp.ndarray] = None

    def normalize(self) -> "ToyDatasetStruct":
        assert not self.normalized
        # return self

        # Data normalization
        return dataclasses.replace(
            self,
            normalized=True,
            **{
                k: (self.__getattribute__(k) - DATASET_MEANS[k]) / DATASET_STD_DEVS[k]
                for k in DATASET_MEANS.keys()
                if self.__getattribute__(k) is not None
            },
        )

    def unnormalize(self) -> "ToyDatasetStruct":
        assert self.normalized

        # Data normalization
        return dataclasses.replace(
            self,
            normalized=False,
            **{
                k: (self.__getattribute__(k) * DATASET_STD_DEVS[k]) + DATASET_MEANS[k]
                for k in DATASET_MEANS.keys()
                if self.__getattribute__(k) is not None
            },
        )


def load_trajectories(train: bool) -> List[ToyDatasetStruct]:
    """Grabs a list of trajectories from a set of input files.

    Set `train` to False to load validation set.
    """
    trajectories = []

    filename = "toy_tracking_train.hdf5" if train else "toy_tracking_val.hdf5"
    path = fannypack.data.cached_drive_file(filename, DATASET_URLS[filename])

    with fannypack.data.TrajectoriesFile(path) as traj_file:
        for trajectory in tqdm(traj_file):
            trajectory["normalized"] = False
            trajectories.append(
                ToyDatasetStruct(
                    **{
                        # Assume all dataclass field names exist as string keys
                        # in our HDF5 file
                        field.name: trajectory[field.name]
                        for field in dataclasses.fields(ToyDatasetStruct)
                    },
                ).normalize()
            )

    # Print some data statistics
    for field in ("image", "position", "velocity"):
        values = jax.tree_multimap(
            lambda *x: onp.stack(x, axis=0), *trajectories
        ).__getattribute__(field)
        values = values.reshape((-1, values.shape[-1]))
        print(
            f"({field}) Mean, std dev:",
            onp.mean(values, axis=0),
            onp.std(values, axis=0),
        )

    return trajectories


class ToyTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool, subsequence_length: int = 5):
        self.samples: List[ToyDatasetStruct] = []

        for trajectory in load_trajectories(train=train):
            timesteps = len(trajectory.image)
            index = 0
            while index + subsequence_length <= timesteps:
                self.samples.append(
                    jax.tree_util.tree_multimap(
                        lambda x: x[index : index + subsequence_length], trajectory
                    )
                )
                index += subsequence_length // 2

    def __getitem__(self, index: int) -> ToyDatasetStruct:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class ToySingleStepDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool):
        self.samples: List[ToyDatasetStruct] = []

        for trajectory in load_trajectories(train=train):
            timesteps = len(trajectory.image)
            for t in range(timesteps):
                self.samples.append(
                    jax.tree_util.tree_multimap(lambda x: x[t], trajectory)
                )

    def __getitem__(self, index: int) -> ToyDatasetStruct:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


#  @jax.jit
def collate_fn(batch, axis=0):
    return jaxfg.utils.pytree_stack(*batch, axis=axis)
