import dataclasses
from typing import List

import fannypack
import jax
import numpy as onp
import torch
from jax import numpy as jnp

import jaxfg

DATASET_MEANS = {
    "image": 6.942841319444445,
    "position": 0.43589213,
    "velocity": 0.10219889,
}
DATASET_STD_DEVS = {
    "image": 41.49965651510064,
    "position": 26.558552,
    "velocity": 5.8443174,
}


@jax.partial(jaxfg.utils.register_dataclass_pytree, static_fields=("normalized",))
@dataclasses.dataclass(frozen=True)
class ToyDatasetStruct:
    """Fields in our toy dataset. Holds an array or timestep."""

    normalized: bool
    image: jnp.ndarray
    visible_pixels_count: jnp.ndarray
    position: jnp.ndarray
    velocity: jnp.ndarray

    def normalize(self) -> "ToyDatasetStruct":
        assert not self.normalized

        # Data normalization
        return dataclasses.replace(
            self,
            normalized=True,
            **{
                k: (self.__getattribute__(k) - DATASET_MEANS[k]) / DATASET_STD_DEVS[k]
                for k in DATASET_MEANS.keys()
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
            },
        )


def load_trajectories(*paths: str) -> List[ToyDatasetStruct]:
    """Grabs a list of trajectories from a set of input files."""
    trajectories = []
    for path in paths:
        with fannypack.data.TrajectoriesFile(path) as traj_file:
            for trajectory in traj_file:
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
        values = jaxfg.utils.pytree_stack(*trajectories).__getattribute__(field)
        print(f"({field}) Mean, std dev:", onp.mean(values), onp.std(values))

    return trajectories


class ToyTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, *paths: str, subsequence_length: int = 5):
        self.samples: List[ToyDatasetStruct] = []

        for trajectory in load_trajectories(*paths):
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
    def __init__(self, *paths: str):
        self.samples: List[ToyDatasetStruct] = []

        for trajectory in load_trajectories(*paths):
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
