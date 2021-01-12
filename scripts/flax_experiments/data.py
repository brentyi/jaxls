import dataclasses
from typing import List, Optional

import fannypack
import jax
import jaxfg
import numpy as onp
import torch
from jax import numpy as jnp

DATASET_MEANS = {
    "image": onp.array([8.26625, 13.777969, 6.1713543]),
    "position": onp.array([-4.0143294, -19.234016]),
    "velocity": onp.array([1.765901, 1.1825078]),
}
DATASET_STD_DEVS = {
    "image": onp.array([45.161514, 57.650227, 39.186855]),
    "position": onp.array([32.284786, 4.5147696]),
    "velocity": onp.array([8.180047, 2.5003078]),
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
        values = values.reshape((-1, values.shape[-1]))
        print(
            f"({field}) Mean, std dev:",
            onp.mean(values, axis=0),
            onp.std(values, axis=0),
        )

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
