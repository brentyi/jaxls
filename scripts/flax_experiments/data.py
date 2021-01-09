import dataclasses
from typing import List

import fannypack
import jax
import numpy as onp
import torch
from jax import numpy as jnp

import jaxfg


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class ToyDatasetStruct:
    """Fields in our toy dataset. Holds an array or timestep."""

    image: jnp.ndarray
    visible_pixels_count: jnp.ndarray
    position: jnp.ndarray
    velocity: jnp.ndarray


def load_trajectories(*paths: str) -> List[ToyDatasetStruct]:
    """Grabs a list of trajectories from a set of input files."""
    trajectories = []
    for path in paths:
        with fannypack.data.TrajectoriesFile(path) as traj_file:
            for trajectory in traj_file:
                # Data normalization
                trajectory["image"] = (trajectory["image"] - 6.942841319444445).astype(
                    onp.float32
                )
                trajectory["image"] /= 41.49965651510064
                trajectory["position"] -= 0.43589213
                trajectory["position"] /= 26.558552
                trajectory["velocity"] -= 0.10219889
                trajectory["velocity"] /= 5.8443174

                trajectories.append(
                    ToyDatasetStruct(
                        **{
                            # Assume all dataclass field names exist as string keys
                            # in our HDF5 file
                            field.name: trajectory[field.name]
                            for field in dataclasses.fields(ToyDatasetStruct)
                        }
                    )
                )

    # Print some data statistics
    for field in ("image", "position", "velocity"):
        values = jaxfg.utils.pytree_stack(*trajectories).__getattribute__(field)
        print(f"({field}) Mean, std dev:", onp.mean(values), onp.std(values))

    return trajectories


class ToyTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, *paths: str):
        self.samples: List[ToyDatasetStruct] = []

        subsequence_length = 5
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


@jax.jit
def collate_fn(batch):
    return jaxfg.utils.pytree_stack(*batch, axis=1)  # (T, N, ...)
