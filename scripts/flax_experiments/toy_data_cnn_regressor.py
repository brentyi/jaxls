import dataclasses
from typing import List

import fannypack
import flax
import jax
import numpy as onp
import torch
from flax import linen as nn
from jax import numpy as jnp
from tqdm.auto import tqdm

import jaxfg

# TODO:
# [x] Dataloader for single examples
# [ ] Dataloader for subsequences
# [x] CNN definition
# [x] CNN


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class ToyDatasetStruct:
    image: jnp.ndarray
    visible_pixels_count: jnp.ndarray
    position: jnp.ndarray
    velocity: jnp.ndarray


def load_trajectories(*paths: str) -> List[ToyDatasetStruct]:
    trajectories = []
    for path in paths:
        with fannypack.data.TrajectoriesFile(path) as traj_file:
            for trajectory in traj_file:
                timestamps = len(trajectory["image"])

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


class ToyDataset(torch.utils.data.Dataset):
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


class SimpleMLP(nn.Module):
    units: int
    layers: int
    output_dim: int

    @staticmethod
    def make(units: int, layers: int, output_dim: int):
        """Dummy constructor for type-checking."""
        return SimpleMLP(units=units, layers=layers, output_dim=output_dim)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        x = inputs

        for i in range(self.layers):
            x = nn.Dense(self.units)(x)
            x = nn.relu(x)

        x = nn.Dense(self.output_dim)(x)
        return x


class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        x = inputs
        N = x.shape[0]
        assert x.shape == (N, 120, 120, 3)

        # Some conv layers
        for _ in range(3):
            x = nn.Conv(features=32, kernel_size=(3, 3))(x)
            x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)

        # Channel-wise max pool
        x = jnp.max(x, axis=3, keepdims=True)

        # Spanning mean pools (to regularize X/Y coordinate regression)
        assert x.shape == (N, 120, 120, 1)
        x_horizontal = nn.avg_pool(x, window_shape=(120, 1))
        x_vertical = nn.avg_pool(x, window_shape=(1, 120))

        # Concatenate, feed through MLP
        x = jnp.concatenate(
            [x_horizontal.reshape((N, -1)), x_vertical.reshape((N, -1))], axis=1
        )
        assert x.shape == (N, 240)
        x = SimpleMLP.make(units=32, layers=3, output_dim=2)(x)

        return x


@jax.jit
def collate_fn(batch):
    return jaxfg.utils.pytree_stack(*batch, axis=0)


dataloader = torch.utils.data.DataLoader(
    ToyDataset(
        "data/toy_0.hdf5",
    ),
    batch_size=32,
    collate_fn=collate_fn,
)

# Create our network
model = SimpleCNN()
prng_key = jax.random.PRNGKey(0)

# Create our optimizer
dummy_image = jnp.zeros((1, 120, 120, 3))
optimizer = flax.optim.Adam(learning_rate=1e-4).create(
    target=model.init(prng_key, dummy_image)  # Initial MLP parameters
)

# Define loss, gradients, etc
@jax.jit
def mse_loss(
    model_params: jaxfg.types.PyTree,
    batched_images: jnp.ndarray,
    batched_positions: jnp.ndarray,
):
    pred_positions = model.apply(model_params, batched_images)
    assert pred_positions.shape == batched_positions.shape
    return jnp.mean(
        (pred_positions - batched_positions) ** 2
    )


@jax.jit
def get_standard_deviations(minibatch):
    return jnp.std(model.apply(optimizer.target, minibatch.image), axis=0), jnp.std(
        minibatch.position, axis=0
    )


loss_grad_fn = jax.jit(jax.value_and_grad(mse_loss, argnums=0))


num_epochs = 5000
progress = tqdm(range(num_epochs))
losses = []
for epoch in progress:
    minibatch: ToyDatasetStruct
    for i, minibatch in enumerate(dataloader):
        loss_value, grad = loss_grad_fn(
            optimizer.target,
            minibatch.image,
            minibatch.position,
        )
        optimizer = optimizer.apply_gradient(grad)
        losses.append(loss_value)

        if i % 10 == 0:
            print(
                "Standard deviations:",
                get_standard_deviations(minibatch),
            )
            progress.set_description(f"Current loss: {loss_value:10.6f}")
