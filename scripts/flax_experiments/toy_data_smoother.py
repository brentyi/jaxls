import dataclasses
from typing import List, cast

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
        assert x.shape == (N, 120, 120, 3), x.shape

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
    return jaxfg.utils.pytree_stack(*batch, axis=1)  # (T, N, ...)


dataloader = torch.utils.data.DataLoader(
    ToyTrajectoryDataset(
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


def make_factor_graph() -> jaxfg.core.PreparedFactorGraph:
    variables = []
    factors = []
    for t in range(5):
        variables.append(jaxfg.core.RealVectorVariable[2]())
        factors.append(
            jaxfg.core.LinearFactor(
                variables=(variables[-1],),
                A_matrices=(onp.eye(2),),
                b=onp.zeros(2),
                scale_tril_inv=onp.eye(2),
            )
        )
    return jaxfg.core.PreparedFactorGraph.from_factors(factors)


factor_graph = make_factor_graph()
initial_assignments = jaxfg.core.VariableAssignments(
    storage=onp.zeros(10),
    storage_metadata=jaxfg.core.StorageMetadata.from_variables(factor_graph.variables),
)

# Define loss, gradients, etc
@jax.jit
def mse_loss(
    model_params: jaxfg.types.PyTree,
    batched_images: jnp.ndarray,
    batched_positions: jnp.ndarray,
):
    N = batched_images.shape[1]
    assert batched_images.shape == (5, N, 120, 120, 3), batched_images.shape

    pred_positions: jnp.ndarray = model.apply(
        model_params, batched_images.reshape((-1, 120, 120, 3))
    ).reshape(5, N, 2)

    def solve_one(pred_positions, label_positions):
        stacked_factor = cast(jaxfg.core.LinearFactor, factor_graph.stacked_factors[0])
        assert (
            stacked_factor.b.shape == pred_positions.shape
        ), f"{stacked_factor.b.shape} {pred_positions.shape}"
        new_stacked_factor = dataclasses.replace(stacked_factor, b=pred_positions)

        solved_assignments = dataclasses.replace(
            factor_graph, stacked_factors=[new_stacked_factor]
        ).solve(
            dataclasses.replace(initial_assignments, storage=label_positions.reshape((-1,))),
            solver=jaxfg.solvers.FixedIterationGaussNewtonSolver(max_iterations=3),
        )

        return solved_assignments.storage.reshape((5, 2))

    # Solve for each minibatch; shapes are (T, N, ...)
    solved_positions = jax.vmap(solve_one, in_axes=1, out_axes=1)(pred_positions, batched_positions)
    assert solved_positions.shape == batched_positions.shape

    return jnp.mean((solved_positions - batched_positions) ** 2)


@jax.jit
def get_standard_deviations(minibatch):
    return (
        jnp.std(
            model.apply(optimizer.target, minibatch.image.reshape((-1, 120, 120, 3))),
            axis=0,
        ),
        jnp.std(minibatch.position, axis=0).reshape((-1, 2)),
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
