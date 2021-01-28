import dataclasses

import datargs
import fannypack
import flax
import jax
import numpy as onp
import torch
from jax import numpy as jnp
from tqdm.auto import tqdm

import data
import networks
import toy_ekf
import toy_system
from trainer import Trainer

# Setup configuration
fannypack.utils.pdb_safety_net()

# Define and parse arguments
@dataclasses.dataclass
class Args:
    experiment_name: str = datargs.arg(help="Experiment name.")


args: Args = datargs.parse(Args)

# Make model that we're going to be optimizing
uncertainty_optimizer = flax.optim.Adam().create(target=0.1)
# trainer = Trainer(experiment_name=args.experiment_name)

# Load up position CNN model
position_model, position_optimizer = networks.make_position_cnn()
position_optimizer = Trainer(experiment_name="position-cnn").load_checkpoint(
    position_optimizer
)
assert position_optimizer.state.step > 0


def predict_positions(images: jnp.ndarray) -> jnp.ndarray:
    """Predict positions from images.

    Input is normalized image, output is unnormalized position.
    """
    N = images.shape[0]
    assert images.shape == (N, 120, 120, 3)

    return (
        data.ToyDatasetStructNormalized(
            position=jax.jit(position_model.apply)(position_optimizer.target, images),
        )
        .unnormalize()
        .position
    )


# Prep for training
subsequence_length = 20
dataloader = torch.utils.data.DataLoader(
    data.ToySubsequenceDataset(train=True, subsequence_length=subsequence_length),
    batch_size=32,
    drop_last=True,
    collate_fn=data.collate_fn,
)


def compute_ekf_mse(
    uncertainty_factor: float,
    trajectory: data.ToyDatasetStructNormalized,
    seed: int,
):

    assert len(trajectory.image.shape) == 4  # (timestep, height, width, channels)
    subsequence_length = trajectory.image.shape[0]

    trajectory_unnormalized = trajectory.unnormalize()
    predicted_positions = predict_positions(trajectory.image)

    # Run EKF
    belief = toy_ekf.Belief(
        mean=toy_system.State.make(
            position=trajectory_unnormalized.position[0],
            velocity=trajectory_unnormalized.velocity[0],
        ).params
        + jax.random.normal(key=jax.random.PRNGKey(seed), shape=(4,)) * 3.0,
        cov=jnp.eye(4) * 9.0,
    )
    # beliefs = [belief]
    positions = [belief.mean[:2]]

    for t in range(1, subsequence_length):
        belief = toy_ekf.predict_step(belief)
        belief = toy_ekf.update_step(
            belief,
            observation=predicted_positions[t, :],
            observation_cov=jnp.eye(2) / (uncertainty_factor ** 2),
        )
        # beliefs.append(belief)
        positions.append(belief.mean[:2])

    positions_predicted = jnp.array(positions)
    positions_label = trajectory.unnormalize().position

    N = positions_label.shape[0]
    assert positions_label.shape == positions_predicted.shape == (N, 2)

    mse = jnp.mean((positions_predicted - positions_label) ** 2)

    return mse


@jax.jit
def mse_loss(
    uncertainty_factor: float,
    batched_trajectory: data.ToyDatasetStructNormalized,
    seed: int,
):
    return jnp.mean(
        jax.vmap(compute_ekf_mse, in_axes=(None, 0, None))(
            uncertainty_factor, batched_trajectory, seed
        )
    )


loss_grad_fn = jax.jit(jax.value_and_grad(mse_loss, argnums=0))

num_epochs = 20
progress = tqdm(range(num_epochs))
losses = []
for epoch in progress:
    minibatch: data.ToyDatasetStructNormalized
    for i, minibatch in enumerate(dataloader):
        loss_value, grad = loss_grad_fn(
            uncertainty_optimizer.target,
            minibatch,
            onp.random.randint(0, 100000000),
        )
        uncertainty_optimizer = uncertainty_optimizer.apply_gradient(
            grad, learning_rate=1e-4
        )
        losses.append(loss_value)

        if (
            uncertainty_optimizer.state.step < 10
            or uncertainty_optimizer.state.step % 5 == 0
        ):
            print(f"Loss: {loss_value}, value: {uncertainty_optimizer.target}")
