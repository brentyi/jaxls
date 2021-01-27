import dataclasses
from typing import Tuple

import datargs
import fannypack
import jax
import numpy as onp
import torch
from jax import numpy as jnp
from tqdm.auto import tqdm

import data
import jaxfg
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
uncertainty_model, uncertainty_optimizer = networks.make_uncertainty_mlp()
trainer = Trainer(experiment_name=args.experiment_name)
uncertainty_optimizer = trainer.load_checkpoint(uncertainty_optimizer)

# Set up tensorboard
summary_writer = torch.utils.tensorboard.SummaryWriter(
    log_dir=f"logs/{args.experiment_name}"
)

# Load up position CNN model
position_model, position_optimizer = networks.make_position_cnn()
position_optimizer = Trainer(experiment_name="overnight").load_checkpoint(
    position_optimizer
)


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
    collate_fn=data.collate_fn,
)


def compute_ekf_mse(
    uncertainty_model_params: float,
    trajectory: data.ToyDatasetStructNormalized,
):

    assert len(trajectory.image.shape) == 4  # (timestep, height, width, channels)
    subsequence_length = trajectory.image.shape[0]

    trajectory_unnormalized = trajectory.unnormalize()
    predicted_positions = predict_positions(trajectory.image)

    # Neural network
    uncertainty_factor = uncertainty_model.apply(
        uncertainty_model_params,
        trajectory.visible_pixels_count.reshape((-1, 1)),
    )
    assert uncertainty_model.shape == (subsequence_length,)

    # Run EKF
    belief = toy_ekf.Belief(
        mean=toy_system.State.make(
            position=trajectory_unnormalized.position[0],
            velocity=trajectory_unnormalized.velocity[0],
        ).params,
        cov=jnp.zeros((4, 4)),
    )
    # beliefs = [belief]
    positions = [belief.mean[:2]]

    for t in range(1, subsequence_length):
        belief = toy_ekf.predict_step(belief)
        belief = toy_ekf.update_step(
            belief,
            observation=predicted_positions[i, :],
            observation_cov=uncertainty_factor[i] * jnp.eye(2),
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
    uncertainty_model_params: float,
    batched_trajectory: data.ToyDatasetStructNormalized,
):
    return jnp.mean(
        jax.vmap(compute_ekf_mse, in_axes=(None, 0))(
            uncertainty_model_params, batched_trajectory
        )
    )


@jax.jit
def get_stats(minibatch: data.ToyDatasetStructNormalized):
    factors = uncertainty_model.apply(
        uncertainty_optimizer.target,
        minibatch.visible_pixels_count.reshape((-1, 1)),
    )
    return jnp.mean(factors), jnp.std(factors)


loss_grad_fn = jax.jit(jax.value_and_grad(mse_loss, argnums=0))

num_epochs = 30
progress = tqdm(range(num_epochs))
losses = []
for epoch in progress:
    minibatch: data.ToyDatasetStructNormalized
    for i, minibatch in enumerate(dataloader):
        loss_value, grad = loss_grad_fn(uncertainty_optimizer.target, minibatch)
        uncertainty_optimizer = uncertainty_optimizer.apply_gradient(
            grad, learning_rate=1e-4
        )
        losses.append(loss_value)

        if (
            uncertainty_optimizer.state.step < 10
            or uncertainty_optimizer.state.step % 10 == 0
        ):
            mean, std_dev = get_stats(minibatch)
            # Log to Tensorboard
            summary_writer.add_scalar(
                "train/loss",
                float(loss_value),
                global_step=uncertainty_optimizer.state.step,
            )
            summary_writer.add_scalar(
                "train/mean",
                float(mean),
                global_step=uncertainty_optimizer.state.step,
            )
            summary_writer.add_scalar(
                "train/std_dev",
                float(std_dev),
                global_step=uncertainty_optimizer.state.step,
            )
            print(f"Loss: {loss_value}")

        if uncertainty_optimizer.state.step % 100 == 0:
            trainer.metadata["loss"] = float(loss_value)
            trainer.save_checkpoint(uncertainty_optimizer)
