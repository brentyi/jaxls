import dataclasses
from typing import Tuple

import datargs
import fannypack
import jax
import jaxfg
import numpy as onp
import torch
from jax import numpy as jnp
from tqdm.auto import tqdm

import data
import networks
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
uncertainty_optimizer = Trainer(experiment_name="uncertainty-ekf").load_checkpoint(uncertainty_optimizer)

# Set up tensorboard
summary_writer = torch.utils.tensorboard.SummaryWriter(
    log_dir=f"logs/{args.experiment_name}"
)

# Load up position CNN model
position_model, position_optimizer = networks.make_position_cnn()
position_optimizer = Trainer(experiment_name="position-cnn").load_checkpoint(
    position_optimizer
)


def predict_positions(images: jnp.ndarray):
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
    shuffle=True,
)

# Define loss, gradients, etc.
# Make factor graph
def make_factor_graph(
    trajectory_length: int,
    include_dynamics: bool = True,  # Set to False to disable dynamics for debugging
) -> jaxfg.core.PreparedFactorGraph:
    variables = []
    factors = []
    for t in range(trajectory_length):
        variables.append(toy_system.StateVariable())
        variables[-1]._timestep = t

        # Add perception constraint
        factors.append(
            toy_system.VisionFactor.make(
                state_variable=variables[-1],
                predicted_position=onp.zeros(2) + 0.8,  # To be populated by network
                scale_tril_inv=onp.identity(2),  # To be populated by network
            )
        )

        # Add dynamics constraint
        if not include_dynamics:
            factors.append(toy_system.DummyVelocityFactor.make(variables[-1]))
        elif t != 0:
            factors.append(
                toy_system.DynamicsFactor.make(
                    before_variable=variables[-2],
                    after_variable=variables[-1],
                )
            )

    return jaxfg.core.PreparedFactorGraph.from_factors(factors)


def update_factor_graph(
    graph_template: jaxfg.core.PreparedFactorGraph,
    trajectory: data.ToyDatasetStructNormalized,
    uncertainty_factor: float,
) -> Tuple[jaxfg.core.PreparedFactorGraph, jaxfg.core.VariableAssignments]:
    """Update factor graph, and produce guess of initial assignments."""
    predicted_positions = predict_positions(trajectory.image)

    # Guess initial assignments
    assignments_dict = {}
    velocity_guesses = (
        jnp.roll(predicted_positions, shift=-1, axis=0) - predicted_positions
    )
    velocity_guesses = velocity_guesses.at[-1].set(velocity_guesses[-2])
    for i, variable in enumerate(graph_template.variables):
        assignments_dict[variable] = toy_system.State.make(
            position=predicted_positions[i],
            velocity=velocity_guesses[i],
        )
    initial_assignments = jaxfg.core.VariableAssignments.from_dict(assignments_dict)

    # Populate positions
    stacked_factors = list(graph_template.stacked_factors)
    stacked_vision_factor: toy_system.VisionFactor = stacked_factors[0]
    assert isinstance(stacked_vision_factor, toy_system.VisionFactor)
    assert predicted_positions.shape == stacked_vision_factor.predicted_position.shape

    stacked_factors[0] = dataclasses.replace(
        stacked_vision_factor,
        predicted_position=predicted_positions,
        scale_tril_inv=stacked_vision_factor.scale_tril_inv
        * uncertainty_factor.reshape((-1, 1, 1)),
    )

    # Return new graph with new factors
    return (
        dataclasses.replace(graph_template, stacked_factors=stacked_factors),
        initial_assignments,
    )


graph_template: jaxfg.core.PreparedFactorGraph = make_factor_graph(
    trajectory_length=subsequence_length,
)


def compute_smoother_mse(
    uncertainty_model_params: float,
    trajectory: data.ToyDatasetStructNormalized,
):

    uncertainty_factor = uncertainty_model.apply(
        uncertainty_model_params,
        trajectory.visible_pixels_count.reshape((-1, 1)),
    )

    graph, initial_assignments = update_factor_graph(
        graph_template,
        trajectory=trajectory,
        uncertainty_factor=uncertainty_factor,
    )
    solved_assignments = graph.solve(
        initial_assignments,
        solver=jaxfg.solvers.FixedIterationGaussNewtonSolver(
            max_iterations=5, verbose=False
        ),
    )
    positions_predicted = solved_assignments.storage.reshape((-1, 4))[:, :2]
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
        jax.vmap(compute_smoother_mse, in_axes=(None, 0))(
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

num_epochs = 60
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
