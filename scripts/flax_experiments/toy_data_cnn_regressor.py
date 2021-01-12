import dataclasses

import datargs
import fannypack
import flax
import jax
import jaxfg
import torch
from jax import numpy as jnp
from tqdm.auto import tqdm

from data import ToyDatasetStruct, ToySingleStepDataset, collate_fn
from networks import SimpleCNN
from trainer import Trainer

# Setup configuration
fannypack.utils.pdb_safety_net()

# Define and parse arguments
@dataclasses.dataclass
class Args:
    experiment_name: str = datargs.arg(help="Experiment name.")


args: Args = datargs.parse(Args)


# Prep for training
dataloader = torch.utils.data.DataLoader(
    ToySingleStepDataset(
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

trainer = Trainer(experiment_name=args.experiment_name)

optimizer = flax.optim.Adam(learning_rate=1e-3).create(
    target=model.init(prng_key, dummy_image),  # Initial MLP parameters
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
    return jnp.mean((pred_positions - batched_positions) ** 2)


@jax.jit
def get_standard_deviations(minibatch):
    return (
        jnp.std(model.apply(optimizer.target, minibatch.image), axis=0),
        jnp.std(minibatch.position, axis=0),
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

        if optimizer.state.step < 10 or optimizer.state.step % 100 == 0:
            print(
                f"{optimizer.state.step} Standard deviations:",
                get_standard_deviations(minibatch),
            )
            progress.set_description(f"Current loss: {loss_value:10.6f}")

            trainer.metadata["loss"] = float(loss_value)
            trainer.save_checkpoint(optimizer)

trainer.save_checkpoint(optimizer)
