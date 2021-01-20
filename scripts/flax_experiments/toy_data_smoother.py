import dataclasses
from typing import cast

import fannypack
import flax
import jax
import jaxfg
import numpy as onp
import torch
from jax import numpy as jnp
from tqdm.auto import tqdm

from data import ToySubsequenceDataset, collate_fn
from networks import SimpleCNN
from toy_system import (
    DummyVelocityFactor,
    DynamicsFactor,
    State,
    StateVariable,
    VisionFactor,
)

fannypack.utils.pdb_safety_net()

# TODO:
# [x] Dataloader for single examples
# [ ] Dataloader for subsequences
# [x] CNN definition
# [x] CNN

# dataloader = torch.utils.data.DataLoader(
#     ToyTrajectoryDataset(
#         "data/toy_0.hdf5",
#     ),
#     batch_size=32,
#     collate_fn=collate_fn,
# )
#
# # Create our network
# model = SimpleCNN()
# prng_key = jax.random.PRNGKey(0)
#
# # Create our optimizer
# dummy_image = jnp.zeros((1, 120, 120, 3))
# optimizer = flax.optim.Adam(learning_rate=1e-4).create(
#     target=model.init(prng_key, dummy_image)  # Initial MLP parameters
# )
#


def make_factor_graph() -> jaxfg.core.PreparedFactorGraph:
    variables = []
    factors = []
    for t in range(5):
        variables.append(StateVariable())

        # Add perception constraint
        factors.append(
            VisionFactor.make(
                state_variable=variables[-1],
                position=onp.zeros(2) + 0.8,  # To be populated by network
                scale_tril_inv=onp.identity(2),  # To be populated by network
            )
        )
        # factors.append(
        #     DummyVelocityFactor.make(
        #         state_variable=variables[-1],
        #     )
        # )

        # Add dynamics constraint
        if t != 0:
            factors.append(
                DynamicsFactor.make(
                    before_variable=variables[-2],
                    after_variable=variables[-1],
                )
            )

    return jaxfg.core.PreparedFactorGraph.from_factors(factors)


factor_graph = make_factor_graph()
# initial_assignments = jaxfg.core.VariableAssignments(
#     storage=onp.zeros(StateVariable.get_local_parameter_dim() * 5),
#     storage_metadata=jaxfg.core.StorageMetadata.from_variables(factor_graph.variables),
# )
initial_assignments = jaxfg.core.VariableAssignments.create_default(
    factor_graph.variables
)
# print(factor_graph.compute_cost(initial_assignments))

solved_assignments = factor_graph.solve(initial_assignments)


# @jax.jit
# def mse_loss(
#     model_params: jaxfg.types.PyTree,
#     batched_images: jnp.ndarray,
#     batched_positions: jnp.ndarray,
# ):
#     N = batched_images.shape[1]
#     assert batched_images.shape == (5, N, 120, 120, 3), batched_images.shape
#
#     pred_positions: jnp.ndarray = model.apply(
#         model_params, batched_images.reshape((-1, 120, 120, 3))
#     ).reshape(5, N, 2)
#
#     def solve_one(pred_positions, label_positions):
#         stacked_factor = cast(jaxfg.core.LinearFactor, factor_graph.stacked_factors[0])
#         assert (
#             stacked_factor.b.shape == pred_positions.shape
#         ), f"{stacked_factor.b.shape} {pred_positions.shape}"
#         new_stacked_factor = dataclasses.replace(stacked_factor, b=pred_positions)
#
#         solved_assignments = dataclasses.replace(
#             factor_graph, stacked_factors=[new_stacked_factor]
#         ).solve(
#             dataclasses.replace(
#                 initial_assignments, storage=label_positions.reshape((-1,))
#             ),
#             solver=jaxfg.solvers.FixedIterationGaussNewtonSolver(max_iterations=3),
#         )
#
#         return solved_assignments.storage.reshape((5, 2))
#
#     # Solve for each minibatch; shapes are (T, N, ...)
#     solved_positions = jax.vmap(solve_one, in_axes=1, out_axes=1)(
#         pred_positions, batched_positions
#     )
#     assert solved_positions.shape == batched_positions.shape
#
#     return jnp.mean((solved_positions - batched_positions) ** 2)
#
#
# @jax.jit
# def get_standard_deviations(minibatch):
#     return (
#         jnp.std(
#             model.apply(optimizer.target, minibatch.image.reshape((-1, 120, 120, 3))),
#             axis=0,
#         ),
#         jnp.std(minibatch.position, axis=0).reshape((-1, 2)),
#     )
#
#
# loss_grad_fn = jax.jit(jax.value_and_grad(mse_loss, argnums=0))
#
#
# num_epochs = 5000
# progress = tqdm(range(num_epochs))
# losses = []
# for epoch in progress:
#     minibatch: ToyDatasetStruct
#     for i, minibatch in enumerate(dataloader):
#         loss_value, grad = loss_grad_fn(
#             optimizer.target,
#             minibatch.image,
#             minibatch.position,
#         )
#         optimizer = optimizer.apply_gradient(grad)
#         losses.append(loss_value)
#
#         if i % 10 == 0:
#             print(
#                 "Standard deviations:",
#                 get_standard_deviations(minibatch),
#             )
#             progress.set_description(f"Current loss: {loss_value:10.6f}")
