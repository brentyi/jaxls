import dataclasses
from typing import List, Tuple

import fannypack
import flax
import jax
import jaxlie
import numpy as onp
from jax import numpy as jnp
from tqdm.auto import tqdm

import data
import jaxfg
import kitti_system
import networks
import trainer

fannypack.utils.pdb_safety_net()


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class LearnableUncertainties:
    observation_diag: jnp.ndarray
    dynamics_diag: jnp.ndarray


def predict_velocities(
    observation_model: networks.SimpleCNN,
    observation_model_params: jaxfg.types.PyTree,
    stacked_images: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    velocities = observation_model.apply(
        observation_model_params,
        stacked_images,
    )[:, :2]

    unnorm_velocities = data.KittiStructNormalized(
        linear_vel=velocities[:, 0],
        angular_vel=velocities[:, 1],
    ).unnormalize()

    return unnorm_velocities.linear_vel, unnorm_velocities.angular_vel


def make_factor_graph_template(sequence_length: int) -> jaxfg.core.PreparedFactorGraph:
    variables: List[kitti_system.StateVariable] = []
    factors: List[jaxfg.core.FactorBase] = []
    for t in range(sequence_length):
        variables.append(kitti_system.StateVariable())

        if t == 0:
            # Add prior constraint
            factors.append(
                kitti_system.PriorFactor.make(
                    variable=variables[-1],
                    mu=kitti_system.StateVariable.get_default_value(),  # To be populated
                    scale_tril_inv=jnp.eye(5)
                    * 1000.0,  # Initialize with ground-truth => very high!
                )
            )

        # Add perception constraint
        factors.append(
            kitti_system.VisionFactor.make(
                state_variable=variables[-1],
                predicted_velocity=onp.zeros(2),  # To be populated
                scale_tril_inv=onp.identity(2),  # To be populated
            )
        )

        # Add dynamics constraint
        if t != 0:
            factors.append(
                kitti_system.DynamicsFactor.make(
                    before_variable=variables[-2],
                    after_variable=variables[-1],
                    scale_tril_inv=jnp.identity(5),  # To be populated
                )
            )

    return jaxfg.core.PreparedFactorGraph.from_factors(factors)


def update_graph(
    graph_template: jaxfg.core.PreparedFactorGraph,
    trajectory: data.KittiStructRaw,
    linear_vel: jnp.ndarray,
    angular_vel: jnp.ndarray,
    uncertainties: LearnableUncertainties,
) -> jaxfg.core.PreparedFactorGraph:

    prior_factors, vision_factors, dynamics_factors = graph_template.stacked_factors
    assert isinstance(prior_factors, kitti_system.PriorFactor)
    assert isinstance(vision_factors, kitti_system.VisionFactor)
    assert isinstance(dynamics_factors, kitti_system.DynamicsFactor)

    N = len(trajectory.image)
    assert trajectory.x.shape == (N,)
    assert trajectory.y.shape == (N,)
    assert trajectory.theta.shape == (N,)

    # Anchor first timestep
    # Note that we need to add a batch dimension
    prior_factors = dataclasses.replace(
        prior_factors,
        mu=jax.tree_map(
            lambda x: x[None, ...],
            kitti_system.State.make(
                pose=jaxlie.SE2.from_xy_theta(
                    x=trajectory.x[0], y=trajectory.y[0], theta=trajectory.theta[0]
                ),
                linear_vel=trajectory.linear_vel[0],
                angular_vel=trajectory.angular_vel[0],
            ),
        ),
    )

    # # Anchor velocities
    # vision_factors = dataclasses.replace(
    #     vision_factors,
    #     predicted_velocity=jnp.stack(
    #         [trajectory.linear_vel, trajectory.angular_vel], axis=-1
    #     ),
    # )

    # Anchor velocities
    assert vision_factors.scale_tril_inv.shape == (N, 2, 2)
    vision_factors = dataclasses.replace(
        vision_factors,
        predicted_velocity=jnp.stack([linear_vel, angular_vel], axis=-1),
        scale_tril_inv=jnp.tile(
            jnp.diag(uncertainties.observation_diag)[None, :, :], reps=(N, 1, 1)
        ),
    )

    # Update dynamics uncertainties
    assert dynamics_factors.scale_tril_inv.shape == (N - 1, 5, 5)
    dynamics_factors = dataclasses.replace(
        dynamics_factors,
        scale_tril_inv=jnp.tile(
            jnp.diag(uncertainties.dynamics_diag)[None, :, :], reps=(N - 1, 1, 1)
        ),
    )

    # Return updated graph
    return dataclasses.replace(
        graph_template,
        stacked_factors=(prior_factors, vision_factors, dynamics_factors),
    )


# Learnable uncertainties
uncertainty_optimizer = flax.optim.Adam().create(
    LearnableUncertainties(observation_diag=jnp.ones(2), dynamics_diag=jnp.ones(5))
)

# Create graph
graph_template = make_factor_graph_template(sequence_length=20)

# Create position CNN
observation_model, observation_optimizer = networks.make_observation_cnn(
    seed=1, train_mode=False
)
observation_optimizer = trainer.Trainer(experiment_name="kitti-cnn").load_checkpoint(
    observation_optimizer
)

trajectories = data.load_trajectories(train=False, subsequence_length=20)


@jax.jit
def mse_loss(trajectory: data.KittiStructNormalized, uncertainties: LearnableUncertainties):

    trajectory_unnorm = trajectory.unnormalize()

    linear_vel_pred, angular_vel_pred = predict_velocities(
        observation_model=observation_model,
        observation_model_params=observation_optimizer.target,
        stacked_images=trajectory.get_stacked_image(),
    )
    graph = update_graph(
        graph_template=graph_template,
        trajectory=trajectory_unnorm,
        linear_vel=linear_vel_pred,
        angular_vel=angular_vel_pred,
        uncertainties=uncertainties,
    )
    initial_assignments = jaxfg.core.VariableAssignments.create_default(
        graph_template.variables
    )
    solved_assignments = graph.solve(
        initial_assignments,
        solver=jaxfg.solvers.FixedIterationGaussNewtonSolver(max_iterations=4),
    )
    predicted_x = solved_assignments.storage.reshape((-1, 6))[:, 0]
    predicted_y = solved_assignments.storage.reshape((-1, 6))[:, 1]

    return jnp.mean(
        (predicted_x - trajectory_unnorm.x) ** 2
        + (predicted_y - trajectory_unnorm.y) ** 2
    )


loss = jnp.mean(jnp.array([mse_loss(traj, uncertainty_optimizer.target) for traj in tqdm(trajectories)]))
print(loss)
# print(jnp.mean(jax.vmap(mse_loss)(jaxfg.utils.pytree_stack(*trajectories, axis=0))))
assert False
