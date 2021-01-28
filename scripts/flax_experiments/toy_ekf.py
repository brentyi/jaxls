import dataclasses

import fannypack
import jax
import numpy as onp
from jax import numpy as jnp

import data
import jaxfg
import networks
import toy_system
import trainer


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class Belief:
    mean: jnp.ndarray
    cov: jnp.ndarray


def predict_step(belief: Belief) -> Belief:
    """EKF predict step."""

    belief_pred = toy_system.dynamics_forward(
        toy_system.State(params=belief.mean)
    ).params

    # Get Jacobian of dynamics: output state parameters wrt input state parameters
    # (note that our dynamics are actually linear here)
    A: onp.ndarray = jax.jacfwd(toy_system.dynamics_forward)(
        toy_system.State(params=belief.mean)
    ).params.params
    assert A.shape == (4, 4)

    return Belief(
        mean=belief_pred,
        cov=A @ belief.cov @ A.T + toy_system.DYNAMICS_COVARIANCE,
    )


def update_step(belief: Belief, observation: jnp.ndarray, observation_cov: jnp.ndarray):
    """EKF update step."""

    # Note that we have a linear observation model
    C = jnp.zeros((2, 4)).at[:2, :2].set(jnp.eye(2))

    pred_observation = C @ belief.mean
    assert pred_observation.shape == observation.shape
    innovation = observation - pred_observation
    innovation_cov = C @ belief.cov @ C.T + observation_cov

    K = belief.cov @ C.T @ jnp.linalg.inv(innovation_cov)

    state_dim: int = 4
    belief_updated = Belief(
        mean=belief.mean + K @ innovation,
        cov=(jnp.eye(state_dim) - K @ C) @ belief.cov,
    )
    assert belief_updated.mean.shape == belief.mean.shape
    assert belief_updated.cov.shape == belief.cov.shape

    return belief_updated


__all__ = ["Belief", "predict_step", "update_step"]

if __name__ == "__main__":
    fannypack.utils.pdb_safety_net()

    eval_trajectories = data.load_trajectories(train=False)

    # Make model that we're going to be optimizing
    uncertainty_model, uncertainty_optimizer = networks.make_uncertainty_mlp()
    uncertainty_optimizer = trainer.Trainer(
        experiment_name="uncertainty-ekf"
    ).load_checkpoint(uncertainty_optimizer)

    # Load up position CNN model
    position_model, position_optimizer = networks.make_position_cnn()
    position_optimizer = trainer.Trainer(experiment_name="position-cnn").load_checkpoint(
        position_optimizer
    )

    trajectory_count = len(eval_trajectories)
    subsequence_length = eval_trajectories[0].image.shape[0]
    stacked_trajectories: data.ToyDatasetStructNormalized = jaxfg.utils.pytree_stack(
        *eval_trajectories
    )

    # Get observations for EKF
    observations = (
        data.ToyDatasetStructNormalized(
            position=position_model.apply(
                position_optimizer.target,
                stacked_trajectories.image.reshape((-1, 120, 120, 3)),
            ).reshape((trajectory_count, subsequence_length, 2))
        )
        .unnormalize()
        .position
    )

    observation_covs = (
        # Variable uncertainty
        onp.eye(2)[onp.newaxis, onp.newaxis, :, :]
        / (
            uncertainty_model.apply(
                uncertainty_optimizer.target,
                stacked_trajectories.visible_pixels_count.reshape((-1, 1)),
            ).reshape((trajectory_count, subsequence_length, 1, 1))
            ** 2
        )

        # Trained w/ EKF
        # 1.0 / (0.14734569191932678 ** 2)

        # Trained w/ differentiable filter attempt #2 (bigger batch size)
        # 1.0 / (0.08585679531097412 ** 2)
        * onp.ones((trajectory_count, subsequence_length, 1, 1))
    )

    # Run Kalman filter
    belief = Belief(
        mean=toy_system.State.make(
            position=stacked_trajectories.unnormalize().position[:, 0, :],
            velocity=stacked_trajectories.unnormalize().velocity[:, 0, :],
        ).params,
        cov=onp.zeros((trajectory_count, 4, 4)),
    )
    # positions_predicted_list = []
    # positions_predicted_list.append(toy_system.State(params=belief.mean).position)
    beliefs = [belief]

    predict_step_batch = jax.jit(jax.vmap(predict_step))
    update_step_batch = jax.jit(jax.vmap(update_step))

    for i in range(1, subsequence_length):
        belief = predict_step_batch(belief)
        belief = update_step_batch(
            belief, observations[:, i, :], observation_covs[:, i, :, :]
        )
        beliefs.append(belief)

    positions_predicted = onp.stack([b.mean[..., :2] for b in beliefs], axis=1)
    positions_label = stacked_trajectories.unnormalize().position[
        :, : positions_predicted.shape[1]
    ]
    assert positions_predicted.shape == positions_label.shape == observations.shape
    print(
        "RMSE filter", onp.sqrt(onp.mean((positions_predicted - positions_label) ** 2))
    )
    print("RMSE vision", onp.sqrt(onp.mean((observations - positions_label) ** 2)))
    assert False
    # positions_predicted = onp.array(positions_predicted_list)
