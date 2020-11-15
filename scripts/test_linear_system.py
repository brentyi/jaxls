from typing import Tuple

import linear_system_models
import torch
import torchfilter

import torchfg


# Generate data
def generate_data() -> Tuple[
    torchfilter.types.StatesTorch,
    torchfilter.types.ObservationsNoDictTorch,
    torchfilter.types.ControlsNoDictTorch,
]:
    """Generate `N` (noisy) trajectories using our dynamics and observation models.

    Returns:
        tuple: (states, observations, controls). First dimension of all tensors should
            be `N`.
    """
    torch.random.manual_seed(0)

    # Batch size
    N = 5

    # Timesteps
    T = 100

    # Dimensions
    state_dim = linear_system_models.state_dim
    control_dim = linear_system_models.control_dim
    observation_dim = linear_system_models.observation_dim

    # Models
    dynamics_model = linear_system_models.LinearDynamicsModel()
    measurement_model = linear_system_models.LinearKalmanFilterMeasurementModel()

    # Initialize empty states, observations
    states = torch.zeros((T, N, state_dim))
    observations = torch.zeros((T, N, observation_dim))

    # Generate random control inputs
    controls = torch.randn(size=(T, N, control_dim))

    for t in range(T):
        if t == 0:
            # Initialize random initial state
            states[0, :, :] = torch.randn(size=(N, state_dim))
        else:
            # Update state and add noise
            pred_states, Q_tril = dynamics_model(
                initial_states=states[t - 1, :, :], controls=controls[t, :, :]
            )
            assert pred_states.shape == (N, state_dim)
            assert Q_tril.shape == (N, state_dim, state_dim)

            states[t, :, :] = pred_states + (
                Q_tril @ torch.randn(size=(N, state_dim, 1))
            ).squeeze(-1)

        # Compute observations and add noise
        pred_observations, R_tril = measurement_model(states=states[t, :, :])
        observations[t, :, :] = pred_observations + (
            R_tril @ torch.randn(size=(N, observation_dim, 1))
        ).squeeze(-1)

    return states, observations, controls


############################
# Generate data
############################
states, observations, controls = generate_data()
T, N, state_dim = states.shape

# Generate initial belief from trajectory
initial_mean = states[0, ...]
initial_covariance = (
    torch.zeros(size=(N, state_dim, state_dim)) + torch.eye(state_dim)[None, :, :] * 0.1
)

# Remove first timestep
states = states[1:, ...]
observations = observations[1:, ...]
controls = controls[1:, ...]
T -= 1

############################
# Run on SR-UKF
############################
ukf = torchfilter.filters.SquareRootUnscentedKalmanFilter(
    dynamics_model=linear_system_models.LinearDynamicsModel(),
    measurement_model=linear_system_models.LinearKalmanFilterMeasurementModel(),
)
ukf.initialize_beliefs(mean=initial_mean, covariance=initial_covariance)
ukf_estimated_states = ukf.forward_loop(observations=observations, controls=controls)
assert (
    ukf_estimated_states.shape == states.shape
), "UKF state shape does not match ground-truth"

############################
# Run with factor graph
############################
graph = torchfg.FactorGraph()
dynamics_model = linear_system_models.LinearDynamicsModel()
measurement_model = linear_system_models.LinearKalmanFilterMeasurementModel()


def dynamics_model_forward(x, controls):
    return dynamics_model(initial_states=x, controls=controls)


def measurement_model_forward(x):
    return measurement_model(states=x)


# Create initial state node
state_nodes = [torchfg.RealVectorVariable(states[0].detach().clone())]

# Add prior
graph.add_factors(
    torchfg.PriorFactor(
        state_nodes[0],
        prior_mean=initial_mean,
        prior_scale_tril=torch.cholesky(initial_covariance),
    )
)

# Add factors for each timestep
for t in range(T):
    # Add new state
    state_nodes.append(
        torchfg.RealVectorVariable(ukf_estimated_states[t].detach().clone())
    )

    # Add factors
    graph.add_factors(
        # Dynamics
        torchfg.TransitionFactor(
            before_node=state_nodes[-2],
            after_node=state_nodes[-1],
            transition_fn=dynamics_model_forward,
            transition_fn_kwargs={"controls": controls[t]},
        ),
        # Observations
        torchfg.ObservationFactor(
            node=state_nodes[-1],
            observation=observations[t],
            observation_fn=measurement_model_forward,
        ),
    )

# Check log-likelihood of UKF estimates
print("UKF log-likelihood: \t", torch.mean(graph.evaluate_log_likelihood()))

# Check log-likelihood w/ ground-truth
for t in range(T):
    state_nodes[t + 1].value = states[t]
print("Ground-truth log-likelihood: \t", torch.mean(graph.evaluate_log_likelihood()))

# Check log-likelihood w/ dead-reckoning
for t in range(T):
    next_state, _ = dynamics_model_forward(state_nodes[t].value, controls=controls[t])
    assert next_state.shape == state_nodes[t + 1].value.shape
    state_nodes[t + 1].value = next_state.detach().clone()
print("Dead-reckoning log-likelihood: \t", torch.mean(graph.evaluate_log_likelihood()))

# Solve MAP inference problem for state nodes
graph.solve_map_inference(state_nodes)
print("MAP log-likelihood: \t", torch.mean(graph.evaluate_log_likelihood()))
