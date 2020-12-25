import dataclasses

import jax
import numpy as onp
from flax import optim
from jax import numpy as jnp
from jaxfg import core, geometry, solvers, utils
from jaxlie import SE2
from matplotlib import pyplot as plt

from _fixed_iteration_gn import FixedIterationGaussNewtonSolver

# Compute ground-truth values for three poses: A, B, and C
T_ab = SE2.from_xy_theta(1.0, 0.0, 0.2)
T_bc = SE2.from_xy_theta(1.0, 0.0, 0.2)

T_wa = SE2.identity()
T_wb = T_wa @ T_ab
T_wc = T_wb @ T_bc

# Create variables for each pose: these represent nodes in our factor graph
variable_A = geometry.SE2Variable()
variable_B = geometry.SE2Variable()
variable_C = geometry.SE2Variable()

# Put ground-truth poses in storage object
assignments_ground_truth = core.VariableAssignments.from_dict(
    {
        variable_A: T_wa,
        variable_B: T_wb,
        variable_C: T_wc,
    }
)

# Sample some noise to learn to ignore
def sample_se2_noise(std_dev=1.0):
    return SE2.exp(jnp.array(onp.random.randn(SE2.tangent_dim) * std_dev))


noise1 = sample_se2_noise()
noise2 = sample_se2_noise()

# Graph construction helper
def get_graph(error_scale) -> core.PreparedFactorGraph:
    return core.PreparedFactorGraph.from_factors(
        [
            geometry.PriorFactor.make(
                variable=variable_A, mu=SE2.identity(), scale_tril_inv=jnp.eye(3)
            ),
            geometry.BetweenFactor.make(
                before=variable_A,
                after=variable_B,
                delta=T_ab,
                scale_tril_inv=jnp.eye(3),
            ),
            geometry.BetweenFactor.make(
                before=variable_B,
                after=variable_C,
                delta=T_bc,
                scale_tril_inv=jnp.eye(3),
            ),
            # Noisy factors
            geometry.PriorFactor.make(
                variable=variable_B,
                mu=T_wb @ noise1,
                scale_tril_inv=jnp.eye(3) * error_scale,
            ),
            geometry.PriorFactor.make(
                variable=variable_C,
                mu=T_wc @ noise2,
                scale_tril_inv=jnp.eye(3) * error_scale,
            ),
        ]
    )


# Plotting helper
def plot_poses(label: str, *poses: SE2, color: str):
    for i, pose in enumerate(poses):
        plt.arrow(
            *pose.translation,
            *(pose.rotation.unit_complex * 0.05),
            width=0.01,
            color=color,
            label=label if i == 0 else None,
        )


def plot_assignments(label: str, color: str, assignments: core.VariableAssignments):
    plot_poses(
        label,
        assignments.get_value(variable_A),
        assignments.get_value(variable_B),
        assignments.get_value(variable_C),
        color=color,
    )


# Noisy assignments for initializing nonlinear solvers
assignments_initializer = core.VariableAssignments.from_dict(
    {
        variable_A: T_wa @ sample_se2_noise(0.1),
        variable_B: T_wb @ sample_se2_noise(0.1),
        variable_C: T_wc @ sample_se2_noise(0.1),
    }
)

# Loss computation helper
# Should be zero when `error_scale` is zero
graph = get_graph(1.0)


def compute_loss(params):
    error_scale = params["error_scale"]

    # Incorporate error scale to noisy factors
    stacked_factors = []
    for f in graph.stacked_factors:
        if isinstance(f, geometry.PriorFactor):
            f = dataclasses.replace(
                f,
                scale_tril_inv=f.scale_tril_inv.at[1:, :, :].set(
                    f.scale_tril_inv[1:, :, :] * error_scale
                ),
            )
        stacked_factors.append(f)
    graph_scaled = dataclasses.replace(graph, stacked_factors=stacked_factors)

    # Find optimal poses given our factor graph
    assignments_solved = graph_scaled.solve(
        assignments_ground_truth,
        solver=FixedIterationGaussNewtonSolver(max_iters=5),
    )

    # Loss is difference between ground-truth and solved poses
    def subtract(a: SE2, b: SE2) -> jnp.ndarray:
        return (a @ b.inverse()).log()

    error = jax.vmap(subtract)(
        assignments_solved.get_stacked_value(geometry.SE2Variable),
        assignments_ground_truth.get_stacked_value(geometry.SE2Variable),
    )
    sse = jnp.sum(error ** 2)
    return sse

# Create initial parameters to optimize
params = {"error_scale": 5.0}

# Create ADAM optimizer
optimizer_def = optim.Adam(learning_rate=1e-2)
optimizer = optimizer_def.create(params)
loss_grad_fn = jax.value_and_grad(compute_loss)

# Run optimizer
with utils.stopwatch("Compute loss"):
    loss_value = compute_loss(params)
with utils.stopwatch("Compute loss"):
    loss_value = compute_loss(params)
for i in range(500):  # This is far more steps than necessary
    with utils.stopwatch(f"Loss step {i}: {loss_value}"):
        loss_value, grad = loss_grad_fn(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)

# Print optimized value
print(optimizer.target)

plot_assignments(label="Ground-truth", color="r", assignments=assignments_ground_truth)
plot_assignments(
    label="Noisy factors",
    color="g",
    assignments=get_graph(params["error_scale"]).solve(assignments_initializer),
)
plot_assignments(
    label="Noisy factors + optimized",
    color="b",
    assignments=get_graph(optimizer.target["error_scale"]).solve(
        assignments_initializer
    ),
)
# plot_poses("Noisy", T_wa, T_wb, T_wc, color="g")
# plot_poses("Optimized", T_wa, T_wb, T_wc, color="b")
plt.legend()
plt.show()
exit()
