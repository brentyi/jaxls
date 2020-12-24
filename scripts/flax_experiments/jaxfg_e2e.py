import jax
import numpy as onp
from flax import optim
from jax import numpy as jnp
from jaxfg import core, geometry, solvers, utils
from jaxlie import SE2

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

# Loss computation helper
# Should be zero when `error_scale` is zero
def compute_loss(params):
    error_scale = params["error_scale"]
    graph = core.PreparedFactorGraph.from_factors(
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

    # Find optimal poses given our factor graph
    assignments_solved = graph.solve(
        assignments_ground_truth,
        solver=FixedIterationGaussNewtonSolver(max_iters=3, verbose=False),
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
with utils.stopwatch(f"Compute loss"):
    loss_value = compute_loss(params)
with utils.stopwatch(f"Compute loss"):
    loss_value = compute_loss(params)
for i in range(300):
    with utils.stopwatch(f"Loss step {i}: {loss_value}"):
        loss_value, grad = loss_grad_fn(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)

# Print optimized value
print(optimizer.target)
