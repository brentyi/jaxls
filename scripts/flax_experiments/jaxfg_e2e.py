# Some options
# > Nested optimization

import jax
import numpy as onp
from flax import optim
from jax import numpy as jnp
from jaxfg import core, geometry, solvers, utils
from jaxlie import SE2

from _fixed_iteration_gn import FixedIterationGaussNewtonSolver

T_ab = SE2.from_xy_theta(1.0, 0.0, 0.2)
T_bc = SE2.from_xy_theta(1.0, 0.0, 0.2)

T_wa = SE2.identity()
T_wb = T_wa @ T_ab
T_wc = T_wb @ T_bc

A = geometry.SE2Variable()
B = geometry.SE2Variable()
C = geometry.SE2Variable()


def sample_se2_noise(std_dev=1.0):
    return SE2.exp(jnp.array(onp.random.randn(SE2.tangent_dim) * std_dev))


assignments_ground_truth = core.VariableAssignments.from_dict(
    {
        A: T_wa,
        B: T_wb,
        C: T_wc,
    }
)

noise1 = sample_se2_noise()
noise2 = sample_se2_noise()

def compute_loss(scale):
    graph = core.PreparedFactorGraph.from_factors(
        [
            geometry.PriorFactor.make(
                variable=A, mu=SE2.identity(), scale_tril_inv=jnp.eye(3)
            ),
            geometry.BetweenFactor.make(
                before=A, after=B, delta=T_ab, scale_tril_inv=jnp.eye(3)
            ),
            geometry.BetweenFactor.make(
                before=B, after=C, delta=T_bc, scale_tril_inv=jnp.eye(3)
            ),
            # Noisy factors
            geometry.PriorFactor.make(
                variable=B,
                mu=T_wb @ noise1,
                scale_tril_inv=jnp.eye(3) * scale,
            ),
            geometry.PriorFactor.make(
                variable=C,
                mu=T_wc @ noise2,
                scale_tril_inv=jnp.eye(3) * scale,
            ),
        ]
    )

    # print(graph.compute_sum_squared_error(assignments_ground_truth))
    assignments_solved = graph.solve(
        assignments_ground_truth, solver=FixedIterationGaussNewtonSolver(max_iters=3, verbose=False)
    )

    error = jax.vmap(SE2.log)(
        jax.vmap(SE2.multiply)(
            assignments_solved.get_stacked_value(geometry.SE2Variable),
            jax.vmap(SE2.inverse)(
                assignments_ground_truth.get_stacked_value(geometry.SE2Variable)
            ),
        )
    )
    sse = jnp.sum(error ** 2)
    return sse


print(compute_loss(0.0))
print(compute_loss(100.0))

params = 5.0

optimizer_def = optim.Adam(learning_rate=1e-2)
optimizer = optimizer_def.create(params)
loss_grad_fn = jax.value_and_grad(compute_loss)

loss_value = 0
for i in range(10):
    with utils.stopwatch(f"Loss step {i}: {loss_value}"):
        loss_value, grad = loss_grad_fn(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)

print(optimizer.target)
