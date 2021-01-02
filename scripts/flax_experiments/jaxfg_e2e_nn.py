"""Notes

- Goal here is to integrate a neural network into our factor graph, then learn that
  factor by optimizing end-to-end
- There are two options here...
    (a) Networks that are conditioned on variable nodes themselves
        Real world example: learning a dynamics model
        => this requires dynamic factor parameters, which will need some engineering work...
    (b) Networks that are conditioned on static inputs
        For robots: learning a sensor model (camera image => pose, etc)
        => this is simpler!
- Static input task?
    How simple can we make this?
        => Factor graph with only one node
        => Learning to take a cosine?
        => Sounds reasonable to me :shrug:
"""

import flax
import jax
import matplotlib.pyplot as plt
import numpy as onp
from flax import linen as nn
from jax import numpy as jnp
from tqdm.auto import tqdm

import jaxfg


# Helper class for MLPs
class SimpleMLP(nn.Module):
    units: int
    layers: int
    output_dim: int

    @staticmethod
    def make(units: int, layers: int, output_dim: int):
        """Dummy constructor for type-checking."""
        return SimpleMLP(units=units, layers=layers, output_dim=output_dim)

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        for i in range(self.layers):
            x = nn.Dense(self.units)(x)
            x = nn.relu(x)

        x = nn.Dense(self.output_dim)(x)
        return x


# Create our MLP
model = SimpleMLP.make(units=64, layers=4, output_dim=1)
prng_key = jax.random.PRNGKey(0)

# Create our optimizer
optimizer = flax.optim.Adam(learning_rate=1e-4).create(
    target=model.init(prng_key, jnp.zeros((1,)))  # Initial MLP parameters
)

# End-to-end forward pass
@jax.jit
def e2e_forward(model_params, x: jnp.ndarray) -> jnp.ndarray:
    # Network forward pass: computes a parameter for our linear factor
    mlp_out = model.apply(model_params, x)

    # Inner solve
    variable = jaxfg.core.RealVectorVariable[1]()
    initial_assignments = jaxfg.core.VariableAssignments.from_dict(
        {variable: onp.array([1.0])}
    )
    solved_assignments = jaxfg.core.PreparedFactorGraph.from_factors(
        [
            jaxfg.core.LinearFactor(
                variables=(variable,),
                scale_tril_inv=onp.eye(1),
                A_matrices=(onp.eye(1),),
                b=mlp_out,
            )
        ]
    ).solve(
        initial_assignments,
        solver=jaxfg.solvers.FixedIterationGaussNewtonSolver(max_iterations=3),
    )
    return solved_assignments.get_value(variable)


# Define loss, gradients, etc
def mse_loss(
    model_params: jaxfg.types.PyTree, batched_x: jnp.ndarray, batched_y: jnp.ndarray
):
    def squared_error(x: jnp.ndarray, y: jnp.ndarray):
        # Squared error for a single pair
        x = e2e_forward(model_params, x)
        assert x.shape == (1,) and y.shape == (1,)
        return ((x - y) ** 2).squeeze()

    return jnp.mean(jax.vmap(squared_error)(batched_x, batched_y))


loss_grad_fn = jax.jit(jax.value_and_grad(mse_loss, argnums=0))


# Generate same data & run optimization
minibatch_size = 100
batched_x = (onp.arange(-3140, 3140) / 1000.0)[:, onp.newaxis]
batched_y = onp.cos(batched_x)

progress = tqdm(range(10000))
for i in progress:
    minibatch_indices = onp.random.randint(
        low=0, high=batched_x.shape[0], size=(minibatch_size,)
    )
    loss_value, grad = loss_grad_fn(
        optimizer.target,
        batched_x[minibatch_indices],
        batched_y[minibatch_indices],
    )
    optimizer = optimizer.apply_gradient(grad)
    progress.set_description(f"Current loss: {loss_value:10.6f}")
    # if i % 10 == 0:
    # print(f"Loss step {i}: {loss_value}")


# Take a look at what we learned!
all_x = batched_x * 2.0

model_params = optimizer.target
plt.figure()
plt.plot(
    all_x.flatten(),
    onp.cos(all_x).flatten(),
    label="Ground-truth",
)
vmap_forward = jax.jit(jax.vmap(lambda x: e2e_forward(model_params, x)))
plt.plot(
    all_x.flatten(),
    vmap_forward(all_x).flatten(),
    label="Model extrapolation",
)
plt.plot(
    batched_x.flatten(),
    vmap_forward(batched_x).flatten(),
    label="Model output on training data",
)
plt.show()
plt.legend()
