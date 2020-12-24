from typing import Any, Callable, Optional, Sequence

import flax
import jax
from flax import linen as nn
from flax import optim
from flax.core import freeze, unfreeze
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.config import config

config.enable_omnistaging()  # Linen requires enabling omnistaging

model = nn.Dense(features=5)
key1, key2 = random.split(random.PRNGKey(0))
x = random.normal(key1, (10,))  # Dummy input
params = model.init(key2, x)  # Initialization call
print(jax.tree_map(lambda x: x.shape, params))  # Checking output shapes


print(x)
print(model.apply(params, x))

# Set problem dimensions
nsamples = 20
xdim = 10
ydim = 5

# Generate random ground truth W and b
key = random.PRNGKey(0)
k1, k2 = random.split(key)
W = random.normal(k1, (xdim, ydim))
b = random.normal(k2, (ydim,))
true_params = freeze({"params": {"bias": b, "kernel": W}})

# Generate samples with additional noise
ksample, knoise = random.split(k1)
x_samples = random.normal(ksample, (nsamples, xdim))
y_samples = jnp.dot(x, W) + b
y_samples += 0.1 * random.normal(knoise, (nsamples, ydim))  # Adding noise
print("x shape:", x_samples.shape, "; y shape:", y_samples.shape)


def make_mse_func(x_batched, y_batched):
    def mse(params):
        # Define the squared loss for a single pair (x,y)
        def squared_error(x, y):
            pred = model.apply(params, x)
            return jnp.inner(y - pred, y - pred) / 2.0

        # We vectorize the previous to compute the average of the loss on all samples.
        return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)

    return jax.jit(mse)  # And finally we jit the result.


# Get the sampled loss
loss = make_mse_func(x_samples, y_samples)

optimizer_def = optim.Adam(learning_rate=0.3)  # Choose the method
optimizer = optimizer_def.create(params)  # Create optimizer with initial parameters
loss_grad_fn = jax.value_and_grad(loss)

for i in range(101):
    loss_value, grad = loss_grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    # if i % 10 == 0:
    print(f"Loss step {i}: {loss_value}")
