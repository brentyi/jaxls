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

params = {"a": 0.0}


def loss(params):
    return (3.0 - params["a"] * 5.0) ** 2


optimizer_def = optim.Adam(learning_rate=0.3)  # Choose the method
optimizer = optimizer_def.create(params)  # Create optimizer with initial parameters
loss_grad_fn = jax.value_and_grad(loss)

for i in range(101):
    loss_value, grad = loss_grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    # if i % 10 == 0:
    print(f"Loss step {i}: {loss_value}")

print(optimizer.target)
