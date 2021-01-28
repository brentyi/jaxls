from typing import Tuple

import flax
import jax
import numpy as onp
from flax import linen as nn
from jax import numpy as jnp

import jaxfg


class SimpleMLP(nn.Module):
    units: int
    layers: int
    output_dim: int

    @staticmethod
    def make(units: int, layers: int, output_dim: int):
        """Dummy constructor for type-checking."""
        return SimpleMLP(units=units, layers=layers, output_dim=output_dim)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        x = inputs

        for i in range(self.layers):
            x = nn.Dense(self.units)(x)
            x = nn.relu(x)

        x = nn.Dense(self.output_dim)(x)
        return x


class SimpleCNN(nn.Module):
    """CNN.

    Input is (N, 50, 150, 6) images.
    Output is (N, 4). Linear, angular velocities, followed by covariances for each.
    """

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        x = inputs
        N = x.shape[0]
        assert x.shape == (N, 50, 150, 6), x.shape

        # conv1
        x = nn.Conv(features=16, kernel_size=(7, 7))(x)
        x = nn.relu(x)

        # conv2
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(1, 2))(x)
        x = nn.relu(x)

        # conv3
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(1, 2))(x)
        x = nn.relu(x)

        # conv4
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.relu(x)

        # Validate shape
        # Kloss paper reports (N, 25, 18, 16), but this might be a typo from the even
        # stride/odd image dimensions. Or just a Tensorflow vs flax implementation
        # difference.
        assert x.shape == (N, 25, 19, 16)

        # Concatenate, feed through MLP
        x = x.reshape((N, -1))
        x = SimpleMLP.make(units=128, layers=2, output_dim=4)(x)

        return x


def make_observation_cnn(seed: int = 0) -> Tuple[SimpleCNN, flax.optim.Adam]:
    """Make CNN and ADAM optimizer for processing KITTI images.

    Args:
        seed (int): Random seed.

    Returns:
        Tuple[SimpleCNN, jaxfg.types.PyTree]: Tuple of (model, model_parameters).
    """
    model = SimpleCNN()

    prng_key = jax.random.PRNGKey(seed=seed)
    dummy_image = onp.zeros((1, 50, 150, 6))
    return model, flax.optim.Adam().create(model.init(prng_key, dummy_image))
