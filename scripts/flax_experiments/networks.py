from typing import Tuple

import flax
import jax
import jaxfg
import numpy as onp
from flax import linen as nn
from jax import numpy as jnp


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


class PositiveMLP(nn.Module):
    """PositiveMLP."""

    units: int
    layers: int
    output_dim: int

    @staticmethod
    def make(units: int, layers: int, output_dim: int):
        """Dummy constructor for type-checking."""
        return SimpleMLP(units=units, layers=layers, output_dim=output_dim)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        """__call__.

        Args:
            inputs (jnp.ndarray): inputs
        """
        x = inputs

        for i in range(self.layers):
            x = nn.Dense(self.units)(x)
            x = nn.relu(x)

        x = nn.Dense(self.output_dim)(x)
        return x ** 2


class SimpleCNN(nn.Module):
    """CNN.

    Input is (N, 120, 120, 3) images.
    Output is (N, 2) position prediction.
    """

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        x = inputs
        N = x.shape[0]
        assert x.shape == (N, 120, 120, 3), x.shape

        # Some conv layers
        for _ in range(3):
            x = nn.Conv(features=32, kernel_size=(3, 3))(x)
            x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)

        # Channel-wise max pool
        x = jnp.max(x, axis=3, keepdims=True)

        # Spanning mean pools (to regularize X/Y coordinate regression)
        assert x.shape == (N, 120, 120, 1)
        x_horizontal = nn.avg_pool(x, window_shape=(120, 1))
        x_vertical = nn.avg_pool(x, window_shape=(1, 120))

        # Concatenate, feed through MLP
        x = jnp.concatenate(
            [x_horizontal.reshape((N, -1)), x_vertical.reshape((N, -1))], axis=1
        )
        assert x.shape == (N, 240)
        x = SimpleMLP.make(units=32, layers=3, output_dim=2)(x)

        return x


def make_position_cnn(seed: int = 0) -> Tuple[SimpleCNN, flax.optim.Adam]:
    """Make CNN and ADAM optimizer for disk tracking predictions.

    Args:
        seed (int): Random seed.

    Returns:
        Tuple[SimpleCNN, jaxfg.types.PyTree]: Tuple of (model, model_parameters).
    """
    model = SimpleCNN()

    prng_key = jax.random.PRNGKey(seed=seed)
    dummy_image = onp.zeros((1, 120, 120, 3))
    return model, flax.optim.Adam().create(model.init(prng_key, dummy_image))


def make_uncertainty_mlp(seed: int = 0) -> Tuple[SimpleMLP, flax.optim.Adam]:
    """Make CNN and ADAM optimizer for mapping # of visible pixels => inverse standard
    deviation of position estimate.

    Args:
        seed (int): seed

    Returns:
        Tuple[SimpleMLP, flax.optim.Adam]:
    """
    model = PositiveMLP.make(units=64, layers=4, output_dim=1)

    prng_key = jax.random.PRNGKey(seed=seed)
    dummy_input = onp.zeros((1, 1))
    return model, flax.optim.Adam().create(model.init(prng_key, dummy_input))
