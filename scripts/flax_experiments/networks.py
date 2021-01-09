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


class SimpleCNN(nn.Module):
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
