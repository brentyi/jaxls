import jax_dataclasses as jdc
from jax import numpy as jnp
from overrides import overrides

from .. import hints
from ._noise_model_base import NoiseModelBase


@jdc.pytree_dataclass
class HuberWrapper(NoiseModelBase):
    """Wrapper for applying a Huber loss to standard (eg Gaussian) noise models.

    TODO(brentyi): this should be functional in terms of optimization, but is still an
    experimental state and has minor issues. Notably causes inaccuracy in the value
    returned by `StackedFactorGraph.compute_cost()`."""

    wrapped: NoiseModelBase
    """Underlying noise model."""

    delta: hints.Scalar
    """Threshold parameter for Huber loss. Applied _after_ the wrapped noise model."""

    @overrides
    def get_residual_dim(self) -> int:
        return self.wrapped.get_residual_dim()

    @overrides
    def whiten_residual_vector(self, residual_vector: hints.Array) -> hints.Array:
        residual_vector = self.wrapped.whiten_residual_vector(residual_vector)

        residual_norm = jnp.linalg.norm(residual_vector)
        return jnp.where(
            residual_norm < self.delta,
            residual_vector,
            residual_vector * jnp.sqrt(self.delta / residual_norm),
        )

    @overrides
    def whiten_jacobian(
        self,
        jacobian: hints.Array,
        residual_vector: hints.Array,
    ) -> hints.Array:
        jacobian = self.wrapped.whiten_jacobian(jacobian, residual_vector)

        residual_norm = jnp.linalg.norm(residual_vector)
        return jnp.where(
            residual_norm < self.delta,
            jacobian,
            jacobian * jnp.sqrt(self.delta / residual_norm),
        )
