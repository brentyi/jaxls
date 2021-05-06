import abc

from jax import numpy as jnp
from overrides import EnforceOverrides

from .. import hints


class NoiseModelBase(abc.ABC, EnforceOverrides):
    @abc.abstractmethod
    def get_residual_dim(self) -> int:
        pass

    @abc.abstractmethod
    def whiten_residual_vector(self, residual_vector: hints.Array) -> hints.Array:
        pass

    @abc.abstractmethod
    def whiten_jacobian(
        self,
        jacobian: hints.Array,
        residual_vector: hints.Array,
    ) -> hints.Array:
        pass
