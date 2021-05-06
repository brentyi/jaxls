import dataclasses
from typing import Sequence, Union

from jax import numpy as jnp
from overrides import overrides

from .. import hints
from ..utils import register_dataclass_pytree
from ._noise_model_base import NoiseModelBase


@register_dataclass_pytree
@dataclasses.dataclass
class Gaussian(NoiseModelBase):
    sqrt_precision_matrix: hints.Array
    """Lower-triangular square root precision matrix."""

    @staticmethod
    def make_from_covariance(covariance: hints.Array) -> "Gaussian":
        assert (
            len(covariance.shape) == 2 and covariance.shape[0] == covariance.shape[1]
        ), "Covariance must be a square matrix!"
        return Gaussian(
            sqrt_precision_matrix=jnp.linalg.inv(jnp.linalg.cholesky(covariance))
        )

    @overrides
    def get_residual_dim(self) -> int:
        return self.sqrt_precision_matrix.shape[-1]

    @overrides
    def whiten_residual_vector(self, residual_vector: hints.Array) -> hints.Array:
        return jnp.einsum("ij,j->i", self.sqrt_precision_matrix, residual_vector)

    @overrides
    def whiten_jacobian(
        self,
        jacobian: hints.Array,
        residual_vector: hints.Array,  # Unused
    ) -> hints.Array:
        return jnp.einsum("ij,jk->ik", self.sqrt_precision_matrix, jacobian)


@register_dataclass_pytree
@dataclasses.dataclass
class DiagonalGaussian(NoiseModelBase):
    sqrt_precision_diagonal: hints.Array
    """Diagonal elements of square root precision matrix."""

    @staticmethod
    def make_from_covariance(
        diagonal: Union[hints.Array, Sequence[float]]
    ) -> "DiagonalGaussian":
        return DiagonalGaussian(
            sqrt_precision_diagonal=1.0 / jnp.sqrt(jnp.asarray(diagonal))
        )

    @overrides
    def get_residual_dim(self) -> int:
        return self.sqrt_precision_diagonal.shape[-1]

    @overrides
    def whiten_residual_vector(self, residual_vector: hints.Array) -> hints.Array:
        assert residual_vector.shape == self.sqrt_precision_diagonal.shape
        return self.sqrt_precision_diagonal * residual_vector

    @overrides
    def whiten_jacobian(
        self,
        jacobian: hints.Array,
        residual_vector: hints.Array,  # Unused
    ) -> hints.Array:
        assert len(jacobian.shape) == 2
        assert (
            residual_vector.shape
            == self.sqrt_precision_diagonal.shape
            == (jacobian.shape[0],)
        )
        return self.sqrt_precision_diagonal[:, None] * jacobian
