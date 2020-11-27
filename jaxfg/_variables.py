import abc
import contextlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, Generator, Optional, Set, Tuple, Type

import jax
import numpy as onp
from jax import numpy as jnp
from overrides import overrides

from . import _utils

if TYPE_CHECKING:
    from . import LinearFactor


class VariableBase(abc.ABC):
    _parameter_dim: int

    @staticmethod
    @abc.abstractmethod
    def get_parameter_shape() -> int:
        """Dimensionality of underlying parameterization."""

    @classmethod
    def get_parameter_dim(cls) -> int:
        return cls._parameter_dim

    def __init_subclass__(cls, **kwargs):
        """Register all factors as PyTree nodes."""
        cls._parameter_dim = onp.prod(cls.get_parameter_shape())

    @staticmethod
    @abc.abstractmethod
    def get_local_parameter_dim() -> int:
        """Dimensionality of local parameterization."""

    @staticmethod
    @abc.abstractmethod
    def get_default_value() -> onp.ndarray:
        """Get default (on-manifold) parameter value."""

    @staticmethod
    @abc.abstractmethod
    def add_local(x: jnp.ndarray, local_delta: jnp.ndarray) -> jnp.ndarray:
        """On-manifold retraction.

        Args:
            local_delta (jnp.ndarray): Delta value in local parameterizaiton.
            x (jnp.ndarray): Absolute parameter to update.

        Returns:
            jnp.ndarray: Updated parameterization.
        """

    @staticmethod
    @abc.abstractmethod
    def subtract_local(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Compute the difference between two parameters on the manifold.

        Args:
            x (jnp.ndarray): First parameter to compare. Shape should match `self.get_parameter_shape()`.
            y (jnp.ndarray): Second parameter to compare. Shape should match `self.get_parameter_shape()`.

        Returns:
            jnp.ndarray: Delta vector; dimension should match self.get_local_parameter_dim().
        """

    @overrides
    def __lt__(self, other) -> bool:
        """Compare hashes between variables. Needed to use as pytree key. :shrug:

        Args:
            other: Other object to compare.

        Returns:
            bool: True if `self < other`.
        """
        return hash(self) < hash(other)


# Fake templating; RealVectorVariable[N]
class AbstractRealVectorVariable(VariableBase):
    """Variable for an arbitrary vector of real numbers."""

    @staticmethod
    @overrides
    def add_local(x: jnp.ndarray, local_delta: jnp.ndarray) -> jnp.ndarray:
        return x + local_delta

    @staticmethod
    @overrides
    def subtract_local(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return x - y


_real_vector_variable_cache = {}


class _RealVectorVariableTemplate:
    def __getitem__(self, n: int):
        assert isinstance(n, int)

        if n not in _real_vector_variable_cache:

            class _NDimensionalRealVectorVariable(AbstractRealVectorVariable):
                @staticmethod
                @overrides
                def get_parameter_shape() -> int:
                    return (n,)

                @staticmethod
                @overrides
                def get_local_parameter_dim() -> int:
                    return n

                @staticmethod
                @overrides
                def get_default_value() -> onp.ndarray:
                    return onp.zeros(n)

            _real_vector_variable_cache[n] = _NDimensionalRealVectorVariable
        return _real_vector_variable_cache[n]


RealVectorVariable = _RealVectorVariableTemplate()


# Lie manifolds


def make_lie_variable(group: Type[_utils.MatrixLieGroup]):
    class _LieVariable(VariableBase):
        """Variable containing a 2D rotation."""

        @staticmethod
        @overrides
        def get_parameter_shape() -> int:
            # Full 2x2 rotation matrix
            dim = group.get_matrix_dim()
            return (dim, dim)

        @staticmethod
        @overrides
        def get_local_parameter_dim() -> int:
            return group.get_tangent_dim()

        @staticmethod
        @overrides
        def get_default_value() -> onp.ndarray:
            return group.identity()

        @staticmethod
        #  @jax.custom_jvp
        def add_local(x: jnp.ndarray, local_delta: jnp.ndarray) -> jnp.ndarray:
            return x @ group.exp(local_delta)

        @staticmethod
        @overrides
        def subtract_local(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            # x = world<-A, y = world<-B
            # Difference = A<-B
            return group.log(group.inverse(x) @ y)

    return _LieVariable


SO2Variable = make_lie_variable(_utils.SO2)
SE2Variable = make_lie_variable(_utils.SE2)
