import abc
from typing import TYPE_CHECKING, Generic, Tuple, TypeVar

import numpy as onp
from jax import numpy as jnp
from overrides import overrides

from .. import types

VariableValueType = TypeVar("VariableValueType", bound=types.VariableValue)


class VariableBase(abc.ABC, Generic[VariableValueType]):
    _parameter_dim: int

    @staticmethod
    @abc.abstractmethod
    def get_parameter_dim(cls) -> int:
        """Dimensionality of underlying parameterization."""

    @staticmethod
    @abc.abstractmethod
    def get_local_parameter_dim() -> int:
        """Dimensionality of local parameterization."""

    @staticmethod
    @abc.abstractmethod
    def get_default_value() -> VariableValueType:
        """Get default (on-manifold) parameter value."""

    @staticmethod
    @abc.abstractmethod
    def add_local(
        x: VariableValueType, local_delta: VariableValueType
    ) -> VariableValueType:
        """On-manifold retraction.

        Args:
            x (VariableValue): Absolute parameter to update.
            local_delta (VariableValue): Delta value in local parameterizaiton.

        Returns:
            jnp.ndarray: Updated parameterization.
        """

    @staticmethod
    @abc.abstractmethod
    def subtract_local(
        x: VariableValueType, y: VariableValueType
    ) -> types.LocalVariableValue:
        """Compute the local difference between two variable values.

        Args:
            x (VariableValue): First parameter to compare. Shape should match `self.get_parameter_dim()`.
            y (VariableValue): Second parameter to compare. Shape should match `self.get_parameter_dim()`.

        Returns:
            LocalVariableValue: Delta vector; dimension should match self.get_local_parameter_dim().
        """

    @staticmethod
    @abc.abstractmethod
    def flatten(x: VariableValueType) -> jnp.ndarray:
        """Flatten variable value to 1D array.
        Should be similar to `jax.flatten_util.ravel_pytree`.

        Args:
            flat (jnp.ndarray): 1D vector.

        Returns:
            VariableValueType: Variable value.
        """

    @staticmethod
    @abc.abstractmethod
    def unflatten(flat: jnp.ndarray) -> VariableValueType:
        """Get variable value from flattened representation.

        Args:
            flat (jnp.ndarray): 1D vector.

        Returns:
            VariableValueType: Variable value.
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
class AbstractRealVectorVariable(VariableBase[jnp.ndarray]):
    """Variable for an arbitrary vector of real numbers."""

    @staticmethod
    @overrides
    def add_local(x: jnp.ndarray, local_delta: jnp.ndarray) -> jnp.ndarray:
        return x + local_delta

    @staticmethod
    @overrides
    def subtract_local(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return x - y

    @staticmethod
    @overrides
    def flatten(x: jnp.ndarray) -> jnp.ndarray:
        return x

    @staticmethod
    @overrides
    def unflatten(flat: jnp.ndarray) -> jnp.ndarray:
        return flat


_real_vector_variable_cache = {}


class _RealVectorVariableTemplate:
    def __getitem__(self, n: int):
        assert isinstance(n, int)

        if n not in _real_vector_variable_cache:

            class _NDimensionalRealVectorVariable(AbstractRealVectorVariable):
                @staticmethod
                @overrides
                def get_parameter_dim() -> Tuple[int, ...]:
                    return n

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
