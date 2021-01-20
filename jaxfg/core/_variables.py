import abc
from typing import TYPE_CHECKING, Dict, Generic, Mapping, Tuple, Type, TypeVar

import numpy as onp
from jax import numpy as jnp
from overrides import overrides

from .. import types

VariableValueType = TypeVar("VariableValueType", bound=types.VariableValue)


class VariableBase(abc.ABC, Generic[VariableValueType]):
    _parameter_dim: int

    @staticmethod
    @abc.abstractmethod
    def get_parameter_dim() -> int:
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
    def manifold_retract(
        x: VariableValueType, local_delta: VariableValueType
    ) -> VariableValueType:
        r"""Retract local delta to manifold.

        Typically written as `x $\oplus$ local_delta` or `x $\boxplus$ local_delta`.

        Args:
            x (VariableValue): Absolute parameter to update.
            local_delta (VariableValue): Delta value in local parameterizaiton.

        Returns:
            jnp.ndarray: Updated parameterization.
        """

    @staticmethod
    @abc.abstractmethod
    def manifold_inverse_retract(
        x: VariableValueType, y: VariableValueType
    ) -> types.LocalVariableValue:
        """Compute the local difference between two variable values.

        Typically written as `x $\ominus$ y` or `x $\boxminus$ y`.

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
    def manifold_retract(x: jnp.ndarray, local_delta: jnp.ndarray) -> jnp.ndarray:
        return x + local_delta

    @staticmethod
    @overrides
    def manifold_inverse_retract(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return x - y

    @staticmethod
    @overrides
    def flatten(x: jnp.ndarray) -> jnp.ndarray:
        return x

    @staticmethod
    @overrides
    def unflatten(flat: jnp.ndarray) -> jnp.ndarray:
        return flat


class _RealVectorVariableTemplate:
    """Usage: `RealVectorVariable[N]`, where `N` is an integer dimension."""

    _real_vector_variable_cache: Dict[int, Type[AbstractRealVectorVariable]] = {}

    @classmethod
    def __getitem__(cls, dim: int) -> Type[AbstractRealVectorVariable]:
        assert isinstance(dim, int)

        if dim not in cls._real_vector_variable_cache:

            class _NDimensionalRealVectorVariable(AbstractRealVectorVariable):
                @staticmethod
                @overrides
                def get_parameter_dim() -> int:
                    return dim

                @staticmethod
                @overrides
                def get_local_parameter_dim() -> int:
                    return dim

                @staticmethod
                @overrides
                def get_default_value() -> onp.ndarray:
                    return onp.zeros(dim)

            cls._real_vector_variable_cache[dim] = _NDimensionalRealVectorVariable

        return cls._real_vector_variable_cache[dim]


RealVectorVariable: Mapping[int, Type[AbstractRealVectorVariable]]
RealVectorVariable = _RealVectorVariableTemplate()  # type: ignore
