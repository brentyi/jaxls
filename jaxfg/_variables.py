import abc
import contextlib
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Dict,
    Generator,
    Generic,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)

import jax
import jaxlie
import numpy as onp
from jax import numpy as jnp
from overrides import overrides

from . import _types, _utils

if TYPE_CHECKING:
    from . import LinearFactor


VariableValueType = TypeVar("VariableValueType", bound=_types.VariableValue)


class VariableBase(abc.ABC, Generic[VariableValueType]):
    _parameter_dim: int

    @staticmethod
    @abc.abstractmethod
    def get_parameter_shape() -> Tuple[int, ...]:
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
    ) -> _types.LocalVariableValue:
        """Compute the local difference between two variable values.

        Args:
            x (VariableValue): First parameter to compare. Shape should match `self.get_parameter_shape()`.
            y (VariableValue): Second parameter to compare. Shape should match `self.get_parameter_shape()`.

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
                def get_parameter_shape() -> Tuple[int, ...]:
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


# Lie groups


def make_lie_variable(Group: Type[jaxlie.MatrixLieGroup]):
    class _LieVariable(VariableBase[Group]):
        """Variable containing a transformation."""

        @staticmethod
        @overrides
        def get_parameter_shape() -> Tuple[int, ...]:
            return (Group.parameters_dim,)

        @staticmethod
        @overrides
        def get_local_parameter_dim() -> int:
            return Group.tangent_dim

        @staticmethod
        @overrides
        def get_default_value() -> Group:
            return Group.identity()

        @staticmethod
        #  @jax.custom_jvp
        def add_local(x: Group, local_delta: jaxlie.types.TangentVector) -> Group:
            return Group(x) @ Group.exp(local_delta)

        @staticmethod
        @overrides
        def subtract_local(x: Group, y: Group) -> _types.LocalVariableValue:
            # x = world<-A, y = world<-B
            # Difference = A<-B
            return (x.inverse() @ y).log()

        @staticmethod
        @overrides
        def flatten(x: Group) -> jnp.ndarray:
            return x.parameters

        @staticmethod
        @overrides
        def unflatten(flat: jnp.ndarray) -> Group:
            return Group(flat)

    return _LieVariable


SO2Variable = make_lie_variable(jaxlie.SO2)
SE2Variable = make_lie_variable(jaxlie.SE2)
SO3Variable = make_lie_variable(jaxlie.SO3)
SE3Variable = make_lie_variable(jaxlie.SE3)
