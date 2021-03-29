import abc
from typing import Dict, Generic, Mapping, Tuple, Type, TypeVar

from jax import flatten_util
from jax import numpy as jnp
from overrides import EnforceOverrides, final, overrides

from .. import types

VariableValueType = TypeVar("VariableValueType", bound=types.VariableValue)


class VariableBase(abc.ABC, Generic[VariableValueType], EnforceOverrides):
    """Base class for variable types. Also defines helpers for manifold optimization."""

    # (1) Functions that must be overriden in subclasses.

    @classmethod
    @abc.abstractmethod
    def get_default_value(cls) -> VariableValueType:
        """Get default (on-manifold) parameter value."""

    # (2) Functions to override for custom manifolds / local parameterizations.

    @classmethod
    def get_local_parameter_dim(cls) -> int:
        """Dimensionality of local parameterization."""
        return cls.get_parameter_dim()

    @classmethod
    def manifold_retract(
        cls, x: VariableValueType, local_delta: types.LocalVariableValue
    ) -> VariableValueType:
        r"""Retract local delta to manifold.

        Typically written as `x $\oplus$ local_delta` or `x $\boxplus$ local_delta`.

        Args:
            x (VariableValue): Absolute parameter to update.
            local_delta (VariableValue): Delta value in local parameterizaton.

        Returns:
            jnp.ndarray: Updated parameterization.
        """
        return cls.unflatten(cls.flatten(x) + local_delta)

    # (3) Shared implementation details.

    @final
    def __init_subclass__(cls):
        """For subclasses, we determine the parameter dimensionality and unflattening
        procedure from the example provided by `get_default_value()`."""

        example = cls.get_default_value()
        flat, unflatten = flatten_util.ravel_pytree(example)
        (parameter_dim,) = flat.shape

        cls.get_parameter_dim = staticmethod(lambda: parameter_dim)
        cls.unflatten = staticmethod(unflatten)

    @staticmethod
    @final  # Should not be overriden, see __init_subclass__
    def get_parameter_dim() -> int:
        """Dimensionality of underlying parameterization."""
        raise NotImplementedError(
            "Should be defined in VariableBase.__init_subclass__!"
        )

    @staticmethod
    @final
    def flatten(x: VariableValueType) -> jnp.ndarray:
        """Flatten variable value to 1D array.
        Should be similar to `jax.flatten_util.ravel_pytree`.

        Args:
            flat (jnp.ndarray): 1D vector.

        Returns:
            VariableValueType: Variable value.
        """
        flat, _unflatten = flatten_util.ravel_pytree(x)
        return flat

    @staticmethod
    @final  # Should not be overriden, see __init_subclass__
    def unflatten(flat: jnp.ndarray) -> VariableValueType:
        """Get variable value from flattened representation.

        Args:
            flat (jnp.ndarray): 1D vector.

        Returns:
            VariableValueType: Variable value.
        """
        raise NotImplementedError(
            "Should be defined in VariableBase.__init_subclass__!"
        )

    @overrides
    @final
    def __lt__(self, other) -> bool:
        """Compare hashes between variables. Needed to use as PyTree key.

        Args:
            other: Other object to compare.

        Returns:
            bool: True if `self < other`.
        """
        return hash(self) < hash(other)


# Fake templating; RealVectorVariable[N]
class _RealVectorVariableTemplate:
    """Usage: `RealVectorVariable[N]`, where `N` is an integer dimension."""

    _real_vector_variable_cache: Dict[int, Type[VariableBase]] = {}

    @classmethod
    def __getitem__(cls, dim: int) -> Type[VariableBase]:
        assert isinstance(dim, int)

        if dim not in cls._real_vector_variable_cache:

            class _RealVectorVariable(VariableBase[jnp.ndarray]):
                @staticmethod
                @overrides
                @final
                def get_default_value() -> jnp.ndarray:
                    return jnp.zeros(dim)

            cls._real_vector_variable_cache[dim] = _RealVectorVariable

        return cls._real_vector_variable_cache[dim]


RealVectorVariable: Mapping[int, Type[VariableBase[jnp.ndarray]]]
RealVectorVariable = _RealVectorVariableTemplate()  # type: ignore
