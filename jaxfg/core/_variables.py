import abc
import functools
import inspect
from typing import Callable, ClassVar, Generic, Mapping, Type, TypeVar

import jax
import numpy as onp
from jax import flatten_util
from jax import numpy as jnp
from overrides import EnforceOverrides, final, overrides

from .. import hints

VariableType = TypeVar("VariableType", bound="VariableBase")
VariableValueType = TypeVar("VariableValueType", bound=hints.VariableValue)


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
        cls, x: VariableValueType, local_delta: hints.LocalVariableValue
    ) -> VariableValueType:
        r"""Retract local delta to manifold.

        Typically written as `x $\oplus$ local_delta` or `x $\boxplus$ local_delta`.

        Args:
            x: Absolute parameter to update.
            local_delta: Delta value in local parameterizaton.

        Returns:
            Updated parameterization.
        """
        return cls.unflatten(cls.flatten(x) + local_delta)

    # (3) Optional

    @classmethod
    def manifold_retract_jacobian(cls, x: VariableValueType) -> VariableValueType:
        """Jacobian of the variable parameters with respect to the local
        parameterization, linearized around `local_delta=zero`.
        """
        return jax.jacfwd(cls.manifold_retract, argnums=1)(
            x,
            onp.zeros(cls.get_local_parameter_dim()),
        )

    # (4) Shared implementation details.

    _parameter_dim: ClassVar[int]
    """Parameter dimensionality. Set automatically in `__init_subclass__`."""

    _unflatten: ClassVar[Callable[[hints.Array], VariableValueType]]
    """Helper for unflattening variable values. Set in `__init_subclass__`."""

    def __init__(self):
        """Variable constructor. Should take no arguments."""
        super().__init__()

    def __init_subclass__(cls):
        """For non-abstract subclasses, we determine the parameter dimensionality and
        unflattening procedure from the example provided by `get_default_value()`."""

        if inspect.isabstract(cls):
            return

        example = cls.get_default_value()

        flat, unflatten = flatten_util.ravel_pytree(example)
        (parameter_dim,) = flat.shape

        cls._parameter_dim = parameter_dim
        cls._unflatten = unflatten

    @classmethod
    @final
    def get_parameter_dim(cls) -> int:
        """Dimensionality of underlying parameterization."""
        return cls._parameter_dim

    @staticmethod
    @final
    def flatten(x: VariableValueType) -> jnp.ndarray:
        """Flatten variable value to 1D array.
        Should be similar to `jax.flatten_util.ravel_pytree`.

        Args:
            flat (hints.Array): 1D vector.

        Returns:
            VariableValueType: Variable value.
        """
        flat, _unflatten = flatten_util.ravel_pytree(x)
        return flat

    @classmethod
    @final
    def unflatten(cls, flat: hints.Array) -> VariableValueType:
        """Get variable value from flattened representation.

        Args:
            flat (hints.Array): 1D vector.

        Returns:
            VariableValueType: Variable value.
        """
        return cls._unflatten(flat)

    @classmethod
    @functools.lru_cache(maxsize=None)
    def canonical_instance(cls: Type[VariableType]) -> VariableType:
        """Returns the 'canonical instance' of a variable. For a given class, this will
        be the same instance each time the method is called. Used for factor stacking.
        """
        return cls()

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

    @classmethod
    @functools.lru_cache(maxsize=None)
    def __getitem__(cls, dim: int) -> Type[VariableBase]:
        assert isinstance(dim, int)

        class _RealVectorVariable(VariableBase[hints.Array]):
            @staticmethod
            @overrides
            @final
            def get_default_value() -> hints.Array:
                return jnp.zeros(dim)

        return _RealVectorVariable


RealVectorVariable: Mapping[int, Type[VariableBase[hints.Array]]]
RealVectorVariable = _RealVectorVariableTemplate()  # type: ignore
