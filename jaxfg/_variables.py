import abc
import contextlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, Generator, Optional, Set, Tuple

import jax
import numpy as onp
from jax import numpy as jnp
from overrides import overrides

if TYPE_CHECKING:
    from . import LinearFactor


class VariableBase(abc.ABC):
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
    def get_default_value() -> onp.ndarray:
        """Get default (on-manifold) parameter value."""

    @classmethod
    @abc.abstractmethod
    def add_local(cls, x: jnp.ndarray, local_delta: jnp.ndarray) -> jnp.ndarray:
        """On-manifold retraction.

        Args:
            local_delta (jnp.ndarray): Delta value in local parameterizaiton.
            x (jnp.ndarray): Absolute parameter to update.

        Returns:
            jnp.ndarray: Updated parameterization.
        """

    @classmethod
    @abc.abstractmethod
    def subtract_local(cls, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Compute the difference between two parameters on the manifold.

        Args:
            x (jnp.ndarray): First parameter to compare. Shape should match `self.get_parameter_dim()`.
            y (jnp.ndarray): Second parameter to compare. Shape should match `self.get_parameter_dim()`.

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

    @classmethod
    @overrides
    def add_local(cls, x: jnp.ndarray, local_delta: jnp.ndarray) -> jnp.ndarray:
        return x + local_delta

    @classmethod
    @overrides
    def subtract_local(cls, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return x - y


_real_vector_variable_cache = {}


class _RealVectorVariableTemplate:
    def __getitem__(self, n: int):
        assert isinstance(n, int)

        if n not in _real_vector_variable_cache:

            class _NDimensionalRealVectorVariable(AbstractRealVectorVariable):
                @staticmethod
                @overrides
                def get_parameter_dim() -> int:
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


# Lie manifolds


class SO2Variable(VariableBase):
    """Variable containing a 2D rotation."""

    @staticmethod
    @overrides
    def get_parameter_dim() -> int:
        return 2

    @staticmethod
    @overrides
    def get_local_parameter_dim() -> int:
        return 1

    @staticmethod
    @overrides
    def get_default_value() -> onp.ndarray:
        return onp.array([1.0, 0.0])

    @classmethod
    @overrides
    def add_local(cls, x: jnp.ndarray, local_delta: jnp.ndarray) -> jnp.ndarray:
        assert x.shape == (2,) and local_delta.shape == (1,)
        theta = local_delta[0]
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        R = jnp.array(
            [
                [cos, -sin],
                [sin, cos],
            ]
        )
        return R @ x

    @classmethod
    @overrides
    def subtract_local(cls, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        assert x.shape == y.shape == (2,)

        # The intuitive solution based on the definition of a dot product would be:
        # > delta = jnp.arccos(jnp.sum(x * y, axis=0, keepdims=True))
        # But this ignores signs (x and y can be swapped).
        delta = jnp.arctan2(
            x[0] * y[1] - x[1] * y[0],  # "2D cross product"; is there a name for this?
            x @ y,  # Aligned component with dot prodcut
        )[None]
        assert delta.shape == (1,)
        return delta


class SE2Variable(VariableBase):
    """Variable containing a 2D pose."""

    @staticmethod
    @overrides
    def get_parameter_dim() -> int:
        # (x, y, cos, sin)
        return 4

    @staticmethod
    @overrides
    def get_local_parameter_dim() -> int:
        # (x, y, theta)
        return 3

    @staticmethod
    @overrides
    def get_default_value() -> onp.ndarray:
        return onp.array([0.0, 0.0, 1.0, 0.0])

    @classmethod
    @overrides
    def add_local(cls, x: jnp.ndarray, local_delta: jnp.ndarray) -> jnp.ndarray:

        # Our pose is: T_world_A
        # We're getting: T_A_B
        # We want: T_world_B

        assert x.shape == (4,) and local_delta.shape == (3,)
        cos = x[2]
        sin = x[3]
        R = jnp.array(
            [
                [cos, -sin],
                [sin, cos],
            ]
        )
        summed = jnp.concatenate(
            [
                x[:2] + R @ local_delta[:2],
                SO2Variable.add_local(x[2:4], local_delta[2:3]),
            ]
        )
        assert summed.shape == (4,)
        return summed

    @classmethod
    @overrides
    def subtract_local(cls, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        assert x.shape == y.shape == (4,)
        delta = jnp.concatenate(
            [x[:2] - y[:2], SO2Variable.subtract_local(x[2:4], y[2:4])]
        )
        assert delta.shape == (3,)
        return delta
