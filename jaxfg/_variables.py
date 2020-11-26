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


class SO2Variable(VariableBase):
    """Variable containing a 2D rotation."""

    @staticmethod
    @overrides
    def get_parameter_shape() -> int:
        # Full 2x2 rotation matrix
        return (4,)

    @staticmethod
    @overrides
    def get_local_parameter_dim() -> int:
        return 1

    @staticmethod
    @overrides
    def get_default_value() -> onp.ndarray:
        return onp.eye(2)

    @staticmethod
    #  @jax.custom_jvp
    def add_local(x: jnp.ndarray, local_delta: jnp.ndarray) -> jnp.ndarray:
        assert x.shape == (4,) and local_delta.shape == (1,)
        theta = local_delta[0]
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        exp = jnp.array(
            [
                [cos, -sin],
                [sin, cos],
            ]
        )
        return exp @ x.reshape((2, 2))

    @staticmethod
    @overrides
    def subtract_local(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        assert x.shape == y.shape == (4,)

        R_world_A = x.reshape((2, 2))
        R_world_B = y.reshape((2, 2))
        R_A_B = R_world_A.T @ R_world_B

        return jnp.arctan2(R_A_B[1:2, 0], R_A_B[0:1, 0])


# Analytical JVP for SO2 add_local; slower than having Jax figure it out for us hah
#
#  @SO2Variable.add_local.defjvp
#  def add_local_jvp(primals: Tuple[jnp.ndarray, jnp.ndarray], tangents):
#      x, local_delta = primals
#      x_dot, local_delta_dot = tangents
#
#      # Primal out is: exp(skew(local_delta)) @ x
#      primal_out = SO2Variable.add_local(x, local_delta)
#
#      local_delta = local_delta[0]
#      local_delta_dot = local_delta_dot[0]
#
#      cos = jnp.cos(local_delta)
#      sin = jnp.sin(local_delta)
#      R = jnp.array([[cos, -sin], [sin, cos]])
#      so2_generator = jnp.array([[0.0, -local_delta_dot], [local_delta_dot, 0.0]])
#      tangent_out = R @ x_dot + so2_generator @ primal_out
#
#      return primal_out, tangent_out


class SE2Variable(VariableBase):
    """Variable containing a 2D pose."""

    @staticmethod
    @overrides
    def get_parameter_shape() -> int:
        # Full SE(2) homogeneous transform :shrug:
        return (3, 3)

    @staticmethod
    @overrides
    def get_local_parameter_dim() -> int:
        # se(2) generator: (x, y, theta)
        return 3

    @staticmethod
    @overrides
    def get_default_value() -> onp.ndarray:
        return onp.eye(3)

    @staticmethod
    @overrides
    def add_local(x: jnp.ndarray, local_delta: jnp.ndarray) -> jnp.ndarray:

        # y = x (+) delta
        #
        # Our pose is: T_world_A
        # We're getting: T_A_B
        # We want: T_world_B

        T_world_A = x
        assert T_world_A.shape == (3, 3)

        x, y, theta = local_delta
        theta = theta
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        T_A_B = jnp.array(
            [
                [cos, -sin, x],
                [sin, cos, y],
                [0.0, 0.0, 1.0],
            ]
        )

        T_world_B = T_world_A @ T_A_B

        return T_world_B

    @staticmethod
    @overrides
    def subtract_local(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:

        T_world_A = x.reshape((3, 3))
        T_world_B = y.reshape((3, 3))

        R_world_A = T_world_A[:2, :2]
        t_world_A = T_world_A[:2, 2]

        T_A_world = jnp.eye(3)
        T_A_world = T_A_world.at[:2, :2].set(R_world_A.T)
        T_A_world = T_A_world.at[:2, 2].set(-R_world_A.T @ t_world_A)

        T_A_B = T_A_world @ T_world_B

        x = T_A_B[0, 2]
        y = T_A_B[1, 2]
        theta = jnp.arctan2(T_A_B[0, 1], T_A_B[0, 0])

        return jnp.array([x, y, theta])
