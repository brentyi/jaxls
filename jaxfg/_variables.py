import abc
import contextlib
from typing import TYPE_CHECKING, Dict, Generator, Optional, Set, Tuple

import jax
from jax import numpy as jnp
from overrides import overrides

if TYPE_CHECKING:
    from . import LinearFactor


class VariableBase(abc.ABC):
    def __init__(self, parameter_dim: int, local_parameter_dim: int):
        self.parameter_dim = parameter_dim
        """Dimensionality of underlying parameterization."""

        self.local_parameter_dim = local_parameter_dim
        """Dimensionality of local parameterization."""

        self.local_delta_variable: RealVectorVariable
        """Variable for tracking local updates."""
        if isinstance(self, RealVectorVariable):
            self.local_delta_variable = self
        else:
            self.local_delta_variable = RealVectorVariable(local_parameter_dim)

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
            x (jnp.ndarray): First parameter to compare. Shape should match `self.parameter_dim`.
            y (jnp.ndarray): Second parameter to compare. Shape should match `self.parameter_dim`.

        Returns:
            jnp.ndarray: Delta vector; dimension should match self.local_delta_variable.
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


class RealVectorVariable(VariableBase):
    """Variable for an arbitrary vector of real numbers."""

    def __init__(self, parameter_dim):
        super().__init__(parameter_dim=parameter_dim, local_parameter_dim=parameter_dim)

    def compute_error_dual(
        self,
        factors: Set["LinearFactor"],
        error_from_factor: Optional[Dict["LinearFactor", jnp.ndarray]] = None,
    ):
        """Compute dual of error term; eg the terms of `A.T @ error` that correspond to
        this variable.

        Args:
            factors (Set["LinearFactor"]): Linearized factors that are attached to this variable.
            error_from_factor (Dict["LinearFactor", jnp.ndarray]): Mapping from factor to error term.
                Defaults to the `b` constant from each factor.
        """
        dual = jnp.zeros(self.parameter_dim)
        if error_from_factor is None:
            for factor in factors:
                dual = dual + factor.A_transpose_from_variable[self](factor.b)[0]
        else:
            for factor in factors:
                dual = (
                    dual
                    + factor.A_transpose_from_variable[self](error_from_factor[factor])[
                        0
                    ]
                )
        return dual

    @classmethod
    @overrides
    def add_local(cls, x: jnp.ndarray, local_delta: jnp.ndarray) -> jnp.ndarray:
        return x + local_delta

    @classmethod
    @overrides
    def subtract_local(cls, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return x - y


class SO2Variable(VariableBase):
    """Variable containing a 2D rotation."""

    def __init__(self):
        super().__init__(
            parameter_dim=2,  # Parameterized with unit-norm complex number
            local_parameter_dim=1,  # Local delta with scalar radians
        )

    @classmethod
    @overrides
    def add_local(cls, x: jnp.ndarray, local_delta: jnp.ndarray) -> jnp.ndarray:
        assert x.shape == (2,) and local_delta.shape == (1,)
        theta = local_delta[0]
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return jnp.array([cos * x[0] - sin * x[1], sin * x[0] + cos * x[1]])

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

    def __init__(self):
        super().__init__(
            parameter_dim=4,  # (x, y, cos, sin)
            local_parameter_dim=3,  # (x, y, theta)
        )

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
