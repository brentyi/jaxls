from typing import Generic, Type, TypeVar, cast

import jaxlie
from jax import numpy as jnp
from overrides import overrides

from .. import types
from ..core._variables import VariableBase

T = TypeVar("T", bound=jaxlie.MatrixLieGroup)


class LieVariableBase(VariableBase[T], Generic[T]):
    MatrixLieGroupType: Type[T]
    """Lie group type."""


def make_lie_variable(Group: Type[T]):
    class _LieVariable(LieVariableBase[T]):
        """Variable containing a transformation."""

        MatrixLieGroupType = Group

        @staticmethod
        @overrides
        def get_parameter_dim() -> int:
            return Group.parameters_dim

        @staticmethod
        @overrides
        def get_local_parameter_dim() -> int:
            return Group.tangent_dim

        @staticmethod
        @overrides
        def get_default_value() -> T:
            return cast(T, Group.identity())

        @staticmethod
        def manifold_retract(x: T, local_delta: jaxlie.types.TangentVector) -> T:
            return jaxlie.manifold.rplus(x, local_delta)

        @staticmethod
        @overrides
        def manifold_inverse_retract(x: T, y: T) -> types.LocalVariableValue:
            # x = world<-A, y = world<-B
            # Difference = A<-B
            return jaxlie.manifold.rminus(x, y)

        @staticmethod
        @overrides
        def flatten(x: T) -> jnp.ndarray:
            return x.parameters

        @staticmethod
        @overrides
        def unflatten(flat: jnp.ndarray) -> T:
            return Group(flat)

    return _LieVariable


SO2Variable = make_lie_variable(jaxlie.SO2)
SE2Variable = make_lie_variable(jaxlie.SE2)
SO3Variable = make_lie_variable(jaxlie.SO3)
SE3Variable = make_lie_variable(jaxlie.SE3)
