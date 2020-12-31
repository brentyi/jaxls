from typing import Tuple, Type, TypeVar

import jaxlie
from jax import numpy as jnp
from overrides import overrides

from .. import types
from ..core._variables import VariableBase

T = TypeVar("T", bound=jaxlie.MatrixLieGroup)


class LieVariableBase(VariableBase[T]):
    MatrixLieGroupType: Type[jaxlie.MatrixLieGroup] = jaxlie.MatrixLieGroup
    """Lie group type."""


def make_lie_variable(Group: Type[jaxlie.MatrixLieGroup]):
    class _LieVariable(LieVariableBase):
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
        def get_default_value() -> Group:
            return Group.identity()

        @staticmethod
        #  @jax.custom_jvp
        def add_local(x: Group, local_delta: jaxlie.types.TangentVector) -> Group:
            return jaxlie.manifold.rplus(x, local_delta)

        @staticmethod
        @overrides
        def subtract_local(x: Group, y: Group) -> types.LocalVariableValue:
            # x = world<-A, y = world<-B
            # Difference = A<-B
            return jaxlie.manifold.rminus(x, y)

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
