from typing import Tuple, Type

import jaxlie
from jax import numpy as jnp
from overrides import overrides

from .. import types
from ..core._variables import VariableBase

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
        def subtract_local(x: Group, y: Group) -> types.LocalVariableValue:
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
