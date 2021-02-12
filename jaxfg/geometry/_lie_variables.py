from typing import Generic, Type, TypeVar, cast

import jaxlie
from overrides import overrides

from .. import types
from ..core._variables import VariableBase

T = TypeVar("T", bound=jaxlie.MatrixLieGroup)


class LieVariableBase(VariableBase[T], Generic[T]):
    MatrixLieGroupType: Type[T] = jaxlie.MatrixLieGroup
    """Lie group type."""

    @classmethod
    @overrides
    def get_local_parameter_dim(cls) -> int:
        return cls.MatrixLieGroupType.tangent_dim

    @classmethod
    @overrides
    def get_default_value(cls) -> T:
        return cast(T, cls.MatrixLieGroupType.identity())

    @staticmethod
    def manifold_retract(x: T, local_delta: jaxlie.types.TangentVector) -> T:
        return jaxlie.manifold.rplus(x, local_delta)

    @staticmethod
    @overrides
    def manifold_inverse_retract(x: T, y: T) -> types.LocalVariableValue:
        # x = world<-A, y = world<-B
        # Difference = A<-B
        return jaxlie.manifold.rminus(x, y)


class SO2Variable(LieVariableBase[jaxlie.SO2]):
    MatrixLieGroupType = jaxlie.SO2


class SE2Variable(LieVariableBase[jaxlie.SE2]):
    MatrixLieGroupType = jaxlie.SE2


class SO3Variable(LieVariableBase[jaxlie.SO3]):
    MatrixLieGroupType = jaxlie.SO3


class SE3Variable(LieVariableBase[jaxlie.SE3]):
    MatrixLieGroupType = jaxlie.SE3
