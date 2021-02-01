from typing import Generic, Type, TypeVar, cast

import jaxlie
from overrides import overrides

from .. import types
from ..core._variables import VariableBase, concrete_example

T = TypeVar("T", bound=jaxlie.MatrixLieGroup)


class LieVariableBase(VariableBase[T], Generic[T]):
    MatrixLieGroupType: Type[T]
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


@concrete_example(jaxlie.SO2.identity())
class SO2Variable(LieVariableBase[jaxlie.SO2]):
    MatrixLieGroupType = jaxlie.SO2


@concrete_example(jaxlie.SE2.identity())
class SE2Variable(LieVariableBase[jaxlie.SE2]):
    MatrixLieGroupType = jaxlie.SE2


@concrete_example(jaxlie.SO3.identity())
class SO3Variable(LieVariableBase[jaxlie.SO3]):
    MatrixLieGroupType = jaxlie.SO3


@concrete_example(jaxlie.SE3.identity())
class SE3Variable(LieVariableBase[jaxlie.SE3]):
    MatrixLieGroupType = jaxlie.SE3
