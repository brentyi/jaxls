import abc
from typing import Generic, Type, TypeVar, cast

import jaxlie
from overrides import final, overrides

from .. import types
from ..core._variables import VariableBase

T = TypeVar("T", bound=jaxlie.MatrixLieGroup)


class LieVariableBase(VariableBase[T], Generic[T]):
    @staticmethod
    @abc.abstractmethod
    def get_group_type() -> Type[T]:
        """Helper for defining Lie group types."""
        return jaxlie.MatrixLieGroup

    @classmethod
    @overrides
    def get_local_parameter_dim(cls) -> int:
        return cls.get_group_type().tangent_dim

    @classmethod
    @overrides
    def get_default_value(cls) -> T:
        return cast(T, cls.get_group_type().identity())

    @staticmethod
    @overrides
    def manifold_retract(x: T, local_delta: jaxlie.types.TangentVector) -> T:
        return jaxlie.manifold.rplus(x, local_delta)

    @staticmethod
    @overrides
    def manifold_inverse_retract(x: T, y: T) -> types.LocalVariableValue:
        # x = world<-A, y = world<-B
        # Difference = A<-B
        return jaxlie.manifold.rminus(x, y)


class SO2Variable(LieVariableBase[jaxlie.SO2]):
    @staticmethod
    @final
    @overrides
    def get_group_type() -> Type[jaxlie.SO2]:
        return jaxlie.SO2


class SE2Variable(LieVariableBase[jaxlie.SE2]):
    @staticmethod
    @final
    @overrides
    def get_group_type() -> Type[jaxlie.SE2]:
        return jaxlie.SE2


class SO3Variable(LieVariableBase[jaxlie.SO3]):
    @staticmethod
    @final
    @overrides
    def get_group_type() -> Type[jaxlie.SO3]:
        return jaxlie.SO3


class SE3Variable(LieVariableBase[jaxlie.SE3]):
    @staticmethod
    @final
    @overrides
    def get_group_type() -> Type[jaxlie.SE3]:
        return jaxlie.SE3
