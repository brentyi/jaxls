import abc
from typing import Generic, Type, TypeVar, cast

import jaxlie
from overrides import final, overrides

from ..core._variables import VariableBase

T = TypeVar("T", bound=jaxlie.MatrixLieGroup)


class LieVariableBase(VariableBase[T], Generic[T]):
    @staticmethod
    @abc.abstractmethod
    def get_group_type() -> Type[T]:
        """Helper for defining Lie group types."""
        # return jaxlie.MatrixLieGroup  # type: ignore

    @classmethod
    @final
    @overrides
    def get_local_parameter_dim(cls) -> int:
        return cls.get_group_type().tangent_dim

    @classmethod
    @final
    @overrides
    def get_default_value(cls) -> T:
        return cast(T, cls.get_group_type().identity())

    @classmethod
    @final
    @overrides
    def manifold_retract(cls, x: T, local_delta: jaxlie.hints.TangentVector) -> T:
        return jaxlie.manifold.rplus(x, local_delta)

    @classmethod
    @final
    @overrides
    def manifold_retract_jacobian(cls, x: T) -> T:
        # `jaxlie` returns Jacobians as bare arrays, but emulating
        # jax.jacfwd/jax.jacrev requires each Jacobian to be a pytree.
        return cls.get_group_type()(
            jaxlie.manifold.rplus_jacobian_parameters_wrt_delta(x)
        )


class SO2Variable(LieVariableBase[jaxlie.SO2]):
    @staticmethod
    @overrides
    def get_group_type() -> Type[jaxlie.SO2]:
        return jaxlie.SO2


class SE2Variable(LieVariableBase[jaxlie.SE2]):
    @staticmethod
    @overrides
    def get_group_type() -> Type[jaxlie.SE2]:
        return jaxlie.SE2


class SO3Variable(LieVariableBase[jaxlie.SO3]):
    @staticmethod
    @overrides
    def get_group_type() -> Type[jaxlie.SO3]:
        return jaxlie.SO3


class SE3Variable(LieVariableBase[jaxlie.SE3]):
    @staticmethod
    @overrides
    def get_group_type() -> Type[jaxlie.SE3]:
        return jaxlie.SE3
