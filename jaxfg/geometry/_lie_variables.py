import abc
from typing import ClassVar, Generic, Type, TypeVar, cast

import jaxlie
from overrides import final, overrides

from ..core._variables import VariableBase

T = TypeVar("T", bound=jaxlie.MatrixLieGroup)


class LieVariableBase(Generic[T], VariableBase[T]):
    """Variable definition for Lie groups."""

    # Group type to be set in subclasses. This is a method instead of an attribute so
    # the class can be properly marked as abstract.
    @staticmethod
    @abc.abstractmethod
    def get_group_type() -> Type[T]:
        pass

    # (1) Required for all variables: an example value. This is most critically used to
    # generate functions for flattening and unflattening values.

    @classmethod
    @final
    @overrides
    def get_default_value(cls) -> T:
        return cast(T, cls.get_group_type().identity())

    # (2) Methods that need to be overridden for defining non-Euclidean manifolds.

    @classmethod
    @final
    @overrides
    def get_local_parameter_dim(cls) -> int:
        return cls.get_group_type().tangent_dim

    @classmethod
    @final
    @overrides
    def manifold_retract(cls, x: T, local_delta: jaxlie.hints.TangentVector) -> T:
        return jaxlie.manifold.rplus(x, local_delta)

    # (3) Optional: analytical Jacobian for manifold retraction. If not defined, this
    # will be handled via autodiff.

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
