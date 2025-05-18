from typing import Any
import jaxlie

from ._variables import Var


class SO2Var(
    Var[Any], default_factory=jaxlie.SO2.identity,
    retract_fn=jaxlie.manifold.rplus,
    tangent_dim=jaxlie.SO2.tangent_dim,
):
    @classmethod
    def __class_getitem__(cls, params):
        return cls
    ...


class SO3Var(
    Var[Any], default_factory=jaxlie.SO3.identity,
    retract_fn=jaxlie.manifold.rplus,
    tangent_dim=jaxlie.SO3.tangent_dim,
):
    @classmethod
    def __class_getitem__(cls, params):
        return cls
    ...


class SE2Var(
    Var[Any], default_factory=jaxlie.SE2.identity,
    retract_fn=jaxlie.manifold.rplus,
    tangent_dim=jaxlie.SE2.tangent_dim,
):
    @classmethod
    def __class_getitem__(cls, params):
        return cls
    ...


class SE3Var(
    Var[Any], default_factory=jaxlie.SE3.identity,
    retract_fn=jaxlie.manifold.rplus,
    tangent_dim=jaxlie.SE3.tangent_dim,
):
    @classmethod
    def __class_getitem__(cls, params):
        return cls
    ...
