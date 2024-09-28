import jaxlie

from ._variables import Var


class SO2Var(
    Var[jaxlie.SO2],
    default_factory=jaxlie.SO2.identity,
    retract_fn=jaxlie.manifold.rplus,
    tangent_dim=jaxlie.SO2.tangent_dim,
): ...


class SO3Var(
    Var[jaxlie.SO3],
    default_factory=jaxlie.SO3.identity,
    retract_fn=jaxlie.manifold.rplus,
    tangent_dim=jaxlie.SO3.tangent_dim,
): ...


class SE2Var(
    Var[jaxlie.SE2],
    default_factory=jaxlie.SE2.identity,
    retract_fn=jaxlie.manifold.rplus,
    tangent_dim=jaxlie.SE2.tangent_dim,
): ...


class SE3Var(
    Var[jaxlie.SE3],
    default_factory=jaxlie.SE3.identity,
    retract_fn=jaxlie.manifold.rplus,
    tangent_dim=jaxlie.SE3.tangent_dim,
): ...
