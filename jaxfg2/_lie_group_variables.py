import jax_dataclasses as jdc
import jaxlie

from ._variables import Var


class SO2Var(
    Var[jaxlie.SO2],
    default=jaxlie.SO2.identity(),
    tangent_dim=jaxlie.SO2.tangent_dim,
    retract_fn=jaxlie.manifold.rplus,
):
    ...


class SO3Var(
    Var[jaxlie.SO3],
    default=jaxlie.SO3.identity(),
    tangent_dim=jaxlie.SO3.tangent_dim,
    retract_fn=jaxlie.manifold.rplus,
):
    ...


class SE2Var(
    Var[jaxlie.SE2],
    default=jaxlie.SE2.identity(),
    tangent_dim=jaxlie.SE2.tangent_dim,
    retract_fn=jaxlie.manifold.rplus,
):
    ...


class SE3Var(
    Var[jaxlie.SE3],
    default=jaxlie.SE3.identity(),
    tangent_dim=jaxlie.SE3.tangent_dim,
    retract_fn=jaxlie.manifold.rplus,
):
    ...
