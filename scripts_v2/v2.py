import jax
import jaxlie
from jaxfg2 import Factor, StackedFactorGraph, Var, VarValues


class SO3Var(
    Var[jaxlie.SO3],
    default=jaxlie.SO3.identity(),
    tangent_dim=jaxlie.SO3.tangent_dim,
    retract_fn=jaxlie.manifold.rplus,
):
    ...


var0 = SO3Var(0)
var1 = SO3Var(1)
var2 = SO3Var(2)


def compute_prior(values: VarValues, var_x: SO3Var, var_y: SO3Var) -> jax.Array:
    x = values[var_x]
    y = values[var_y]
    return (x.inverse() @ y.inverse()).log()


factors = [
    Factor.make(compute_prior, args=(var0, var1)),
]
