# minimize (2 - x)^2 / 10 + (3 - x)^2 / 4
# -> x = 19/7

# Note that here we have more confidence for the prior=3 as for prior=2

# we have the following factor graph:

#     ┌────────┐
#     │   x    ├──────── Prior_x (=3)
#     └───┬────┘
#         │
#         │
#       Prior_x (=2)

from jaxfg.core import VariableBase
import jaxfg
import jax_dataclasses
from jaxfg.core._factor_base import FactorBase
import jax.numpy as jnp
from typing import Generic, Tuple, TypeVar
from overrides import final, overrides

T = TypeVar("T")

@jax_dataclasses.pytree_dataclass
class RealValue:
    value: jnp.ndarray

class MyVariable(Generic[T], VariableBase[T]):

    @classmethod
    @final
    @overrides
    def get_default_value(cls) -> T:
        return RealValue(1.0)

variables = [MyVariable()]

PriorValueTuple = Tuple[RealValue]

@jax_dataclasses.pytree_dataclass
class MyFactor(FactorBase[PriorValueTuple]):

    prior_value: jnp.ndarray

    @overrides
    def compute_residual_vector(self, variable_values: PriorValueTuple) -> jnp.ndarray:
        return variable_values[0].value - self.prior_value

factors = [
    MyFactor(
        prior_value=jnp.array([2.0]),
        variables=variables,
        noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance([10]),
    ),
    MyFactor(
        prior_value=jnp.array([3.0]),
        variables=variables,
        noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance([4]),
    ),
]

graph = jaxfg.core.StackedFactorGraph.make(factors)

initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(
    variables
)

solution_assignments = graph.solve(initial_assignments)

print(f"solution: {solution_assignments.get_value(variables[0]).value}")
print(f"exact solution: {19/7}")