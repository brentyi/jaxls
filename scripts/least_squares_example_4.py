# minimize L(x, y)
# where L = (x - 2)^2 / 10 + (y - 5)^2 / 2 + (x - y)^2 / 7
# analytical solution gives x = 68/19, y = 89/19

# we have the following factor graph:

#     ┌────────┐             ┌────────┐
#     │   x    ├───Between───┤    y   │
#     └───┬────┘             └────┬───┘
#         │                       │
#         │                       │
#       Prior_x (=2)           Prior_y (=5)

from jaxfg.core import VariableBase
import jaxfg
import jax_dataclasses
from jaxfg.core._factor_base import FactorBase
import jax.numpy as jnp
from typing import Generic, Tuple, TypeVar, NamedTuple
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

variables = [
    MyVariable(), # x
    MyVariable(), # y
]

PriorValueTuple = Tuple[RealValue]

@jax_dataclasses.pytree_dataclass
class MyFactor(FactorBase[PriorValueTuple]):

    prior_value: jnp.ndarray

    @overrides
    def compute_residual_vector(self, variable_values: PriorValueTuple) -> jnp.ndarray:
        return variable_values[0].value - self.prior_value

class BetweenValueTuple(NamedTuple):
    x: RealValue
    y: RealValue

@jax_dataclasses.pytree_dataclass
class MyBetweenFactor(FactorBase[BetweenValueTuple]):

    @overrides
    def compute_residual_vector(
        self, variable_values: BetweenValueTuple
    ) -> jnp.ndarray:
        x = variable_values.x
        y = variable_values.y

        return jnp.array([x.value - y.value])

factors = [
    MyFactor(
        prior_value=jnp.array([2.0]),
        variables=variables[0:1],
        noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance([10]),
    ),
    MyFactor(
        prior_value=jnp.array([5.0]),
        variables=variables[1:],
        noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance([2]),
    ),
    MyBetweenFactor(
        variables=variables,
        noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance([7]),
    )
]

graph = jaxfg.core.StackedFactorGraph.make(factors)

initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(
    variables
)

solution_assignments = graph.solve(initial_assignments)

print(f"solution for x: {solution_assignments.get_value(variables[0]).value}")
print(f"exact solution: {68/19}")

print(f"solution for y: {solution_assignments.get_value(variables[1]).value}")
print(f"exact solution: {89/19}")