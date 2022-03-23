import itertools
from typing import Tuple

import jax_dataclasses as jdc
import jaxfg
import numpy as onp
from jax import numpy as jnp
from overrides import overrides

@jdc.pytree_dataclass
class VariableValue:
    scalar: jaxfg.hints.Scalar


class Variable_A(jaxfg.core.VariableBase[VariableValue]):
    @classmethod
    @overrides
    def get_default_value(cls) -> VariableValue:
        return VariableValue(scalar=1.0)


class Variable_B(jaxfg.core.VariableBase[VariableValue]):
    @classmethod
    @overrides
    def get_default_value(cls) -> VariableValue:
        return VariableValue(scalar=1.0)


VariableValueSingle = Tuple[VariableValue]
VariableValuePair = Tuple[VariableValue, VariableValue]


@jdc.pytree_dataclass
class PriorFactor(jaxfg.core.FactorBase[VariableValueSingle]):
    prior: jaxfg.hints.Scalar

    @staticmethod
    def make(variable: Variable_A, prior: jaxfg.hints.Scalar = 1.0) -> "PriorFactor":
        return PriorFactor(
            variables=(variable,),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(jnp.ones(1)),
            prior=prior,
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: VariableValueSingle
    ) -> jnp.ndarray:
        (variable,) = variable_values
        return jnp.array([variable.scalar - self.prior])


@jdc.pytree_dataclass
class PairFactor(jaxfg.core.FactorBase[VariableValuePair]):
    pair_value: jaxfg.hints.Scalar

    @staticmethod
    def make(
        variable1: Variable_A, variable2: Variable_B, pair_value: jaxfg.hints.Scalar
    ) -> "PairFactor":
        return PairFactor(
            variables=(variable1, variable2),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(jnp.ones(1)),
            pair_value=pair_value,
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: VariableValuePair
    ) -> jnp.ndarray:
        (v1, v2) = variable_values
        return jnp.array([(v1.scalar + v2.scalar) - self.pair_value])


def test_variable_assignments_permutation():
    ## Simple graph
    # [A==3]                        [A==7]
    #   |                             |
    #  (A)--[A+B==1]--(B)--[A+B==2]--(A)
    ##
    variables = [Variable_A(), Variable_A(), Variable_B()]
    graph = jaxfg.core.StackedFactorGraph.make(
        [
            PriorFactor.make(variables[0], prior=3),
            PriorFactor.make(variables[1], prior=7),
            PairFactor.make(variables[0], variables[2], pair_value=1),
            PairFactor.make(variables[1], variables[2], pair_value=2),
        ]
    )

    costs = []
    for variable_combination in itertools.permutations(variables, r=len(variables)):
        initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(
            variable_combination
        )

        solution_assignments = graph.solve(initial_assignments)
        cost = graph.compute_joint_nll(solution_assignments)
        costs.append(cost)

    costs = onp.array(costs)
    onp.testing.assert_allclose(
        costs.min(),
        costs.max(),
        atol=1e-8,
        rtol=1e-5,
    )
