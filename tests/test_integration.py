"""Simple pose graph tests. Could use cleanup/refactoring."""

from typing import Tuple

import jax_dataclasses as jdc
from jax import numpy as jnp
from overrides import overrides

import jaxfg


@jdc.pytree_dataclass
class VariableValue:
    a: jaxfg.hints.Array
    b: jaxfg.hints.Scalar


class Variable(jaxfg.core.VariableBase[VariableValue]):
    @classmethod
    @overrides
    def get_default_value(cls) -> VariableValue:
        return VariableValue(jnp.array([1.0, 2.0]), 3.0)


VariableValueSingle = Tuple[VariableValue]
VariableValuePair = Tuple[VariableValue, VariableValue]


@jdc.pytree_dataclass
class UniFactor(jaxfg.core.FactorBase[VariableValueSingle]):
    @staticmethod
    def make(variable: Variable) -> "UniFactor":
        return UniFactor(
            variables=(variable,),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(jnp.ones(4)),
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: VariableValueSingle
    ) -> jnp.ndarray:
        (variable,) = variable_values
        return jnp.array(
            [
                variable.a[0] - 2.0,
                variable.a[1] - 3.0,
                variable.b - 2.0,
                variable.b - 7.0,
            ]
        )


@jdc.pytree_dataclass
class BiFactor(jaxfg.core.FactorBase[VariableValuePair]):
    @staticmethod
    def make(variable1: Variable, variable2: Variable) -> "BiFactor":
        return BiFactor(
            variables=(variable1, variable2),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(jnp.ones(4)),
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: VariableValuePair
    ) -> jnp.ndarray:
        (v1, v2) = variable_values
        return jnp.array(
            [
                v1.a[0] - v2.a[1],
                v1.a[1] - v2.a[0],
                v1.b - v2.b,
                jnp.sqrt(jnp.abs(v2.b - 7.0)),
            ]
        )


def test_integration():
    """Make sure everything runs without exploding!"""

    variables = [Variable() for i in range(2)]
    graph = jaxfg.core.StackedFactorGraph.make(
        [
            UniFactor.make(variables[0]),
            BiFactor.make(variables[0], variables[1]),
        ]
    )

    initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(variables)
    solution_assignments = graph.solve(initial_assignments)

    assert graph.compute_joint_nll(initial_assignments) > graph.compute_joint_nll(
        solution_assignments
    )
