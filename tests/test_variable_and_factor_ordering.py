import itertools
from typing import Tuple

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp
from overrides import overrides

import jaxfg

ScalarArray = jnp.ndarray


class VariableA(jaxfg.core.VariableBase[ScalarArray]):
    @classmethod
    @overrides
    def get_default_value(cls) -> ScalarArray:
        return jnp.array(1.0)


class VariableB(jaxfg.core.VariableBase[ScalarArray]):
    @classmethod
    @overrides
    def get_default_value(cls) -> ScalarArray:
        return jnp.array(1.0)


ScalarValueSingle = Tuple[ScalarArray]
ScalarPair = Tuple[ScalarArray, ScalarArray]


@jdc.pytree_dataclass
class PriorFactor(jaxfg.core.FactorBase[ScalarValueSingle]):
    prior: jaxfg.hints.Scalar

    @staticmethod
    def make(variable: VariableA, prior: jaxfg.hints.Scalar = 1.0) -> "PriorFactor":
        return PriorFactor(
            variables=(variable,),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(jnp.ones(1)),
            prior=prior,
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: ScalarValueSingle
    ) -> jnp.ndarray:
        (variable,) = variable_values
        return jnp.array([variable - self.prior])


@jdc.pytree_dataclass
class PairFactor(jaxfg.core.FactorBase[ScalarPair]):
    pair_value: jaxfg.hints.Scalar

    @staticmethod
    def make(
        variable1: VariableA, variable2: VariableB, pair_value: jaxfg.hints.Scalar
    ) -> "PairFactor":
        return PairFactor(
            variables=(variable1, variable2),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(jnp.ones(1)),
            pair_value=pair_value,
        )

    @overrides
    def compute_residual_vector(self, variable_values: ScalarPair) -> jnp.ndarray:
        (v1, v2) = variable_values
        return jnp.array([(v1 + v2) - self.pair_value])


def test_variable_and_factor_ordering():
    """Make sure that MAP inference is invariant to ordering of variables or factors."""

    # Simple graph:
    #     [A==3]                        [A==7]
    #       |                             |
    #      (A)--[A+B==1]--(B)--[A+B==2]--(A)
    variables = [VariableA(), VariableA(), VariableB()]
    factors = [
        PriorFactor.make(variables[0], prior=3),
        PriorFactor.make(variables[1], prior=7),
        PairFactor.make(variables[0], variables[2], pair_value=1),
        PairFactor.make(variables[1], variables[2], pair_value=2),
    ]

    solutions = []
    solver = jaxfg.solvers.GaussNewtonSolver(verbose=False)
    for factor_combination, variable_combination in itertools.product(
        itertools.permutations(factors, r=len(factors)),
        (variables, [variables[1], variables[0], variables[2]]),
    ):
        graph = jaxfg.core.StackedFactorGraph.make(factor_combination)
        initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(
            variable_combination
        )

        solution_assignments = graph.solve(initial_assignments, solver=solver)
        solutions.append(solution_assignments)

    for s0, s1 in zip(solutions[:-1], solutions[1:]):
        for v in variables:
            onp.testing.assert_allclose(
                s0.get_value(v),
                s1.get_value(v),
                atol=1e-8,
                rtol=1e-5,
            )
