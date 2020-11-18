import dataclasses
import types
from typing import Dict, Optional, Set, Tuple, cast

import jax
from jax import numpy as jnp
from overrides import overrides

from .. import _types as types
from .._factors import LinearFactor
from .._variables import RealVectorVariable, VariableBase
from ._factor_graph_base import FactorGraphBase
from ._linear_factor_graph import LinearFactorGraph


@dataclasses.dataclass(frozen=True)
class FactorGraph(FactorGraphBase):
    """General nonlinear factor graph."""

    # More specific typing for factors and variables
    factors: Set[LinearFactor] = dataclasses.field(
        default_factory=lambda: set(), init=False
    )
    factors_from_variable: Dict[VariableBase, Set[LinearFactor]] = dataclasses.field(
        default_factory=lambda: {}, init=False
    )

    # Use default object hash rather than dataclass one
    __hash__ = object.__hash__

    @overrides
    def with_factors(self, *to_add: LinearFactor) -> "FactorGraph":
        return cast(FactorGraph, super().with_factors(*to_add))

    @overrides
    def without_factors(self, *to_remove: LinearFactor) -> "FactorGraph":
        return cast(FactorGraph, super().without_factors(*to_remove))

    def solve(
        self,
        initial_assignments: Optional[types.VariableAssignments] = None,
    ) -> types.RealVectorVariableAssignments:

        variables = self.factors_from_variable.keys()

        # Define variables for local perturbations
        variable: VariableBase
        delta_variables = tuple(variable.local_delta_variable for variable in variables)

        # Sort out initial variable assignments
        assignments: types.VariableAssignments
        if initial_assignments is None:
            assignments = {
                variable: jnp.zeros(variable.parameter_dim) for variable in variables
            }
        else:
            assignments = initial_assignments

        # Run some Gauss-Newton iterations
        # for i in range(10):
        print(assignments)
        for i in range(10):
            assignments = self._gauss_newton_step(assignments, delta_variables)

        return assignments

    @jax.partial(jax.jit, static_argnums=(0, 2))
    def _gauss_newton_step(
        self,
        assignments: types.VariableAssignments,
        delta_variables: Tuple[RealVectorVariable, ...],
    ) -> types.VariableAssignments:
        # Linearize factors
        linearized_factors = [
            LinearFactor.linearize_from_factor(factor, assignments)
            for factor in self.factors
        ]

        # Solve for deltas
        delta_from_variable: types.RealVectorVariableAssignments = (
            LinearFactorGraph().with_factors(*linearized_factors).solve()
        )

        # Update assignments
        assignments = {
            variable: variable.retract(delta_from_variable[variable], value)
            for variable, value in assignments.items()
        }
        return assignments
