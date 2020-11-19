import dataclasses
from typing import Dict, Optional, Set, Tuple, cast

import jax
from jax import numpy as jnp
from overrides import overrides

from .. import _types as types
from .._factors import LinearFactor
from .._variables import RealVectorVariable
from ._factor_graph_base import FactorGraphBase


# @dataclasses.dataclass(frozen=True)
class LinearFactorGraph(FactorGraphBase[LinearFactor, RealVectorVariable]):
    """Simple LinearFactorGraph."""

    # Use default object hash rather than dataclass one
    __hash__ = object.__hash__

    def solve(
        self,
        initial_assignments: Optional[types.VariableAssignments] = None,
    ) -> types.VariableAssignments:
        """Finds a solution for our factor graph via an iterative conjugate gradient solver.

        Implicitly defines the normal equations `A.T @ A @ x = A.T @ b`.

        Args:
            initial_assignments (Optional[types.VariableAssignments]): Initial
                variable assignments.

        Returns:
            types.VariableAssignments: Best assignments.
        """

        variables = tuple(self.factors_from_variable.keys())

        def A_function(
            x: types.VariableAssignments,
        ) -> Dict[RealVectorVariable, jnp.ndarray]:
            """Left-multiplies a vector with our Hessian/information matrix.

            Args:
                x (types.VariableAssignments): x

            Returns:
                Dict[RealVectorVariable, jnp.ndarray]: `A^TAx`
            """
            # x => Apply Jacobian => Ax
            error_from_factor: Dict[LinearFactor, jnp.ndarray] = {
                factor: factor.compute_error_linear_component(assignments=x)
                for factor in self.factors
            }

            # Ax => Apply Jacobian-transpose => A^TAx
            value_from_variable: Dict[RealVectorVariable, jnp.ndarray] = {}
            for variable in variables:
                value_from_variable[variable] = variable.compute_error_dual(
                    self.factors_from_variable[variable], error_from_factor
                )

            return value_from_variable

        # Compute rhs (A.T @ b)
        b: Dict[RealVectorVariable, jnp.ndarray] = {
            variable: variable.compute_error_dual(self.factors_from_variable[variable])
            for variable in variables
        }

        assignments_solution, _unused_info = jax.scipy.sparse.linalg.cg(
            A=A_function, b=b, x0=initial_assignments
        )
        return assignments_solution
