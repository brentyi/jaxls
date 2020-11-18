import abc
import types
from typing import TYPE_CHECKING, Callable, Dict, Hashable, Tuple

import jax
from jax import numpy as jnp
from overrides import overrides

from . import _types as _types

if TYPE_CHECKING:
    from . import RealVectorVariable, VariableBase


class FactorBase(abc.ABC):
    def __init__(
        self,
        variables: Tuple["VariableBase", ...],
        error_dim: int,
        scale_tril_inv: _types.ScaleTrilInv,
    ):
        self.variables = variables
        """Variables connected to this factor. Immutable. (currently assumed but unenforced)"""

        self.error_dim: int = error_dim
        """Error dimensionality."""

        self.scale_tril_inv = scale_tril_inv
        """Inverse square root of covariance matrix."""

    @abc.abstractmethod
    def compute_error(self, assignments: _types.VariableAssignments):
        """Compute error vector.

        Args:
            assignments (_types.VariableAssignments): assignments
        """


class LinearFactor(FactorBase):
    """Linearized factor, corresponding to the simple residual:
    $$
    r = ( \Sum_i A_i x_i ) - b_i
    $$
    """

    def __init__(
        self,
        A_from_variable: Dict[
            "RealVectorVariable", Callable[[jnp.ndarray], jnp.ndarray]
        ],
        b: jnp.ndarray,
    ):

        variables = tuple(A_from_variable.keys())
        error_dim = b.shape[0]
        super().__init__(
            variables=variables, error_dim=error_dim, scale_tril_inv=jnp.eye(error_dim)
        )

        self.variables: Tuple["RealVectorVariable"]
        self.A_from_variable = A_from_variable

        primal = b
        self.A_transpose_from_variable = {
            variable: jax.linear_transpose(A, primal)
            for variable, A in A_from_variable.items()
        }
        self.b = b

    @classmethod
    def linearize_from_factor(
        cls, factor: FactorBase, assignments: _types.VariableAssignments
    ):
        A_from_variable: Dict["RealVectorVariable"] = {}

        # Pull out only the assignments that we care about
        assignments = {variable: assignments[variable] for variable in factor.variables}

        for variable in factor.variables:

            def f(new_value: jnp.ndarray) -> jnp.ndarray:
                # Make copy of assignments with updated variable value
                assignments_copy = assignments.copy()
                assignments_copy[variable] = new_value

                # Return error
                return factor.compute_error(assignments_copy)

            # Linearize around variable
            f_jvp = jax.linearize(f, assignments[variable])[1]
            A_from_variable[variable.local_delta_variable] = f_jvp

        error = factor.compute_error(assignments=assignments)
        return LinearFactor(
            A_from_variable=A_from_variable, b=-factor.scale_tril_inv @ error
        )

    @overrides
    def compute_error(self, assignments: _types.VariableAssignments):
        return self.compute_error_linear_component(assignments=assignments) - self.b

    def compute_error_linear_component(self, assignments: _types.VariableAssignments):
        error = jnp.zeros(self.error_dim)
        for variable, A in self.A_from_variable.items():
            error = error + A(assignments[variable])
        return error
