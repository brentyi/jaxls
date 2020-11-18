import abc
from typing import Dict, Hashable, Tuple, TYPE_CHECKING

from jax import numpy as jnp

from . import _types as types

if TYPE_CHECKING:
    from . import RealVectorVariable, VariableBase


class FactorBase(abc.ABC):
    def __init__(
        self, variables: Tuple["VariableBase", ...], scale_tril_inv: types.ScaleTrilInv
    ):
        self.variables = variables
        """Variables connected to this factor. Immutable. (currently assumed but unenforced)"""

        self.scale_tril_inv = scale_tril_inv
        """Inverse square root of covariance matrix for residual term.
        State dimensionality should match local parameterizations/manifolds."""

        self.error_dim: int = scale_tril_inv.shape[0]
        """Error dimensionality."""

    # def group_key(self) -> Hashable:
    #     """Key used for determining which factors can be computed in parallel.
    #
    #     Returns:
    #         Hashable: Key to organize factors by.
    #     """
    #     return self


class LinearFactor(FactorBase):
    """Linearized factor, with the simple residual:
    $$
    r = ( \Sum_i A_i x_i ) - b_i
    $$
    """

    def __init__(
        self,
        A_from_variable: Dict["RealVectorVariable", jnp.ndarray],
        b: jnp.ndarray,
        scale_tril_inv: types.ScaleTrilInv,
    ):

        variables = tuple(A_from_variable.keys())
        super().__init__(variables=variables, scale_tril_inv=scale_tril_inv)

        self.variables: Tuple["RealVectorVariable"]
        self.A_from_variable = A_from_variable
        self.b = b

    def compute_error(self, assignments: types.VariableAssignments):
        """Compute error vector.

        Args:
            assignments (types.VariableAssignments): assignments
        """

        error = jnp.zeros(self.error_dim)
        for variable, A in self.A_from_variable.items():
            error = error + A @ assignments[variable]
        return error
