import abc
from typing import Dict, Generator, Hashable, List, Optional, Set, Tuple

import jax
from jax import numpy as jnp
from overrides import overrides

from . import _types as types
from ._factors import FactorBase, LinearFactor
from ._variables import RealVectorVariable, VariableBase


class FactorGraphBase(abc.ABC):
    def __init__(self):
        # self.factors: Dict[Hashable, List[FactorBase]] = {}
        # """Container for storing factors in our graph. Note that factors are organized
        # by a group key, which dictates whether they can be computed in parallel or not."""
        self.factors: Set[FactorBase] = set()

        self.factors_from_variable: Dict[VariableBase, Set[FactorBase]] = {}
        """Container for tracking variables in our graph + what factors they're
        connected to."""

    def add_factors(self, *to_add: FactorBase) -> None:
        """Add factor(s) to our graph."""

        for factor in to_add:
            ## Grouped factor logi
            # group_key = factor.group_key()
            # # Create group if it doesn't exist
            # if group_key not in self.factors:
            #     self.factors[group_key] = []

            # # Add factor to group
            # group = self.factors[group_key]
            # assert factor not in group, "Factor already added!"
            # group.append(factor)
            assert factor not in self.factors
            self.factors.add(factor)

            # Add variables to graph
            for v in factor.variables:
                if v not in self.factors_from_variable:
                    self.factors_from_variable[v] = set()
                self.factors_from_variable[v].add(factor)

    def remove_factors(self, *to_remove: FactorBase) -> None:
        """Remove factor(s) from our graph."""

        for factor in to_remove:
            ## Grouped factor logi
            # group_key = factor.group_key()
            # # Remove factor from group
            # group = self.factors[group_key]
            # assert factor in group, "Factor not in graph!"
            # group.remove(factor)
            #
            # # Remove group if empty
            # if len(group) == 0:
            #     self.factors.pop(group_key)
            assert factor in self.factors
            self.factors.remove(factor)

            # Remove variables from graph
            for v in factor.variables:
                self.factors_from_variable[v].remove(factor)
                if len(self.factors_from_variable[v]) == 0:
                    self.factors_from_variable.pop(v)


class LinearFactorGraph(FactorGraphBase):
    def __init__(self):
        # More specific typing for factors and varibles
        self.factors: Set[LinearFactor] = set()
        self.factors_from_variable: Dict[RealVectorVariable, Set[LinearFactor]] = {}

    @overrides
    def add_factors(self, *to_add: LinearFactor) -> None:
        super().add_factors(*to_add)

    @overrides
    def remove_factors(self, *to_remove: LinearFactor) -> None:
        super().remove_factors(*to_remove)

    def solve(
        self,
        variables: Tuple[RealVectorVariable],
        initial_assignments: Optional[types.RealVectorVariableAssignments] = None,
    ) -> types.RealVectorVariableAssignments:

        # TODO: error whitening

        b: Dict[RealVectorVariable, jnp.ndarray] = {}
        for variable in variables:
            b[variable] = variable.compute_error_dual(
                self.factors_from_variable[variable]
            )

        def A_function(x: types.RealVectorVariableAssignments):
            # x => Apply Jacobian => Ax
            error_from_factor: Dict[Hashable, jnp.ndarray] = {
                factor: factor.compute_error(assignments=x) for factor in self.factors
            }

            # Ax => Apply Jacobian-transpose => A^TAx
            value_from_variable: Dict[RealVectorVariable, jnp.ndarray] = {}
            factor: LinearFactor
            for variable in variables:
                value_from_variable[variable] = variable.compute_error_dual(
                    self.factors_from_variable[variable], error_from_factor
                )

            return value_from_variable

        assignments_solution, _unused_info = jax.scipy.sparse.linalg.cg(
            A=A_function, b=b, x0=initial_assignments
        )
        return assignments_solution

        # # Unfinished old code that uses a single vector for all parameters
        #
        # # Figure out some dimensions & indices
        # parameter_dim = 0
        # split_indices: List[int] = [0]
        # for v in variables:
        #     parameter_dim += v.parameter_dim
        #     split_indices.append(parameter_dim)
        #
        # # Get initial x0, b values naively
        # # - Note that this would be trivial to write with `jax.lax.fori_loop`, but we'd
        # #   then lose reverse-mode differentiability
        # # - We can probably implement this with `jax.lax.scan`? Maybe?
        # x0 = jnp.zeros(parameter_dim)
        # b = jnp.zeros(parameter_dim)
        # for i, v in enumerate(variables):
        #     indices = slice(index_offset, index_offset + v.parameters.shape[0])
        #
        #     # Update x0
        #     x0 = x0.at[indices].set(v.parameters)
        #
        #     # b for CGLS should be A.T @ b
        #     factor: LinearFactor
        #     for factor in self.factors_from_variable[v]:
        #         A = factor.A_from_variable[v]
        #         b = b.at[indices].add(A.T @ factor.b)
        #
        # # Define function for computing A.T @ A @ x
        # def A_function(x: Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray]:
        #     """compute_error_vectors.
        #
        #     Args:
        #         x (Tuple[jnp.ndarray]): Variables.
        #
        #     Returns:
        #         Tuple[jnp.ndarray]: A^TAx
        #     """
        #
        #     def compute_A_body(carry, x_row):
        #         # carry is (factor #, offset)
        #
        #         return carry, x
        #
        #     Ax = jax.lax.scan(
        #         f=compute_Ax,
        #         init=jnp.zeros(2),
        #         xs=x,
        #         length=parameter_dim,
        #     )
        #     ATAx = None
        #     return ATAx
        #
        # # Solve least squares problem via CGLS
        # x_solution, info = jax.scipy.sparse.linalg.cg(A=A_function, b=b, x0=x0)
        #
        # # Update variables
        # for i, v in enumerate(variables):
        #     v.parameters = x_solution[split_indices[i] : split_indices[i + 1]]
