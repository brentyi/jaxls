import dataclasses
from typing import Dict, Optional, Set, Tuple, cast

import jax
import numpy as onp
from jax import numpy as jnp
from overrides import overrides

from .. import _types as types
from .._factors import LinearFactor
from .._variables import RealVectorVariable, VariableBase
from ._factor_graph_base import FactorGraphBase


# @dataclasses.dataclass(frozen=True)
class LinearFactorGraph(FactorGraphBase[LinearFactor, RealVectorVariable]):
    """Simple LinearFactorGraph."""

    # Use default object hash rather than dataclass one
    __hash__ = object.__hash__

    @jax.partial(jax.jit, static_argnums=(0,))
    def new_solve(
        self,
        initial_assignments: Optional[types.VariableAssignments] = None,
    ):
        # Make default initial assignments if unavailable
        if initial_assignments is None:
            initial_assignments = types.VariableAssignments.create_default(
                self.variables
            )
        assert set(initial_assignments.variables) == set(self.variables)

        # Get A matrices + indices from values, errors
        A_matrices_from_shape = {}
        value_indices_from_shape = {}
        error_indices_from_shape = {}
        b_list = []
        error_index = 0
        for group_key, group in self.factors.items():
            for factor in group:
                for variable, A_matrix in factor.A_from_variable.items():
                    A_shape = A_matrix.shape

                    if A_shape not in A_matrices_from_shape:
                        A_matrices_from_shape[A_shape] = []
                        value_indices_from_shape[A_shape] = []
                        error_indices_from_shape[A_shape] = []

                    A_matrices_from_shape[A_shape].append(A_matrix)
                    value_indices_from_shape[A_shape].append(
                        onp.arange(variable.parameter_dim)
                        + initial_assignments.storage_pos_from_variable[variable]
                    )
                    error_indices_from_shape[A_shape].append(
                        onp.arange(factor.error_dim) + error_index
                    )
                    error_index += factor.error_dim

                b_list.append(factor.b)

        A_matrices_from_shape = {
            k: onp.array(v) for k, v in A_matrices_from_shape.items()
        }
        value_indices_from_shape = {
            k: onp.array(v) for k, v in value_indices_from_shape.items()
        }
        error_indices_from_shape = {
            k: onp.array(v) for k, v in error_indices_from_shape.items()
        }
        b: onp.ndarray = onp.concatenate(b_list)

        # Solve least squares
        solution_storage = self._solve(
            A_matrices_from_shape=A_matrices_from_shape,
            value_indices_from_shape=value_indices_from_shape,
            error_indices_from_shape=error_indices_from_shape,
            initial_x=initial_assignments.storage,
            b=b,
        )

        # Return new assignment mapping
        return types.VariableAssignments(
            storage=solution_storage,
            storage_pos_from_variable=initial_assignments.storage_pos_from_variable,
        )

    @classmethod
    def _solve(
        cls,
        A_matrices_from_shape: Dict[Tuple[int, int], jnp.ndarray],
        value_indices_from_shape: Dict[Tuple[int, int], jnp.ndarray],
        error_indices_from_shape: Dict[Tuple[int, int], jnp.ndarray],
        initial_x: jnp.ndarray,
        b: jnp.ndarray,
    ):
        """Solves a block-sparse `Ax = b` least squares problem via CGLS.


        How do we pass in the sparse A matrix?
            Standard sparse: 1D vector of values, row indices, col indices
              Pros:
                - Super easy to take matrix product
                - Simple simple simple w/ integer indexing
              Cons:
                - We know the matrix is block-sparse, a lot of unnecessary indices
                - ^That's not a huge con, asymptotic memory usage is still linear

            Dictionary: (block shape) => dense blocks, row indices, col indices
              Pros:
                - Much fewer indices (linear vs quadratic)
                - Einsum: faster than pure integer indexing?
              Cons:
                - Complexity?
                - Loops when we have many Jacobian shapes

        """
        assert len(b.shape) == 1, "b should be 1D!"

        def ATA_function(x: jnp.ndarray):

            error_dim = b.shape[0]

            # Compute Ax
            Ax: jnp.ndarray = jnp.zeros(error_dim)
            for shape, A_matrices in A_matrices_from_shape.items():
                # Get indices
                value_indices = value_indices_from_shape[shape]
                error_indices = error_indices_from_shape[shape]

                # Check shapes
                num_factors, error_dim, variable_dim = A_matrices.shape
                assert shape == (
                    error_dim,
                    variable_dim,
                ), "Incorrect `A` matrix dimension"
                assert value_indices.shape == (num_factors, variable_dim)
                assert error_indices.shape == (num_factors, error_dim)

                # Batched matrix multiply
                # (f) num factors, (e) error dim, (v) variable dim
                Ax = Ax.at[error_indices].add(
                    jnp.einsum("fev,fv->fe", A_matrices, x[value_indices])
                )

            # Compute A^TAx
            ATAx: jnp.ndarray = jnp.zeros_like(x)
            for shape, A_matrices in A_matrices_from_shape.items():
                # Get indices
                value_indices = value_indices_from_shape[shape]
                error_indices = error_indices_from_shape[shape]

                # (f) num factors, (e) error dim, (v) variable dim
                ATAx = ATAx.at[value_indices].add(
                    jnp.einsum("fev,fe->fv", A_matrices, Ax[error_indices])
                )

            return ATAx

        # Compute ATb
        ATb: jnp.ndarray = jnp.zeros_like(initial_x)
        for shape, A_matrices in A_matrices_from_shape.items():
            # Get indices
            value_indices = value_indices_from_shape[shape]
            error_indices = error_indices_from_shape[shape]

            # (f) num factors, (e) error dim, (v) variable dim
            ATb = ATb.at[value_indices].add(
                jnp.einsum("fev,fe->fv", A_matrices, b[error_indices])
            )

        print("Running conjugate gradient")
        solution_values, _unused_info = jax.scipy.sparse.linalg.cg(
            A=ATA_function, b=ATb, x0=initial_x
        )

        print("Done solving!")
        return solution_values

    @jax.partial(jax.jit, static_argnums=(0,))
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

        print("Getting variables")
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
            error_from_factor: Dict[LinearFactor, jnp.ndarray] = {}
            for group in self.factors.values():
                for factor in group:
                    error_from_factor[factor] = factor.compute_error_linear_component(
                        assignments=x
                    )

            # Ax => Apply Jacobian-transpose => A^TAx
            value_from_variable: Dict[RealVectorVariable, jnp.ndarray] = {}
            for variable in variables:
                value_from_variable[variable] = LinearFactorGraph.compute_error_dual(
                    variable, self.factors_from_variable[variable], error_from_factor
                )

            return value_from_variable

        print("Computing b")
        # Compute rhs (A.T @ b)
        b: Dict[RealVectorVariable, jnp.ndarray] = {
            variable: LinearFactorGraph.compute_error_dual(
                variable, self.factors_from_variable[variable]
            )
            for variable in variables
        }

        print("Running conjugate gradient")
        assignments_solution, _unused_info = jax.scipy.sparse.linalg.cg(
            A=A_function, b=b, x0=initial_assignments
        )

        print("Done solving!")
        return assignments_solution

    @classmethod
    def compute_error_dual(
        cls,
        variable: RealVectorVariable,
        factors: Set["LinearFactor"],
        error_from_factor: Optional[Dict["LinearFactor", jnp.ndarray]] = None,
    ):
        """Compute dual of error term; eg the terms of `A.T @ error` that correspond to
        this variable.

        Args:
            factors (Set["LinearFactor"]): Linearized factors that are attached to this variable.
            error_from_factor (Dict["LinearFactor", jnp.ndarray]): Mapping from factor to error term.
                Defaults to the `b` constant from each factor.
        """
        dual = jnp.zeros(variable.parameter_dim)
        if error_from_factor is None:
            for factor in factors:
                dual = dual + factor.A_from_variable[variable].T @ factor.b
        else:
            for factor in factors:
                dual = (
                    dual
                    + factor.A_from_variable[variable].T @ error_from_factor[factor]
                )
        return dual
