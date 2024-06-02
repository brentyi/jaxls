from collections import defaultdict
from typing import Collection, DefaultDict, Dict, Hashable, Iterable, List, Tuple, cast

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp

from .. import hints, noises, sparse
from ..solvers import GaussNewtonSolver, NonlinearSolverBase
from ._factor_base import FactorBase
from ._factor_stack import FactorStack
from ._variable_assignments import StorageLayout, VariableAssignments
from ._variables import VariableBase

# Key for determining which factors are grouped for stacking
GroupKey = Hashable


@jdc.pytree_dataclass
class StackedFactorGraph:
    """Dataclass for vectorized factor graph computations.

    Improves runtime by stacking factors based on their group key.
    """

    factor_stacks: List[FactorStack]
    jacobian_coords: sparse.SparseCooCoordinates
    storage_layout: jdc.Static[StorageLayout]
    local_storage_layout: jdc.Static[StorageLayout]
    residual_dim: jdc.Static[int]

    # Shape checks break under vmap
    # def __post_init__(self):
    #     """Check that inputs make sense!"""
    #     for stacked_factor in self.factor_stacks:
    #         N = stacked_factor.num_factors
    #         for value_indices, variable in zip(
    #             stacked_factor.value_indices,
    #             stacked_factor.factor.variables,
    #         ):
    #             assert value_indices.shape == (N, variable.get_parameter_dim())

    def get_variables(self) -> Collection[VariableBase]:
        return self.local_storage_layout.get_variables()

    @staticmethod
    def make(
        factors: Iterable[FactorBase],
        use_onp: bool = True,
    ) -> "StackedFactorGraph":
        """Create a factor graph from a set of factors."""

        # Start by grouping our factors and grabbing a list of (ordered!) variables
        factors_from_group: DefaultDict[GroupKey, List[FactorBase]] = defaultdict(list)
        variables_ordered_set: Dict[VariableBase, None] = {}
        for factor in factors:
            # Each factor is ultimately just a pytree node; in order for a set of
            # factors to be batchable, they must share the same:
            group_key: GroupKey = (
                # (1) Treedef. Note that variables can be different as long as their
                # types are the same.
                jax.tree_structure(factor.anonymize_variables()),
                # (2) Leaf shapes: contained array shapes must match
                tuple(
                    leaf.shape if hasattr(leaf, "shape") else ()
                    for leaf in jax.tree_leaves(factor)
                ),
            )

            # Record factor and variables
            factors_from_group[group_key].append(factor)
            for v in factor.variables:
                variables_ordered_set[v] = None
        variables = list(variables_ordered_set.keys())

        # Fields we want to populate
        stacked_factors: List[FactorStack] = []
        jacobian_coords: List[sparse.SparseCooCoordinates] = []

        # Create storage layout: this describes which parts of our storage object is
        # allocated to each variable
        storage_layout = StorageLayout.make(variables, local=False)
        local_storage_layout = StorageLayout.make(variables, local=True)

        # Prepare each factor group
        residual_offset = 0
        for group_key, group in factors_from_group.items():
            # Make factor stack
            stacked_factors.append(
                FactorStack.make(
                    group,
                    storage_layout,
                    use_onp=use_onp,
                )
            )

            # Compute Jacobian coordinates
            #
            # These should be N pairs of (row, col) indices, where rows correspond to
            # residual indices and columns correspond to local parameter indices
            jacobian_coords.extend(
                FactorStack.compute_jacobian_coords(
                    factors=group,
                    local_storage_layout=local_storage_layout,
                    row_offset=residual_offset,
                )
            )
            residual_offset += stacked_factors[-1].get_residual_dim()

        jacobian_coords_concat: sparse.SparseCooCoordinates = jax.tree_map(
            lambda *arrays: onp.concatenate(arrays, axis=0), *jacobian_coords
        )

        return StackedFactorGraph(
            factor_stacks=stacked_factors,
            jacobian_coords=jacobian_coords_concat,
            storage_layout=storage_layout,
            local_storage_layout=local_storage_layout,
            residual_dim=residual_offset,
        )

    @jdc.jit
    def compute_whitened_residual_vector(
        self, assignments: VariableAssignments
    ) -> jnp.ndarray:
        """Computes flattened+whitened residual vector associated with our factor graph.

        Args:
            assignments (VariableAssignments): Variable assignments.

        Returns:
            jnp.ndarray: Residual vector.
        """

        # Resolve storage layout mismatches. Factor stack computations will raise an
        # assertion error if the storage layout is incorrect.
        assignments = assignments.update_storage_layout(self.storage_layout)

        # Flatten and concatenate residuals from all groups.
        stacked_factor: FactorStack
        residual_vector = jnp.concatenate(
            [
                jax.vmap(
                    type(stacked_factor.factor.noise_model).whiten_residual_vector
                )(
                    stacked_factor.factor.noise_model,
                    stacked_factor.compute_residual_vector(assignments),
                ).flatten()
                for stacked_factor in self.factor_stacks
            ],
            axis=0,
        )
        assert residual_vector.shape == (self.residual_dim,)
        return residual_vector

    @jdc.jit
    def compute_cost(
        self, assignments: VariableAssignments
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the sum of squared residuals associated with a factor graph. Also
        returns intermediate (whitened) residual vector.

        Args:
            assignments (VariableAssignments): Variable assignments.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Scalar cost, residual vector.
        """
        residual_vector = self.compute_whitened_residual_vector(assignments)
        cost = jnp.sum(residual_vector**2)
        return cost, residual_vector

    @jdc.jit
    def compute_joint_nll(
        self,
        assignments: VariableAssignments,
        include_constants: bool = False,
    ) -> jnp.ndarray:
        """Compute the joint negative log-likelihood density associated with a set of
        variable assignments. Assumes Gaussian noise models.

        Args:
            assignments (VariableAssignments): Variable assignments.
            include_constants (bool): Whether or not to include constant terms.

        Returns:
            jnp.ndarray: Scalar cost negative log-likelihood.
        """

        if include_constants:
            raise NotImplementedError()

        # Add Mahalanobis distance terms to NLL
        joint_nll = jnp.sum(self.compute_whitened_residual_vector(assignments) ** 2)

        # Add log-determinant terms
        for stacked_factor in self.factor_stacks:
            noise_model = stacked_factor.factor.noise_model
            if isinstance(noise_model, noises.Gaussian):
                cov_determinants = -2.0 * jnp.log(
                    jnp.abs(
                        jnp.linalg.det(
                            cast(noises.Gaussian, noise_model).sqrt_precision_matrix
                        )
                    )
                )
            elif isinstance(noise_model, noises.DiagonalGaussian):
                cov_determinants = -2.0 * jnp.log(
                    jnp.abs(
                        jnp.prod(
                            cast(
                                noises.DiagonalGaussian, noise_model
                            ).sqrt_precision_diagonal,
                            axis=-1,
                        )
                    )
                )
            else:
                assert False, f"Joint NLL not supported  for {type(noise_model)}"
            assert cov_determinants.shape == (stacked_factor.num_factors,)

            joint_nll = joint_nll + jnp.sum(cov_determinants)

        return joint_nll

    @jdc.jit
    def compute_whitened_residual_jacobian(
        self,
        assignments: VariableAssignments,
        residual_vector: hints.Array,
    ) -> sparse.SparseCooMatrix:
        """Compute the Jacobian of a graph's residual vector with respect to the stacked
        local delta vectors. Shape should be `(residual_dim, local_delta_storage_dim)`.
        """

        # Resolve storage layout mismatches. Factor stack computations will raise an
        # assertion error if the storage layout is incorrect.
        assignments = assignments.update_storage_layout(self.storage_layout)

        # Linearize factors by group.
        A_values_list: List[jnp.ndarray] = []
        residual_start = 0
        residual_end = 0
        for stacked_factor in self.factor_stacks:
            residual_end = residual_start + stacked_factor.get_residual_dim()
            stacked_residual_vector = residual_vector[
                residual_start:residual_end
            ].reshape(
                (
                    stacked_factor.num_factors,
                    stacked_factor.factor.get_residual_dim(),
                )
            )

            # Compute all Jacobians and whiten.
            for jacobian in stacked_factor.compute_residual_jacobian(assignments):
                A_values_list.append(
                    jax.vmap(type(stacked_factor.factor.noise_model).whiten_jacobian)(
                        stacked_factor.factor.noise_model,
                        jacobian,
                        residual_vector=stacked_residual_vector,
                    )
                )
            residual_start = residual_end
        assert residual_end != 0
        assert residual_end == self.residual_dim

        # Build Jacobian.
        A = sparse.SparseCooMatrix(
            values=jnp.concatenate([A.flatten() for A in A_values_list]),
            coords=self.jacobian_coords,
            shape=(self.residual_dim, self.local_storage_layout.dim),
        )
        return A

    def solve(
        self,
        initial_assignments: VariableAssignments,
        solver: NonlinearSolverBase = GaussNewtonSolver(),
    ) -> VariableAssignments:
        """Solve MAP inference problem."""
        # Note that the solver will handle storage layout mismatches.
        return solver.solve(graph=self, initial_assignments=initial_assignments)
