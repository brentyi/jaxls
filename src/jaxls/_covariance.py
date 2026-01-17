"""Covariance estimation for nonlinear least squares problems."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Callable, Literal

import jax
import jax_dataclasses as jdc
import scipy.sparse
from jax import numpy as jnp

from ._solvers import ConjugateGradientConfig
from ._variables import Var

if TYPE_CHECKING:
    pass


@jdc.pytree_dataclass
class LinearSolverCovarianceEstimatorConfig:
    """Configuration for covariance estimation using linear solves.

    This estimator computes covariance blocks by solving (J^T J) x = e_i
    for each tangent dimension. It is flexible and GPU-friendly (with CG),
    but requires linear solves for each covariance() call.
    """

    linear_solver: (
        Literal["conjugate_gradient", "dense_cholesky"] | ConjugateGradientConfig
    ) = "conjugate_gradient"
    """Linear solver for computing covariance columns.

    - "conjugate_gradient": Iterative solver, GPU-friendly, uses block-Jacobi
      preconditioner. Converges quickly when variables are weakly correlated.
    - "dense_cholesky": Direct solver, caches Cholesky factor for efficient
      repeated solves. Only suitable for small-medium problems.
    - ConjugateGradientConfig: Custom CG configuration. Note that Eisenstat-Walker
      tolerance parameters are ignored; only tolerance_min is used.
    """


class CovarianceEstimator(abc.ABC):
    """Abstract base class for covariance estimation.

    Covariance estimators compute blocks of the covariance matrix (J^T J)^{-1},
    representing uncertainty in the tangent space of estimated variables.

    See :meth:`AnalyzedLeastSquaresProblem.make_covariance_estimator` for
    constructing covariance estimators.
    """

    @abc.abstractmethod
    def covariance(
        self,
        var0: Var[Any],
        var1: Var[Any] | None = None,
    ) -> jax.Array:
        """Compute covariance block between two variables.

        Args:
            var0: First variable (determines rows of the covariance block).
            var1: Second variable (determines columns). If None, returns
                the marginal covariance Cov(var0, var0).

        Returns:
            Covariance block of shape (var0.tangent_dim, var1.tangent_dim).
        """
        ...


@jdc.pytree_dataclass
class SpinvCovarianceEstimator(CovarianceEstimator):
    """Covariance estimator using CHOLMOD sparse Cholesky factorization.

    Precomputes the full covariance matrix (J^T J)^{-1} using CHOLMOD's
    sparse Cholesky factorization, allowing O(1) extraction of covariance
    blocks.

    This is efficient for extracting many covariance blocks, but requires
    sksparse and only runs on CPU. For large problems, consider using
    LinearSolveCovarianceEstimator instead.
    """

    _sparse_cov: jdc.Static[scipy.sparse.spmatrix]
    """Sparse covariance matrix (J^T J)^{-1} from spinv."""
    _residual_variance: jax.Array
    """Estimated residual variance for scaling."""
    _tangent_start_from_var_type: jdc.Static[dict[type[Var[Any]], int]]
    _sorted_ids_from_var_type: jdc.Static[dict[type[Var[Any]], jax.Array]]

    def covariance(
        self,
        var0: Var[Any],
        var1: Var[Any] | None = None,
    ) -> jax.Array:
        """Extract covariance block between two variables.

        Note: entries outside the sparsity pattern will be zero.
        """
        if var1 is None:
            var1 = var0

        # Compute tangent indices.
        def get_tangent_slice(var: Var[Any]) -> tuple[int, int]:
            var_type = type(var)
            tangent_start = self._tangent_start_from_var_type[var_type]
            sorted_ids = self._sorted_ids_from_var_type[var_type]
            position = int(jnp.searchsorted(sorted_ids, var.id))
            start = tangent_start + position * var_type.tangent_dim
            end = start + var_type.tangent_dim
            return start, end

        start0, end0 = get_tangent_slice(var0)
        start1, end1 = get_tangent_slice(var1)

        # Extract block from sparse matrix.
        cov_block = self._sparse_cov[start0:end0, start1:end1].toarray()  # type: ignore
        return jnp.asarray(cov_block) * self._residual_variance


@jdc.pytree_dataclass
class LinearSolveCovarianceEstimator(CovarianceEstimator):
    """Covariance estimator using linear solves.

    Computes covariance blocks by solving (J^T J) x = e_i for each
    tangent dimension of var0. Supports conjugate gradient (GPU-friendly)
    or dense Cholesky solvers.

    With CG and block-Jacobi preconditioning, convergence is fast when
    variables are weakly correlated (the common case in sparse problems).
    """

    _solve_fn: jdc.Static[Callable[[jax.Array], jax.Array]]
    """Function that solves (J^T J) x = b for a given b."""
    _tangent_dim: jdc.Static[int]
    """Total tangent dimension."""
    _residual_variance: jax.Array
    """Estimated residual variance for scaling."""
    _tangent_start_from_var_type: jdc.Static[dict[type[Var[Any]], int]]
    _sorted_ids_from_var_type: jdc.Static[dict[type[Var[Any]], jax.Array]]

    def covariance(
        self,
        var0: Var[Any],
        var1: Var[Any] | None = None,
    ) -> jax.Array:
        """Compute covariance block between two variables."""
        if var1 is None:
            var1 = var0

        var0_type = type(var0)
        var1_type = type(var1)
        d0 = var0_type.tangent_dim
        d1 = var1_type.tangent_dim

        # Compute tangent indices.
        def get_tangent_start(var: Var[Any]) -> jax.Array:
            var_type = type(var)
            tangent_start = self._tangent_start_from_var_type[var_type]
            sorted_ids = self._sorted_ids_from_var_type[var_type]
            position = jnp.searchsorted(sorted_ids, var.id)
            return tangent_start + position * var_type.tangent_dim

        var0_start = get_tangent_start(var0)
        var1_start = get_tangent_start(var1)
        var0_indices = var0_start + jnp.arange(d0)
        var1_indices = var1_start + jnp.arange(d1)

        # Solve (J^T J) x = e_i for each tangent dimension of var0.
        def solve_for_column(idx: jax.Array) -> jax.Array:
            e_i = jnp.zeros(self._tangent_dim).at[idx].set(1.0)
            return self._solve_fn(e_i)

        solutions = jax.vmap(solve_for_column)(var0_indices)

        # Extract var1 entries and scale by residual variance.
        cov_block = solutions[:, var1_indices] * self._residual_variance
        return cov_block
