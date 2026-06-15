from __future__ import annotations

import abc
from typing import Any

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp


@jdc.pytree_dataclass
class LinearSolverCovarianceEstimatorConfig:
    linear_solver: Any = "conjugate_gradient"


class CovarianceEstimator(abc.ABC):
    @abc.abstractmethod
    def covariance(
        self,
        var0: Any,
        var1: Any = None,
    ) -> Any: ...


@jdc.pytree_dataclass
class SpinvCovarianceEstimator(CovarianceEstimator):
    _sparse_cov: jdc.Static[Any]
    _residual_variance: Any
    _tangent_start_from_var_type: jdc.Static[Any]
    _sorted_ids_from_var_type: jdc.Static[Any]

    def covariance(
        self,
        var0: Any,
        var1: Any = None,
    ) -> Any:
        if var1 is None:
            var1 = var0

        def get_tangent_slice(var: Any) -> Any:
            var_type = type(var)
            tangent_start = self._tangent_start_from_var_type[var_type]
            sorted_ids = self._sorted_ids_from_var_type[var_type]
            position = int(jnp.searchsorted(sorted_ids, var.id))
            start = tangent_start + position * var_type.tangent_dim
            end = start + var_type.tangent_dim
            return start, end

        start0, end0 = get_tangent_slice(var0)
        start1, end1 = get_tangent_slice(var1)

        cov_block = self._sparse_cov[start0:end0, start1:end1].toarray()
        return jnp.asarray(cov_block) * self._residual_variance


@jdc.pytree_dataclass
class LinearSolveCovarianceEstimator(CovarianceEstimator):
    _solve_fn: jdc.Static[Any]
    _tangent_dim: jdc.Static[Any]
    _residual_variance: Any
    _tangent_start_from_var_type: jdc.Static[Any]
    _sorted_ids_from_var_type: jdc.Static[Any]

    def covariance(
        self,
        var0: Any,
        var1: Any = None,
    ) -> Any:
        if var1 is None:
            var1 = var0

        var0_type = type(var0)
        var1_type = type(var1)
        d0 = var0_type.tangent_dim
        d1 = var1_type.tangent_dim

        def get_tangent_start(var: Any) -> Any:
            var_type = type(var)
            tangent_start = self._tangent_start_from_var_type[var_type]
            sorted_ids = self._sorted_ids_from_var_type[var_type]
            position = jnp.searchsorted(sorted_ids, var.id)
            return tangent_start + position * var_type.tangent_dim

        var0_start = get_tangent_start(var0)
        var1_start = get_tangent_start(var1)
        var0_indices = var0_start + jnp.arange(d0)
        var1_indices = var1_start + jnp.arange(d1)

        def solve_for_column(idx: Any) -> Any:
            e_i = jnp.zeros(self._tangent_dim).at[idx].set(1.0)
            return self._solve_fn(e_i)

        solutions = jax.vmap(solve_for_column)(var0_indices)

        cov_block = solutions[:, var1_indices] * self._residual_variance
        return cov_block
