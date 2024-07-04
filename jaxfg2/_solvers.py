from __future__ import annotations

import abc
import functools
from typing import Hashable, override

import jax
import jax.flatten_util
import jax_dataclasses as jdc
import sksparse.cholmod
from jax import numpy as jnp

from jaxfg2.utils import jax_log

from ._factor_graph import StackedFactorGraph, TangentStorageLayout
from ._sparse_matrices import SparseCooMatrix
from ._variables import VarValues

# Linear solvers.

_cholmod_analyze_cache: dict[Hashable, sksparse.cholmod.Factor] = {}


@jdc.pytree_dataclass
class CholmodSolver:
    def solve(
        self,
        A: SparseCooMatrix,
        ATb: jax.Array,
        lambd: float | jax.Array,
        iterations: int | jax.Array,
    ) -> jax.Array:
        del iterations
        return jax.pure_callback(
            self._solve_on_host,
            ATb,  # Result shape/dtype.
            A,
            ATb,
            lambd,
            vectorized=False,
        )

    def _solve_on_host(
        self,
        A: SparseCooMatrix,
        ATb: jax.Array,
        lambd: float | jax.Array,
    ) -> jax.Array:
        A_T = A.T
        A_T_scipy = A_T.as_scipy_coo_matrix().tocsc(copy=False)

        # Cache sparsity pattern analysis
        self_hash = object.__hash__(self)
        if self_hash not in _cholmod_analyze_cache:
            _cholmod_analyze_cache[self_hash] = sksparse.cholmod.analyze_AAt(A_T_scipy)

        # Factorize and solve
        _cholmod_analyze_cache[self_hash].cholesky_AAt_inplace(
            A_T_scipy,
            beta=lambd
            + 1e-5,  # Some simple linear problems blow up without this 1e-5 term
        )
        return _cholmod_analyze_cache[self_hash].solve_A(ATb)


# Nonlinear solve utils.


@jdc.pytree_dataclass
class _TerminationCriteriaMixin:
    """Mixin for Ceres-style termination criteria."""

    max_iterations: int = 100
    """Maximum number of iterations."""

    cost_tolerance: float = 1e-5
    """We terminate if `|cost change| / cost < cost_tolerance`."""

    gradient_tolerance: float = 1e-7
    """We terminate if `norm_inf(x - rplus(x, linear delta)) < gradient_tolerance`."""

    gradient_tolerance_start_step: int = 10
    """When to start checking the gradient tolerance condition. Helps solve precision
    issues caused by inexact Newton steps."""

    parameter_tolerance: float = 1e-6
    """We terminate if `norm_2(linear delta) < (norm2(x) + parameter_tolerance) * parameter_tolerance`."""

    def check_convergence(
        self,
        state_prev: NonlinearSolverState,
        cost_updated: jax.Array,
        tangent: jax.Array,
        tangent_layout: TangentStorageLayout,
        ATb: jax.Array,
    ) -> jax.Array:
        """Check for convergence!"""

        # Cost tolerance
        converged_cost = (
            jnp.abs(cost_updated - state_prev.cost) / state_prev.cost
            < self.cost_tolerance
        )

        # Gradient tolerance
        flat_vals = jax.flatten_util.ravel_pytree(state_prev.vals)[0]
        converged_gradient = jnp.where(
            state_prev.iterations >= self.gradient_tolerance_start_step,
            jnp.max(
                flat_vals
                - jax.flatten_util.ravel_pytree(
                    state_prev.vals._retract(ATb, tangent_layout.ordering)
                )[0]
            )
            < self.gradient_tolerance,
            False,
        )

        # Parameter tolerance
        converged_parameters = (
            jnp.linalg.norm(jnp.abs(tangent))
            < (jnp.linalg.norm(flat_vals) + self.parameter_tolerance)
            * self.parameter_tolerance
        )

        return jnp.any(
            jnp.array(
                [
                    converged_cost,
                    converged_gradient,
                    converged_parameters,
                    state_prev.iterations >= (self.max_iterations - 1),
                ]
            ),
            axis=0,
        )


# Nonlinear solvers.


@jdc.pytree_dataclass
class NonlinearSolverState:
    iterations: int | jax.Array
    vals: VarValues
    cost: float | jax.Array
    residual_vector: jax.Array
    done: bool | jax.Array


@jdc.pytree_dataclass
class NonlinearSolver[TState: NonlinearSolverState]:
    linear_solver: CholmodSolver = CholmodSolver()
    verbose: jdc.Static[bool] = True
    """Set to `True` to enable printing."""

    @abc.abstractmethod
    def _initialize_state(self, graph: StackedFactorGraph, vals: VarValues) -> TState:
        ...

    @abc.abstractmethod
    def _step(self, graph: StackedFactorGraph, state: TState) -> TState:
        ...

    @jdc.jit
    def solve(self, graph: StackedFactorGraph, initial_vals: VarValues) -> VarValues:
        vals = initial_vals
        state = self._initialize_state(graph, vals)

        # Optimization.
        state = jax.lax.while_loop(
            cond_fun=lambda state: jnp.logical_not(state.done),
            body_fun=functools.partial(self._step, graph),
            init_val=state,
        )
        if self.verbose:
            jax_log(
                "Terminated @ iteration #{i}: cost={cost:.4f}",
                i=state.iterations,
                cost=state.cost,
            )
        return state.vals


@jdc.pytree_dataclass
class GaussNewtonSolver(
    NonlinearSolver[NonlinearSolverState], _TerminationCriteriaMixin
):
    @override
    def _initialize_state(
        self, graph: StackedFactorGraph, vals: VarValues
    ) -> NonlinearSolverState:
        residual_vector = graph.compute_residual_vector(vals)
        return NonlinearSolverState(
            iterations=0,
            vals=vals,
            cost=jnp.sum(residual_vector**2),
            residual_vector=residual_vector,
            done=False,
        )

    @override
    def _step(
        self, graph: StackedFactorGraph, state: NonlinearSolverState
    ) -> NonlinearSolverState:
        A = graph._compute_jacobian_wrt_tangent(state.vals)
        ATb = -(A.T @ state.residual_vector)

        tangent = self.linear_solver.solve(
            A, ATb, lambd=0.0, iterations=state.iterations
        )
        vals = state.vals._retract(tangent, graph.tangent_layout.ordering)

        if self.verbose:
            jax_log(
                "Gauss-Newton step #{i}: cost={cost:.4f}",
                i=state.iterations,
                cost=state.cost,
            )
        with jdc.copy_and_mutate(state) as state_next:
            state_next.vals = vals
            state_next.residual_vector = graph.compute_residual_vector(vals)
            state_next.cost = jnp.sum(state_next.residual_vector**2)
            state_next.iterations += 1
            state_next.done = self.check_convergence(
                state,
                cost_updated=state_next.cost,
                tangent=tangent,
                tangent_layout=graph.tangent_layout,
                ATb=ATb,
            )
        return state_next
