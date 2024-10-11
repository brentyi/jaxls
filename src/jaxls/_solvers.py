from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, Hashable, cast

import jax
import jax.experimental.sparse
import jax.flatten_util
import jax_dataclasses as jdc
import scipy
import scipy.sparse
import sksparse.cholmod
from jax import numpy as jnp

from ._sparse_matrices import SparseCooMatrix, SparseCsrMatrix
from ._variables import VarTypeOrdering, VarValues
from .utils import jax_log

if TYPE_CHECKING:
    from ._factor_graph import FactorGraph


_cholmod_analyze_cache: dict[Hashable, sksparse.cholmod.Factor] = {}


@jdc.pytree_dataclass
class CholmodLinearSolver:
    """Direct solver for sparse linear systems. Runs on CPU."""

    def _solve(
        self, A: SparseCsrMatrix, ATb: jax.Array, lambd: float | jax.Array
    ) -> jax.Array:
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
        A: SparseCsrMatrix,
        ATb: jax.Array,
        lambd: float | jax.Array,
    ) -> jax.Array:
        # Matrix is transposed when we convert CSR to CSC.
        A_T_scipy = scipy.sparse.csc_matrix(
            (A.values, A.coords.indices, A.coords.indptr), shape=A.coords.shape[::-1]
        )

        # Cache sparsity pattern analysis.
        cache_key = (
            A.coords.indices.tobytes(),
            A.coords.indptr.tobytes(),
            A.coords.shape,
        )
        factor = _cholmod_analyze_cache.get(cache_key, None)
        if factor is None:
            factor = sksparse.cholmod.analyze_AAt(A_T_scipy)
            _cholmod_analyze_cache[cache_key] = factor

            max_cache_size = 512
            if len(_cholmod_analyze_cache) > max_cache_size:
                _cholmod_analyze_cache.pop(next(iter(_cholmod_analyze_cache)))

        # Factorize and solve
        factor = factor.cholesky_AAt(
            A_T_scipy,
            # Some simple linear problems blow up without this 1e-5 term.
            beta=lambd + 1e-5,
        )
        return factor.solve_A(ATb)


@jdc.pytree_dataclass
class ConjugateGradientLinearSolver:
    """Iterative solver for sparse linear systems. Can run on CPU or GPU."""

    tolerance: float = 1e-5
    inexact_step_eta: float | None = None
    """Forcing sequence parameter for inexact Newton steps. CG tolerance is set to
    `eta / iteration #`.

    For reference, see AN INEXACT LEVENBERG-MARQUARDT METHOD FOR LARGE SPARSE NONLINEAR
    LEAST SQUARES, Wright & Holt 1983."""

    def _solve(
        self,
        A_coo: jax.experimental.sparse.BCOO,
        ATA_multiply: Callable[[jax.Array], jax.Array],
        ATb: jax.Array,
        iterations: int | jax.Array,
    ) -> jax.Array:
        assert len(ATb.shape) == 1, "ATb should be 1D!"

        # Get diagonals of ATA for preconditioning.
        ATA_diagonals = jnp.zeros_like(ATb).at[A_coo.indices[:, 1]].add(A_coo.data**2)

        # Solve with conjugate gradient.
        initial_x = jnp.zeros(ATb.shape)
        solution_values, _ = jax.scipy.sparse.linalg.cg(
            A=ATA_multiply,
            b=ATb,
            x0=initial_x,
            # https://en.wikipedia.org/wiki/Conjugate_gradient_method#Convergence_properties
            maxiter=len(initial_x),
            tol=cast(
                float,
                jnp.maximum(self.tolerance, self.inexact_step_eta / (iterations + 1)),
            )
            if self.inexact_step_eta is not None
            else self.tolerance,
            M=lambda x: x / ATA_diagonals,  # Jacobi preconditioner.
        )
        return solution_values


# Nonlinear solvers.


@jdc.pytree_dataclass
class NonlinearSolverState:
    iterations: int | jax.Array
    vals: VarValues
    cost: float | jax.Array
    residual_vector: jax.Array
    done: bool | jax.Array
    lambd: float | jax.Array


@jdc.pytree_dataclass
class NonlinearSolver:
    """Helper class for solving using Gauss-Newton or Levenberg-Marquardt."""

    linear_solver: CholmodLinearSolver | ConjugateGradientLinearSolver
    trust_region: TrustRegionConfig | None
    termination: TerminationConfig
    verbose: jdc.Static[bool]

    @jdc.jit
    def solve(self, graph: FactorGraph, initial_vals: VarValues) -> VarValues:
        vals = initial_vals
        residual_vector = graph.compute_residual_vector(vals)
        state = NonlinearSolverState(
            iterations=0,
            vals=vals,
            cost=jnp.sum(residual_vector**2),
            residual_vector=residual_vector,
            done=False,
            lambd=self.trust_region.lambda_initial
            if self.trust_region is not None
            else 0.0,
        )

        # Optimization.
        state = jax.lax.while_loop(
            cond_fun=lambda state: jnp.logical_not(state.done),
            body_fun=functools.partial(self.step, graph),
            init_val=state,
        )
        if self.verbose:
            jax_log(
                "Terminated @ iteration #{i}: cost={cost:.4f}",
                i=state.iterations,
                cost=state.cost,
            )
        return state.vals

    def step(
        self, graph: FactorGraph, state: NonlinearSolverState
    ) -> NonlinearSolverState:
        jac_values, A_blocksparse = graph._compute_jac_values(state.vals)
        A_coo = SparseCooMatrix(jac_values, graph.jac_coords_coo).as_jax_bcoo()
        A_multiply = A_blocksparse.multiply
        AT_multiply = A_blocksparse.transpose().multiply

        # Equivalently:
        #     AT_multiply = lambda vec: jax.linear_transpose(
        #         A_blocksparse.multiply, jnp.zeros((A_blocksparse.shape[1],))
        #     )(vec)[0]

        ATb = -AT_multiply(state.residual_vector)

        if isinstance(self.linear_solver, ConjugateGradientLinearSolver):
            tangent = self.linear_solver._solve(
                A_coo,
                # We could also use (lambd * ATA_diagonals * vec) for
                # scale-invariant damping. But this is hard to match with CHOLMOD.
                lambda vec: AT_multiply(A_multiply(vec)) + state.lambd * vec,
                ATb,
                iterations=state.iterations,
            )
        elif isinstance(self.linear_solver, CholmodLinearSolver):
            A_csr = SparseCsrMatrix(jac_values, graph.jac_coords_csr)
            tangent = self.linear_solver._solve(A_csr, ATb, lambd=state.lambd)
        else:
            assert False

        vals = state.vals._retract(tangent, graph.tangent_ordering)
        if self.verbose:
            jax_log(
                " step #{i}: cost={cost:.4f} lambd={lambd:.4f}",
                i=state.iterations,
                cost=state.cost,
                lambd=state.lambd,
                ordered=True,
            )
            residual_index = 0
            for f, count in zip(graph.stacked_factors, graph.factor_counts):
                stacked_dim = count * f.residual_dim
                partial_cost = jnp.sum(
                    state.residual_vector[residual_index : residual_index + stacked_dim]
                    ** 2
                )
                residual_index += stacked_dim
                jax_log(
                    "     - "
                    + f"{f.compute_residual.__name__}({count}):".ljust(15)
                    + " {:.5f} (avg {:.5f})",
                    partial_cost,
                    partial_cost / stacked_dim,
                    ordered=True,
                )

        with jdc.copy_and_mutate(state) as state_next:
            proposed_residual_vector = graph.compute_residual_vector(vals)
            proposed_cost = jnp.sum(proposed_residual_vector**2)

            # Always accept Gauss-Newton steps.
            if self.trust_region is None:
                state_next.vals = vals
                state_next.residual_vector = proposed_residual_vector
                state_next.cost = proposed_cost

            # For Levenberg-Marquardt, we need to evaluate the step quality.
            else:
                step_quality = (proposed_cost - state.cost) / (
                    jnp.sum((A_coo @ tangent + state.residual_vector) ** 2) - state.cost
                )
                accept_flag = step_quality >= self.trust_region.step_quality_min

                state_next.vals = jax.tree_map(
                    lambda proposed, current: jnp.where(accept_flag, proposed, current),
                    vals,
                    state.vals,
                )
                state_next.residual_vector = jnp.where(
                    accept_flag, proposed_residual_vector, state.residual_vector
                )
                state_next.cost = jnp.where(accept_flag, proposed_cost, state.cost)
                state_next.lambd = jnp.where(
                    accept_flag,
                    # If accept, decrease damping: note that we *don't* enforce any bounds here
                    state.lambd / self.trust_region.lambda_factor,
                    # If reject: increase lambda and enforce bounds
                    jnp.maximum(
                        self.trust_region.lambda_min,
                        jnp.minimum(
                            state.lambd * self.trust_region.lambda_factor,
                            self.trust_region.lambda_max,
                        ),
                    ),
                )

            state_next.iterations += 1
            state_next.done = self.termination._check_convergence(
                state,
                cost_updated=state_next.cost,
                tangent=tangent,
                tangent_ordering=graph.tangent_ordering,
                ATb=ATb,
            )
        return state_next


@jdc.pytree_dataclass
class TrustRegionConfig:
    # Levenberg-Marquardt parameters.
    lambda_initial: float = 5e-4
    """Initial damping factor. Only used for Levenberg-Marquardt."""
    lambda_factor: float = 2.0
    """Factor to increase or decrease damping. Only used for Levenberg-Marquardt."""
    lambda_min: float = 1e-5
    """Minimum damping factor. Only used for Levenberg-Marquardt."""
    lambda_max: float = 1e10
    """Maximum damping factor. Only used for Levenberg-Marquardt."""
    step_quality_min: float = 1e-3
    """Minimum step quality for Levenberg-Marquardt. Only used for Levenberg-Marquardt."""


@jdc.pytree_dataclass
class TerminationConfig:
    # Termination criteria.
    max_iterations: int = 100
    cost_tolerance: float = 1e-6
    """We terminate if `|cost change| / cost < cost_tolerance`."""
    gradient_tolerance: float = 1e-8
    """We terminate if `norm_inf(x - rplus(x, linear delta)) < gradient_tolerance`."""
    gradient_tolerance_start_step: int = 10
    """When to start checking the gradient tolerance condition. Helps solve precision
    issues caused by inexact Newton steps."""
    parameter_tolerance: float = 1e-7
    """We terminate if `norm_2(linear delta) < (norm2(x) + parameter_tolerance) * parameter_tolerance`."""

    def _check_convergence(
        self,
        state_prev: NonlinearSolverState,
        cost_updated: jax.Array,
        tangent: jax.Array,
        tangent_ordering: VarTypeOrdering,
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
                    state_prev.vals._retract(ATb, tangent_ordering)
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
