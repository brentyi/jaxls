from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, Hashable, Literal, assert_never, cast

import jax
import jax.flatten_util
import jax_dataclasses as jdc
import scipy
import scipy.sparse
from jax import numpy as jnp

from jaxls._preconditioning import (
    make_block_jacobi_precoditioner,
    make_point_jacobi_precoditioner,
)

from ._sparse_matrices import BlockRowSparseMatrix, SparseCooMatrix, SparseCsrMatrix
from ._variables import VarTypeOrdering, VarValues
from .utils import jax_log

if TYPE_CHECKING:
    import sksparse.cholmod

    from ._core import AnalyzedLeastSquaresProblem


_cholmod_analyze_cache: dict[Hashable, sksparse.cholmod.Factor] = {}


def _cholmod_solve(
    A: SparseCsrMatrix, ATb: jax.Array, lambd: float | jax.Array
) -> jax.Array:
    """JIT-friendly linear solve using CHOLMOD."""
    return jax.pure_callback(
        _cholmod_solve_on_host,
        ATb,  # Result shape/dtype.
        A,
        ATb,
        lambd,
        vectorized=False,
    )


def _cholmod_solve_on_host(
    A: SparseCsrMatrix,
    ATb: jax.Array,
    lambd: float | jax.Array,
) -> jax.Array:
    """Solve a linear system using CHOLMOD. Should be called on the host."""
    import sksparse.cholmod

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
    cost = _cholmod_analyze_cache.get(cache_key, None)
    if cost is None:
        cost = sksparse.cholmod.analyze_AAt(A_T_scipy)
        _cholmod_analyze_cache[cache_key] = cost

        max_cache_size = 512
        if len(_cholmod_analyze_cache) > max_cache_size:
            _cholmod_analyze_cache.pop(next(iter(_cholmod_analyze_cache)))

    # Factorize and solve
    cost = cost.cholesky_AAt(
        A_T_scipy,
        # Some simple linear problems blow up without this 1e-5 term.
        beta=lambd + 1e-5,
    )
    return cost.solve_A(ATb)


@jdc.pytree_dataclass
class _ConjugateGradientState:
    """State used for Eisenstat-Walker criterion in ConjugateGradientLinearSolver."""

    ATb_norm_prev: float | jax.Array
    """Previous norm of ATb."""
    eta: float | jax.Array
    """Current tolerance."""


@jdc.pytree_dataclass
class ConjugateGradientConfig:
    """Iterative solver for sparse linear systems. Can run on CPU or GPU.

    For inexact steps, we use the Eisenstat-Walker criterion. For reference,
    see "Choosing the Forcing Terms in an Inexact Newton Method", Eisenstat &
    Walker, 1996."
    """

    tolerance_min: float = 1e-7
    tolerance_max: float = 1e-2

    eisenstat_walker_gamma: float = 0.9
    """Eisenstat-Walker criterion gamma term. Controls how quickly the tolerance
    decreases. Typical values range from 0.5 to 0.9. Higher values lead to more
    aggressive tolerance reduction."""
    eisenstat_walker_alpha: float = 2.0
    """ Eisenstat-Walker criterion alpha term. Determines rate at which the
    tolerance changes based on residual reduction. Typical values are 1.5 or
    2.0. Higher values make the tolerance more sensitive to residual changes."""

    preconditioner: jdc.Static[Literal["block_jacobi", "point_jacobi"] | None] = (
        "block_jacobi"
    )
    """Preconditioner to use for linear solves."""

    def _solve(
        self,
        problem: AnalyzedLeastSquaresProblem,
        A_blocksparse: BlockRowSparseMatrix,
        ATA_multiply: Callable[[jax.Array], jax.Array],
        ATb: jax.Array,
        prev_linear_state: _ConjugateGradientState,
    ) -> tuple[jax.Array, _ConjugateGradientState]:
        assert len(ATb.shape) == 1, "ATb should be 1D!"

        # Preconditioning setup.
        if self.preconditioner == "block_jacobi":
            preconditioner = make_block_jacobi_precoditioner(problem, A_blocksparse)
        elif self.preconditioner == "point_jacobi":
            preconditioner = make_point_jacobi_precoditioner(A_blocksparse)
        elif self.preconditioner is None:
            preconditioner = lambda x: x
        else:
            assert_never(self.preconditioner)

        # Calculate tolerance using Eisenstat-Walker criterion.
        ATb_norm = jnp.linalg.norm(ATb)
        current_eta = jnp.minimum(
            self.eisenstat_walker_gamma
            * (ATb_norm / (prev_linear_state.ATb_norm_prev + 1e-7))
            ** self.eisenstat_walker_alpha,
            self.tolerance_max,
        )
        current_eta = jnp.maximum(
            self.tolerance_min, jnp.minimum(current_eta, prev_linear_state.eta)
        )

        # Solve with conjugate gradient.
        initial_x = jnp.zeros(ATb.shape)
        solution_values, _ = jax.scipy.sparse.linalg.cg(
            A=ATA_multiply,
            b=ATb,
            x0=initial_x,
            # https://en.wikipedia.org/wiki/Conjugate_gradient_method#Convergence_properties
            maxiter=len(initial_x),
            tol=cast(float, current_eta),
            M=preconditioner,
        )
        return solution_values, _ConjugateGradientState(
            ATb_norm_prev=ATb_norm, eta=current_eta
        )


# Nonlinear solvers.


@jdc.pytree_dataclass
class NonlinearSolverState:
    iterations: int | jax.Array
    vals: VarValues
    cost: float | jax.Array
    residual_vector: jax.Array
    termination_criteria: jax.Array
    termination_deltas: jax.Array
    lambd: float | jax.Array

    # Conjugate gradient state. Not used for other solvers.
    cg_state: _ConjugateGradientState | None


@jdc.pytree_dataclass
class NonlinearSolver:
    """Helper class for solving using Gauss-Newton or Levenberg-Marquardt."""

    linear_solver: jdc.Static[
        Literal["conjugate_gradient", "cholmod", "dense_cholesky"]
    ]
    trust_region: TrustRegionConfig | None
    termination: TerminationConfig
    conjugate_gradient_config: ConjugateGradientConfig | None
    sparse_mode: jdc.Static[Literal["blockrow", "coo", "csr"]]
    verbose: jdc.Static[bool]

    @jdc.jit
    def solve(
        self, problem: AnalyzedLeastSquaresProblem, initial_vals: VarValues
    ) -> VarValues:
        vals = initial_vals
        residual_vector = problem.compute_residual_vector(vals)

        state = NonlinearSolverState(
            iterations=0,
            vals=vals,
            cost=jnp.sum(residual_vector**2),
            residual_vector=residual_vector,
            termination_criteria=jnp.array([False, False, False, False]),
            termination_deltas=jnp.zeros(3),
            lambd=self.trust_region.lambda_initial
            if self.trust_region is not None
            else 0.0,
            cg_state=None
            if self.linear_solver != "conjugate_gradient"
            else _ConjugateGradientState(
                ATb_norm_prev=0.0,
                eta=(
                    ConjugateGradientConfig()
                    if self.conjugate_gradient_config is None
                    else self.conjugate_gradient_config
                ).tolerance_max,
            ),
        )

        # Optimization.
        state = jax.lax.while_loop(
            cond_fun=lambda state: jnp.logical_not(jnp.any(state.termination_criteria)),
            body_fun=functools.partial(self.step, problem),
            init_val=state,
        )
        if self.verbose:
            jax_log(
                "Terminated @ iteration #{i}: cost={cost:.4f} criteria={criteria}, term_deltas={cost_delta:.1e},{grad_mag:.1e},{param_delta:.1e}",
                i=state.iterations,
                cost=state.cost,
                criteria=state.termination_criteria.astype(jnp.int32),
                cost_delta=state.termination_deltas[0],
                grad_mag=state.termination_deltas[1],
                param_delta=state.termination_deltas[2],
            )
        return state.vals

    def step(
        self, problem: AnalyzedLeastSquaresProblem, state: NonlinearSolverState
    ) -> NonlinearSolverState:
        # Get nonzero values of Jacobian.
        A_blocksparse = problem._compute_jac_values(state.vals)

        # Get flattened version for COO/CSR matrices.
        jac_values = jnp.concatenate(
            [
                block_row.blocks_concat.flatten()
                for block_row in A_blocksparse.block_rows
            ],
            axis=0,
        )

        # linear_transpose() will return a tuple, with one element per primal.
        if self.sparse_mode == "blockrow":
            A_multiply = A_blocksparse.multiply
            AT_multiply_ = jax.linear_transpose(
                A_multiply, jnp.zeros((A_blocksparse.shape[1],))
            )
            AT_multiply = lambda vec: AT_multiply_(vec)[0]
        elif self.sparse_mode == "coo":
            A_coo = SparseCooMatrix(
                values=jac_values, coords=problem.jac_coords_coo
            ).as_jax_bcoo()
            AT_coo = A_coo.transpose()
            A_multiply = lambda vec: A_coo @ vec
            AT_multiply = lambda vec: AT_coo @ vec
        elif self.sparse_mode == "csr":
            A_csr = SparseCsrMatrix(
                values=jac_values, coords=problem.jac_coords_csr
            ).as_jax_bcsr()
            A_multiply = lambda vec: A_csr @ vec
            AT_multiply_ = jax.linear_transpose(
                A_multiply, jnp.zeros((A_blocksparse.shape[1],))
            )
            AT_multiply = lambda vec: AT_multiply_(vec)[0]
        else:
            assert_never(self.sparse_mode)

        # Compute right-hand side of normal equation.
        ATb = -AT_multiply(state.residual_vector)

        linear_state = None
        if (
            isinstance(self.linear_solver, ConjugateGradientConfig)
            or self.linear_solver == "conjugate_gradient"
        ):
            # Use default CG config is specified as a string, otherwise use the provided config.
            cg_config = (
                ConjugateGradientConfig()
                if self.linear_solver == "conjugate_gradient"
                else self.linear_solver
            )
            assert isinstance(state.cg_state, _ConjugateGradientState)
            local_delta, linear_state = cg_config._solve(
                problem,
                A_blocksparse,
                # We could also use (lambd * ATA_diagonals * vec) for
                # scale-invariant damping. But this is hard to match with CHOLMOD.
                lambda vec: AT_multiply(A_multiply(vec)) + state.lambd * vec,
                ATb=ATb,
                prev_linear_state=state.cg_state,
            )
        elif self.linear_solver == "cholmod":
            # Use CHOLMOD for direct solve.
            A_csr = SparseCsrMatrix(jac_values, problem.jac_coords_csr)
            local_delta = _cholmod_solve(A_csr, ATb, lambd=state.lambd)
        elif self.linear_solver == "dense_cholesky":
            A_dense = A_blocksparse.to_dense()
            ATA = A_dense.T @ A_dense
            diag_idx = jnp.arange(ATA.shape[0])
            ATA = ATA.at[diag_idx, diag_idx].add(state.lambd)
            cho_factor = jax.scipy.linalg.cho_factor(ATA)
            local_delta = jax.scipy.linalg.cho_solve(cho_factor, ATb)
        else:
            assert_never(self.linear_solver)

        vals = state.vals._retract(local_delta, problem.tangent_ordering)
        if self.verbose:
            if state.cg_state is None:
                jax_log(
                    " step #{i}: cost={cost:.4f} lambd={lambd:.4f} term_deltas={cost_delta:.1e},{grad_mag:.1e},{param_delta:.1e}",
                    i=state.iterations,
                    cost=state.cost,
                    lambd=state.lambd,
                    cost_delta=state.termination_deltas[0],
                    grad_mag=state.termination_deltas[1],
                    param_delta=state.termination_deltas[2],
                    ordered=True,
                )
            else:
                jax_log(
                    " step #{i}: cost={cost:.4f} lambd={lambd:.4f} term_deltas={cost_delta:.1e},{grad_mag:.1e},{param_delta:.1e} inexact_tol={inexact_tol:.1e}",
                    i=state.iterations,
                    cost=state.cost,
                    lambd=state.lambd,
                    cost_delta=state.termination_deltas[0],
                    grad_mag=state.termination_deltas[1],
                    param_delta=state.termination_deltas[2],
                    inexact_tol=state.cg_state.eta,
                    ordered=True,
                )
            residual_index = 0
            for f, count in zip(problem.stacked_costs, problem.cost_counts):
                stacked_dim = count * f.residual_flat_dim
                partial_cost = jnp.sum(
                    state.residual_vector[residual_index : residual_index + stacked_dim]
                    ** 2
                )
                residual_index += stacked_dim
                jax_log(
                    "     - "
                    + f"{f._get_name()}({count}):".ljust(15)
                    + " {:.5f} (avg {:.5f})",
                    partial_cost,
                    partial_cost / stacked_dim,
                    ordered=True,
                )

        with jdc.copy_and_mutate(state) as state_next:
            proposed_residual_vector = problem.compute_residual_vector(vals)
            proposed_cost = jnp.sum(proposed_residual_vector**2)

            # Update ATb_norm for Eisenstat-Walker criterion.
            if linear_state is not None:
                state_next.cg_state = linear_state

            # Always accept Gauss-Newton steps.
            if self.trust_region is None:
                state_next.vals = vals
                state_next.residual_vector = proposed_residual_vector
                state_next.cost = proposed_cost
                accept_flag = None
            # For Levenberg-Marquardt, we need to evaluate the step quality.
            else:
                step_quality = (proposed_cost - state.cost) / (
                    jnp.sum(
                        (A_blocksparse.multiply(local_delta) + state.residual_vector)
                        ** 2
                    )
                    - state.cost
                )
                accept_flag = step_quality >= self.trust_region.step_quality_min

                state_next.vals = jax.tree.map(
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

            # Compute termination criteria.
            state_next.termination_criteria, state_next.termination_deltas = (
                self.termination._check_convergence(
                    state,
                    cost_updated=proposed_cost,
                    tangent=local_delta,
                    tangent_ordering=problem.tangent_ordering,
                    ATb=ATb,
                    accept_flag=accept_flag,
                )
            )

            state_next.iterations += 1
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
    cost_tolerance: float = 1e-5
    """We terminate if `|cost change| / cost < cost_tolerance`."""
    gradient_tolerance: float = 1e-4
    """We terminate if `norm_inf(x - rplus(x, linear delta)) < gradient_tolerance`."""
    gradient_tolerance_start_step: int = 10
    """When to start checking the gradient tolerance condition. Helps solve precision
    issues caused by inexact Newton steps."""
    parameter_tolerance: float = 1e-6
    """We terminate if `norm_2(linear delta) < (norm2(x) + parameter_tolerance) * parameter_tolerance`."""

    def _check_convergence(
        self,
        state_prev: NonlinearSolverState,
        cost_updated: jax.Array,
        tangent: jax.Array,
        tangent_ordering: VarTypeOrdering,
        ATb: jax.Array,
        accept_flag: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Check for convergence!"""

        # Cost tolerance
        cost_absdelta = jnp.abs(cost_updated - state_prev.cost)
        cost_reldelta = cost_absdelta / state_prev.cost
        converged_cost = cost_reldelta < self.cost_tolerance

        # Gradient tolerance
        flat_vals = jax.flatten_util.ravel_pytree(state_prev.vals)[0]
        gradient_mag = jnp.max(
            flat_vals
            - jax.flatten_util.ravel_pytree(
                state_prev.vals._retract(ATb, tangent_ordering)
            )[0]
        )
        converged_gradient = jnp.where(
            state_prev.iterations >= self.gradient_tolerance_start_step,
            gradient_mag < self.gradient_tolerance,
            False,
        )

        # Parameter tolerance
        param_delta = jnp.linalg.norm(jnp.abs(tangent)) / (
            jnp.linalg.norm(flat_vals) + self.parameter_tolerance
        )
        converged_parameters = param_delta < self.parameter_tolerance

        # Check termination flags. We'll terminate if any of the conditions are met.
        term_flags = jnp.array(
            [
                converged_cost,
                converged_gradient,
                converged_parameters,
                state_prev.iterations >= (self.max_iterations - 1),
            ]
        )

        # Only consider the first three conditions if steps are accepted.
        if accept_flag is not None:
            term_flags = term_flags.at[:3].set(
                jnp.logical_and(
                    term_flags[:3],
                    # We ignore accept_flag if the cost _actually_ didn't change at all.
                    jnp.logical_or(accept_flag, cost_absdelta == 0.0),
                )
            )

        return term_flags, jnp.array([cost_reldelta, gradient_mag, param_delta])
