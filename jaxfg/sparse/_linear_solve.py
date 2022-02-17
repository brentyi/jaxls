import abc
from typing import Dict, Hashable, NamedTuple

import jax
import jax.experimental.host_callback as hcb
import jax_dataclasses as jdc
import numpy as onp
import sksparse
from jax import numpy as jnp
from overrides import EnforceOverrides, overrides

from .. import hints
from ._sparse_matrix import SparseCooMatrix


class LinearSubproblemSolverBase(abc.ABC, EnforceOverrides):
    """Linear solver base class."""

    @abc.abstractmethod
    def solve_subproblem(
        self,
        A: SparseCooMatrix,
        ATb: hints.Array,
        lambd: hints.Scalar,
        iteration: hints.Scalar,
    ) -> jnp.ndarray:
        """Solve a linear subproblem."""


class _LinearSolverArgs(NamedTuple):
    A: SparseCooMatrix
    ATb: hints.Array
    lambd: hints.Scalar


_cholmod_analyze_cache: Dict[Hashable, sksparse.cholmod.Factor] = {}


@jdc.pytree_dataclass
class CholmodSolver(LinearSubproblemSolverBase):
    r"""CHOLMOD-based sparse linear solver. This is the default solver for performance
    reasons, but also less stable than `ConjugateGradientSolver`.

    Runs via an XLA host callback, and has some usage caveats:
    - Caching is a little bit sketchy. Assumes that a given solver is not reused for
      systems with different sparsity patterns.
    - Does not support function transforms (`vmap`, `pmap`, etc), due to current
      limitations of `hcb.call()`.
    - Does not support autodiff. A custom JVP or VJP definition should be easy to
      implement, but not super useful without batch axis support.
    - Regularization consistency. We use a vanilla $$\lambda I$$ regularization term
      here, but the conjugate gradient solver uses a scale invariant $$\lambda
      diag(A^TA)$$ term.

    For applications where JAX transformations are necessary, ConjugateGradientSolver
    is written in vanilla JAX should be less caveat-y.
    """

    @overrides
    def solve_subproblem(
        self,
        A: SparseCooMatrix,
        ATb: hints.Array,
        lambd: hints.Scalar,
        iteration: hints.Scalar,  # Unused
    ) -> jnp.ndarray:
        # JAX-compatible sparse Cholesky factorization with a host callback. Similar to:
        #     self._solve(_LinearSolverArgs(A, ATb, lambd))
        return hcb.call(self._solve, _LinearSolverArgs(A, ATb, lambd), result_shape=ATb)

    def _solve(self, args: _LinearSolverArgs) -> jnp.ndarray:
        # Convert our custom sparse matrix format to a scipy CSC matrix
        # This could likely be optimized!
        A_T = args.A.T
        A_T_scipy = A_T.as_scipy_coo_matrix().tocsc(copy=False)

        # Cache sparsity pattern analysis
        self_hash = object.__hash__(self)
        if self_hash not in _cholmod_analyze_cache:
            _cholmod_analyze_cache[self_hash] = sksparse.cholmod.analyze_AAt(A_T_scipy)

        # Factorize and solve
        _cholmod_analyze_cache[self_hash].cholesky_AAt_inplace(
            A_T_scipy,
            beta=args.lambd
            + 1e-5,  # Some simple linear problems blow up without this 1e-5 term
        )
        return _cholmod_analyze_cache[self_hash].solve_A(args.ATb)


class _ConjugateGradientSolver(LinearSubproblemSolverBase, abc.ABC):
    @abc.abstractmethod
    def _get_cg_tolerance(self, iteration: hints.Scalar):
        ...

    @overrides
    def solve_subproblem(
        self,
        A: SparseCooMatrix,
        ATb: hints.Array,
        lambd: hints.Scalar,
        iteration: hints.Scalar,
    ) -> jnp.ndarray:
        assert len(A.values.shape) == 1, "A.values should be 1D"
        assert len(ATb.shape) == 1, "ATb should be 1D!"

        initial_x = onp.zeros(ATb.shape)

        # Get diagonals of ATA, for regularization + Jacobi preconditioning
        ATA_diagonals = jnp.zeros_like(initial_x).at[A.coords.cols].add(A.values**2)

        # Form normal equation
        def ATA_function(x: hints.Array):
            # Compute ATAx
            ATAx = A.T @ (A @ x)

            # Return regularized (scale-invariant)
            return ATAx + lambd * ATA_diagonals * x

            # Vanilla regularization
            # return ATAx + lambd * x

        def jacobi_preconditioner(x):
            return x / ATA_diagonals

        # Solve with conjugate gradient
        solution_values, _unused_info = jax.scipy.sparse.linalg.cg(
            A=ATA_function,
            b=ATb,
            x0=initial_x,
            maxiter=len(
                initial_x
            ),  # https://en.wikipedia.org/wiki/Conjugate_gradient_method#Convergence_properties
            tol=self._get_cg_tolerance(iteration),
            M=jacobi_preconditioner,
        )
        return solution_values


@jdc.pytree_dataclass
class ConjugateGradientSolver(_ConjugateGradientSolver):
    tolerance: float = 1e-5
    """CG convergence tolerance."""

    @overrides
    def _get_cg_tolerance(self, iteration: hints.Scalar):
        return self.tolerance


@jdc.pytree_dataclass
class InexactStepConjugateGradientSolver(_ConjugateGradientSolver):
    inexact_step_eta: float = 1e-2
    """Forcing sequence parameter for inexact Newton steps. CG tolerance is set to
    `eta / iteration #`.

    For reference, see AN INEXACT LEVENBERG-MARQUARDT METHOD FOR LARGE SPARSE NONLINEAR
    LEAST SQUARES, Wright & Holt 1983."""

    @overrides
    def _get_cg_tolerance(self, iteration: hints.Scalar):
        return self.inexact_step_eta / (iteration + 1)
