import jax
from jax import numpy as jnp

from .. import hints
from ._sparse_matrix import SparseCooMatrix


@jax.jit
def linear_solve(
    A: SparseCooMatrix,
    ATb: hints.Array,
    initial_x: hints.Array,
    tol: hints.Scalar,
    lambd: hints.Scalar,
) -> hints.Array:
    """Solves a block-sparse `Ax = b` least squares problem via CGLS.

    More specifically: solves `(A^TA + lambd * diag(A^TA)) x = b` if `diagonal_damping`
    is `True`, otherwise `(A^TA + lambd I) x = b`.

    TODO: consider adding `atol` term back in.
    """

    assert len(A.values.shape) == 1, "A.values should be 1D"
    assert len(ATb.shape) == 1, "ATb should be 1D!"

    # Get diagonals of ATA, for regularization + Jacobi preconditioning
    ATA_diagonals = jnp.zeros_like(initial_x).at[A.coords.cols].add(A.values ** 2)

    # Form normal equation
    def ATA_function(x: jnp.ndarray):
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
        tol=tol,
        M=jacobi_preconditioner,
    )
    return solution_values
