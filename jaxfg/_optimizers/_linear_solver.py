import jax
import jax.numpy as jnp


def sparse_linear_solve(
    A_values: jnp.ndarray,
    A_coords: jnp.ndarray,
    initial_x: jnp.ndarray,
    b: jnp.ndarray,
    tol=1e-5,
) -> jnp.ndarray:
    """Solves a block-sparse `Ax = b` least squares problem via CGLS."""

    assert len(A_values.shape) == 1, "A_values should be 1D"
    assert A_coords.shape == (
        A_values.shape[0],
        2,
    ), "A_coords should be rows of (row, col)"
    assert len(b.shape) == 1, "b should be 1D!"

    def ATA_function(x: jnp.ndarray):
        # Compute Ax
        Ax = jnp.zeros_like(b).at[A_coords[:, 0]].add(A_values * x[A_coords[:, 1]])

        # Compute ATAx
        ATAx = jnp.zeros_like(x).at[A_coords[:, 1]].add(A_values * Ax[A_coords[:, 0]])

        return ATAx

    # Compute ATb
    ATb = jnp.zeros_like(initial_x).at[A_coords[:, 1]].add(A_values * b[A_coords[:, 0]])

    # Basic Jacobi preconditioning, using diagonals of ATA
    diagonals = jnp.zeros_like(initial_x).at[A_coords[:, 1]].add(A_values ** 2)

    def jacobi_preconditioner(x):
        return x / diagonals

    # Solve with conjugate gradient
    solution_values, _unused_info = jax.scipy.sparse.linalg.cg(
        A=ATA_function,
        b=ATb,
        x0=initial_x,
        maxiter=len(initial_x),  # Default value used by Eigen
        tol=tol,
        M=jacobi_preconditioner,
    )
    return solution_values
