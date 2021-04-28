import jax
import numpy as onp
import pytest
import scipy

import jaxfg


@pytest.mark.parametrize(
    "solver",
    [
        jaxfg.sparse.CholmodSolver(),
        jaxfg.sparse.ConjugateGradientSolver(inexact_step_eta=1e-8),
    ],
)
def test_solver_no_jit(solver: jaxfg.sparse.LinearSubproblemSolverBase):
    # Build sparse matrix
    A_shape = (20, 5)
    A_onp = onp.random.randn(*A_shape) * onp.random.randint(low=0, high=2, size=A_shape)
    A = jaxfg.sparse.SparseCooMatrix.from_scipy_coo_matrix(
        scipy.sparse.coo_matrix(A_onp)
    )
    ATb = onp.random.randn(A_shape[1])

    # Compute two separate ways
    x_ours = solver.solve_subproblem(
        A=A,
        ATb=ATb,
        lambd=0.0,
        iteration=0,  # Unused
    )
    x_onp = onp.linalg.solve(A_onp.T @ A_onp, ATb)

    # Validate
    onp.testing.assert_allclose(x_ours, x_onp, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "solver",
    [
        jaxfg.sparse.CholmodSolver(),
        jaxfg.sparse.ConjugateGradientSolver(inexact_step_eta=1e-8),
    ],
)
def test_solver_jit(solver: jaxfg.sparse.LinearSubproblemSolverBase):
    # Build sparse matrix
    A_shape = (20, 5)
    A_onp = onp.random.randn(*A_shape) * onp.random.randint(low=0, high=2, size=A_shape)
    A = jaxfg.sparse.SparseCooMatrix.from_scipy_coo_matrix(
        scipy.sparse.coo_matrix(A_onp)
    )
    ATb = onp.random.randn(A_shape[1])

    # Compute two separate ways
    @jax.jit
    def solve():
        # jax.jit(solver.solve_subproblem) does not work in Python 3.7
        return solver.solve_subproblem(
            A=A,
            ATb=ATb,
            lambd=0.0,
            iteration=0,  # Unused
        )

    x_ours = solve()
    x_onp = onp.linalg.solve(A_onp.T @ A_onp, ATb)

    # Validate
    onp.testing.assert_allclose(x_ours, x_onp, atol=1e-5, rtol=1e-5)
