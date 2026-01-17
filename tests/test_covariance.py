"""Tests for covariance estimation."""

import jax
import jax.numpy as jnp
import jaxlie
import jaxls


class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(1)):
    """Simple scalar variable for testing."""

    pass


def test_simple_linear_covariance():
    """Test covariance estimation on a simple linear problem with known solution.

    For a linear problem y = Ax + noise, where noise has variance sigma^2,
    the covariance of x is (A^T A)^{-1} * sigma^2.
    """
    # Create a simple problem: two scalar measurements of a scalar variable.
    # y1 = x + noise1, y2 = x + noise2
    # J = [[1], [1]], so J^T J = [[2]]
    # (J^T J)^{-1} = [[0.5]]
    var = ScalarVar(0)

    @jaxls.Cost.factory
    def measurement_cost(
        vals: jaxls.VarValues, var: ScalarVar, target: float
    ) -> jax.Array:
        return vals[var] - target

    # Two measurements: 1.0 and 2.0.
    costs = [
        measurement_cost(var, 1.0),
        measurement_cost(var, 2.0),
    ]

    problem = jaxls.LeastSquaresProblem(costs, [var]).analyze()
    solution = problem.solve(verbose=False)

    # Solution should be mean of measurements: 1.5.
    assert jnp.allclose(solution[var], jnp.array([1.5]), atol=1e-5)

    # Test CG-based covariance estimator.
    estimator_cg = problem.make_covariance_estimator(
        solution, scale_by_residual_variance=False
    )
    cov_cg = estimator_cg.covariance(var)

    # (J^T J)^{-1} = [[0.5]] for this problem.
    assert cov_cg.shape == (1, 1)
    assert jnp.allclose(cov_cg, jnp.array([[0.5]]), atol=1e-4)

    # Test dense Cholesky estimator gives same result.
    estimator_dense = problem.make_covariance_estimator(
        solution,
        method=jaxls.LinearSolverCovarianceEstimatorConfig(
            linear_solver="dense_cholesky"
        ),
        scale_by_residual_variance=False,
    )
    cov_dense = estimator_dense.covariance(var)
    assert jnp.allclose(cov_cg, cov_dense, atol=1e-6)


def test_two_variable_covariance():
    """Test covariance with two variables and cross-covariance."""
    var0 = ScalarVar(0)
    var1 = ScalarVar(1)

    @jaxls.Cost.factory
    def prior_cost(vals: jaxls.VarValues, var: ScalarVar, target: float) -> jax.Array:
        return vals[var] - target

    @jaxls.Cost.factory
    def relative_cost(
        vals: jaxls.VarValues, var0: ScalarVar, var1: ScalarVar, delta: float
    ) -> jax.Array:
        return vals[var1] - vals[var0] - delta

    # Prior on var0 at 0, prior on var1 at 2, relative constraint of 1.
    costs = [
        prior_cost(var0, 0.0),
        prior_cost(var1, 2.0),
        relative_cost(var0, var1, 1.0),
    ]

    problem = jaxls.LeastSquaresProblem(costs, [var0, var1]).analyze()
    solution = problem.solve(verbose=False)

    # Test covariance estimator.
    estimator = problem.make_covariance_estimator(
        solution, scale_by_residual_variance=False
    )

    # Marginal covariances.
    cov_00 = estimator.covariance(var0)
    cov_11 = estimator.covariance(var1)
    assert cov_00.shape == (1, 1)
    assert cov_11.shape == (1, 1)

    # Cross-covariance.
    cov_01 = estimator.covariance(var0, var1)
    cov_10 = estimator.covariance(var1, var0)
    assert cov_01.shape == (1, 1)
    assert cov_10.shape == (1, 1)

    # Cross-covariance should be symmetric.
    assert jnp.allclose(cov_01, cov_10.T, atol=1e-5)


def test_pose_graph_covariance():
    """Test covariance estimation on a pose graph problem.

    Verifies that CG, dense Cholesky, and CHOLMOD all give consistent results.
    """
    var0 = jaxls.SE2Var(0)
    var1 = jaxls.SE2Var(1)

    @jaxls.Cost.factory
    def prior_cost(
        vals: jaxls.VarValues, var: jaxls.SE2Var, target: jaxlie.SE2
    ) -> jax.Array:
        return (vals[var] @ target.inverse()).log()

    @jaxls.Cost.factory
    def between_cost(
        vals: jaxls.VarValues,
        var0: jaxls.SE2Var,
        var1: jaxls.SE2Var,
        delta: jaxlie.SE2,
    ) -> jax.Array:
        return ((vals[var0].inverse() @ vals[var1]) @ delta.inverse()).log()

    costs = [
        prior_cost(var0, jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
        prior_cost(var1, jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
        between_cost(var0, var1, jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0)),
    ]

    problem = jaxls.LeastSquaresProblem(costs, [var0, var1]).analyze()
    solution = problem.solve(verbose=False)

    # Test CG estimator (default).
    estimator_cg = problem.make_covariance_estimator(
        solution, scale_by_residual_variance=False
    )

    # SE2 has tangent_dim = 3.
    cov_cg_00 = estimator_cg.covariance(var0)
    cov_cg_11 = estimator_cg.covariance(var1)
    cov_cg_01 = estimator_cg.covariance(var0, var1)
    assert cov_cg_00.shape == (3, 3)
    assert cov_cg_11.shape == (3, 3)
    assert cov_cg_01.shape == (3, 3)

    # Covariance should be positive semi-definite.
    eigvals_00 = jnp.linalg.eigvalsh(cov_cg_00)
    eigvals_11 = jnp.linalg.eigvalsh(cov_cg_11)
    assert jnp.all(eigvals_00 >= -1e-10)
    assert jnp.all(eigvals_11 >= -1e-10)

    # Test dense Cholesky estimator.
    estimator_dense = problem.make_covariance_estimator(
        solution,
        method=jaxls.LinearSolverCovarianceEstimatorConfig(
            linear_solver="dense_cholesky"
        ),
        scale_by_residual_variance=False,
    )
    cov_dense_00 = estimator_dense.covariance(var0)
    cov_dense_11 = estimator_dense.covariance(var1)
    cov_dense_01 = estimator_dense.covariance(var0, var1)

    # CG and dense should match.
    assert jnp.allclose(cov_cg_00, cov_dense_00, atol=1e-4)
    assert jnp.allclose(cov_cg_11, cov_dense_11, atol=1e-4)
    assert jnp.allclose(cov_cg_01, cov_dense_01, atol=1e-4)

    # Test CHOLMOD estimator if available.
    estimator_cholmod = problem.make_covariance_estimator(
        solution, method="cholmod_spinv", scale_by_residual_variance=False
    )
    cov_cholmod_00 = estimator_cholmod.covariance(var0)
    cov_cholmod_11 = estimator_cholmod.covariance(var1)
    cov_cholmod_01 = estimator_cholmod.covariance(var0, var1)

    # CHOLMOD should match CG.
    assert jnp.allclose(cov_cg_00, cov_cholmod_00, atol=1e-4)
    assert jnp.allclose(cov_cg_11, cov_cholmod_11, atol=1e-4)
    assert jnp.allclose(cov_cg_01, cov_cholmod_01, atol=1e-4)


def test_cg_and_dense_consistency():
    """Test that CG and dense Cholesky give the same results."""
    var0 = jaxls.SE2Var(0)
    var1 = jaxls.SE2Var(1)

    @jaxls.Cost.factory
    def prior_cost(
        vals: jaxls.VarValues, var: jaxls.SE2Var, target: jaxlie.SE2
    ) -> jax.Array:
        return (vals[var] @ target.inverse()).log()

    costs = [
        prior_cost(var0, jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
        prior_cost(var1, jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0)),
    ]

    problem = jaxls.LeastSquaresProblem(costs, [var0, var1]).analyze()
    solution = problem.solve(verbose=False)

    # CG estimator.
    estimator_cg = problem.make_covariance_estimator(
        solution, scale_by_residual_variance=False
    )

    # Dense Cholesky estimator.
    estimator_dense = problem.make_covariance_estimator(
        solution,
        method=jaxls.LinearSolverCovarianceEstimatorConfig(
            linear_solver="dense_cholesky"
        ),
        scale_by_residual_variance=False,
    )

    # Compare results.
    cov_cg_00 = estimator_cg.covariance(var0)
    cov_dense_00 = estimator_dense.covariance(var0)
    assert jnp.allclose(cov_cg_00, cov_dense_00, atol=1e-4)

    cov_cg_01 = estimator_cg.covariance(var0, var1)
    cov_dense_01 = estimator_dense.covariance(var0, var1)
    assert jnp.allclose(cov_cg_01, cov_dense_01, atol=1e-4)


def test_residual_variance_scaling():
    """Test that residual variance scaling works correctly."""
    var = ScalarVar(0)

    @jaxls.Cost.factory
    def measurement_cost(
        vals: jaxls.VarValues, var: ScalarVar, target: float
    ) -> jax.Array:
        return vals[var] - target

    costs = [
        measurement_cost(var, 1.0),
        measurement_cost(var, 2.0),
    ]

    problem = jaxls.LeastSquaresProblem(costs, [var]).analyze()
    solution = problem.solve(verbose=False)

    # Without scaling.
    estimator_unscaled = problem.make_covariance_estimator(
        solution, scale_by_residual_variance=False
    )
    cov_unscaled = estimator_unscaled.covariance(var)

    # With scaling.
    estimator_scaled = problem.make_covariance_estimator(
        solution, scale_by_residual_variance=True
    )
    cov_scaled = estimator_scaled.covariance(var)

    # Scaled covariance should be unscaled * residual_variance.
    # residual_variance = ||r||^2 / (m - n) where m=2, n=1.
    residual = problem.compute_residual_vector(solution)
    residual_variance = jnp.sum(residual**2) / (2 - 1)

    assert jnp.allclose(cov_scaled, cov_unscaled * residual_variance, atol=1e-6)


def test_cholmod_consistency():
    """Test that cholmod_spinv gives same results as CG."""
    var0 = jaxls.SE2Var(0)
    var1 = jaxls.SE2Var(1)

    @jaxls.Cost.factory
    def prior_cost(
        vals: jaxls.VarValues, var: jaxls.SE2Var, target: jaxlie.SE2
    ) -> jax.Array:
        return (vals[var] @ target.inverse()).log()

    costs = [
        prior_cost(var0, jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
        prior_cost(var1, jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0)),
    ]

    problem = jaxls.LeastSquaresProblem(costs, [var0, var1]).analyze()
    solution = problem.solve(verbose=False)

    # CG estimator.
    estimator_cg = problem.make_covariance_estimator(
        solution, scale_by_residual_variance=False
    )

    # CHOLMOD estimator.
    estimator_cholmod = problem.make_covariance_estimator(
        solution, method="cholmod_spinv", scale_by_residual_variance=False
    )

    # For diagonal blocks (marginal covariances), cholmod should match CG.
    cov_cg_00 = estimator_cg.covariance(var0)
    cov_cholmod_00 = estimator_cholmod.covariance(var0)
    assert jnp.allclose(cov_cg_00, cov_cholmod_00, atol=1e-4)

    cov_cg_11 = estimator_cg.covariance(var1)
    cov_cholmod_11 = estimator_cholmod.covariance(var1)
    assert jnp.allclose(cov_cg_11, cov_cholmod_11, atol=1e-4)
