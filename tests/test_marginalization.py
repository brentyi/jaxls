import jaxlie
import numpy as onp

import jaxfg


def test_marginalization():

    pose_variable = jaxfg.geometry.SE2Variable()
    pose_value = jaxlie.SE2.identity()

    sqrt_cov = onp.random.randn(3, 3)
    # sqrt_cov = onp.diag(onp.random.randn(3))
    cov = sqrt_cov @ sqrt_cov.T
    scale_tril_inv = onp.linalg.inv(onp.linalg.cholesky(cov))

    graph = jaxfg.core.StackedFactorGraph.make(
        factors=[
            jaxfg.geometry.PriorFactor.make(
                variable=pose_variable, mu=pose_value, scale_tril_inv=scale_tril_inv
            )
        ]
    )
    assignments = jaxfg.core.VariableAssignments.make_from_dict(
        {pose_variable: pose_value}
    )

    cov_computed = jaxfg.experimental.SparseCovariance.make(
        graph, assignments
    ).compute_marginal(pose_variable)

    onp.testing.assert_allclose(cov, cov_computed, atol=1e-8, rtol=1e-5)


def test_marginalization_double():

    pose_variable = jaxfg.geometry.SE2Variable()
    pose_value = jaxlie.SE2.identity()

    sqrt_cov = onp.random.randn(3, 3)
    cov = sqrt_cov @ sqrt_cov.T
    scale_tril_inv = onp.linalg.inv(onp.linalg.cholesky(cov))

    graph = jaxfg.core.StackedFactorGraph.make(
        factors=[
            jaxfg.geometry.PriorFactor.make(
                variable=pose_variable, mu=pose_value, scale_tril_inv=scale_tril_inv
            ),
            jaxfg.geometry.PriorFactor.make(
                variable=pose_variable, mu=pose_value, scale_tril_inv=scale_tril_inv
            ),
        ]
    )
    assignments = jaxfg.core.VariableAssignments.make_from_dict(
        {pose_variable: pose_value}
    )

    cov_computed = jaxfg.experimental.SparseCovariance.make(
        graph, assignments
    ).compute_marginal(pose_variable)

    onp.testing.assert_allclose(cov, cov_computed * 2.0, atol=1e-8, rtol=1e-5)


def test_marginalization_as_dense():

    pose_variable = jaxfg.geometry.SE2Variable()
    pose_value = jaxlie.SE2.identity()

    sqrt_cov = onp.random.randn(3, 3)
    cov = sqrt_cov @ sqrt_cov.T
    scale_tril_inv = onp.linalg.inv(onp.linalg.cholesky(cov))

    graph = jaxfg.core.StackedFactorGraph.make(
        factors=[
            jaxfg.geometry.PriorFactor.make(
                variable=pose_variable, mu=pose_value, scale_tril_inv=scale_tril_inv
            ),
            jaxfg.geometry.PriorFactor.make(
                variable=pose_variable, mu=pose_value, scale_tril_inv=scale_tril_inv
            ),
        ]
    )
    assignments = jaxfg.core.VariableAssignments.make_from_dict(
        {pose_variable: pose_value}
    )

    covariance = jaxfg.experimental.SparseCovariance.make(graph, assignments)

    onp.testing.assert_allclose(
        covariance.as_dense(use_inverse=False),
        covariance.as_dense(use_inverse=True),
        atol=1e-8,
        rtol=1e-5,
    )
