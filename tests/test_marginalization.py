from typing import List

import jaxlie
import numpy as onp

import jaxfg


def test_marginalization() -> None:
    pose_variable = jaxfg.geometry.SE2Variable()
    pose_value = jaxlie.SE2.identity()

    sqrt_cov = onp.random.randn(3, 3)
    # sqrt_cov = onp.diag(onp.random.randn(3))
    cov = sqrt_cov @ sqrt_cov.T

    graph = jaxfg.core.StackedFactorGraph.make(
        factors=[
            jaxfg.geometry.PriorFactor.make(
                variable=pose_variable,
                mu=pose_value,
                noise_model=jaxfg.noises.Gaussian.make_from_covariance(covariance=cov),
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


def test_marginalization_double() -> None:
    pose_variable = jaxfg.geometry.SE2Variable()
    pose_value = jaxlie.SE2.identity()

    sqrt_cov = onp.random.randn(3, 3)
    cov = sqrt_cov @ sqrt_cov.T

    graph = jaxfg.core.StackedFactorGraph.make(
        factors=[
            jaxfg.geometry.PriorFactor.make(
                variable=pose_variable,
                mu=pose_value,
                noise_model=jaxfg.noises.Gaussian.make_from_covariance(covariance=cov),
            ),
            jaxfg.geometry.PriorFactor.make(
                variable=pose_variable,
                mu=pose_value,
                noise_model=jaxfg.noises.Gaussian.make_from_covariance(covariance=cov),
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


def test_marginalization_as_dense() -> None:
    pose_variables = [
        jaxfg.geometry.SE2Variable(),
        jaxfg.geometry.SE2Variable(),
    ]

    factors: List[jaxfg.core.FactorBase] = [
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[0],
            mu=jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0),
            noise_model=jaxfg.noises.Gaussian(onp.eye(3)),
        ),
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[1],
            mu=jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0),
            noise_model=jaxfg.noises.Gaussian(onp.eye(3)),
        ),
        jaxfg.geometry.BetweenFactor.make(
            variable_T_world_a=pose_variables[0],
            variable_T_world_b=pose_variables[1],
            T_a_b=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
            noise_model=jaxfg.noises.Gaussian(onp.eye(3)),
        ),
    ]

    graph = jaxfg.core.StackedFactorGraph.make(factors)
    initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(
        pose_variables
    )

    solution_assignments = graph.solve(initial_assignments)

    sparse_covariance = jaxfg.experimental.SparseCovariance.make(
        graph, solution_assignments
    )

    sqrt_information_matrix = graph.compute_whitened_residual_jacobian(
        assignments=solution_assignments,
        residual_vector=graph.compute_whitened_residual_vector(solution_assignments),
    ).as_dense()

    covariance0 = onp.linalg.inv(sqrt_information_matrix.T @ sqrt_information_matrix)
    covariance1 = sparse_covariance.as_dense(use_inverse=False)
    covariance2 = sparse_covariance.as_dense(use_inverse=True)

    assert covariance0.shape == (6, 6)
    onp.testing.assert_allclose(
        covariance0,
        covariance1,
        atol=1e-8,
        rtol=1e-5,
    )
    onp.testing.assert_allclose(
        covariance1,
        covariance2,
        atol=1e-8,
        rtol=1e-5,
    )
