from typing import List

import jaxlie
import numpy as onp
from jax import numpy as jnp

import jaxfg


def _compute_pose_error_with_outlier(use_huber: bool) -> float:
    pose_variables = [
        jaxfg.geometry.SE2Variable(),
    ]

    gaussian_noise = jaxfg.noises.DiagonalGaussian.make_from_covariance(
        diagonal=jnp.ones(3)
    )
    huber_noise = jaxfg.noises.HuberWrapper(wrapped=gaussian_noise, delta=1.0)
    factors: List[jaxfg.core.FactorBase] = [
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[0],
            mu=jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0),
            noise_model=gaussian_noise,
        ),
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[0],
            mu=jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0),
            noise_model=gaussian_noise,
        ),
        # Outlier!!
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[0],
            mu=jaxlie.SE2.from_xy_theta(2000.0, 200.0, 0.0),
            noise_model=huber_noise if use_huber else gaussian_noise,
        ),
    ]

    graph = jaxfg.core.StackedFactorGraph.make(factors)
    initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(
        pose_variables
    )

    solution_assignments = graph.solve(
        initial_assignments,
        solver=jaxfg.solvers.GaussNewtonSolver(
            linear_solver=jaxfg.sparse.CholmodSolver()
        ),
    )
    assert type(repr(solution_assignments)) == str
    assert isinstance(solution_assignments.get_value(pose_variables[0]), jaxlie.SE2)
    assert isinstance(
        solution_assignments.get_stacked_value(jaxfg.geometry.SE2Variable), jaxlie.SE2
    )
    assert jnp.all(
        solution_assignments.get_value(pose_variables[0]).parameters()
        == solution_assignments.get_stacked_value(
            jaxfg.geometry.SE2Variable
        ).parameters()[0]
    )

    return float(
        onp.linalg.norm(solution_assignments.get_value(pose_variables[0]).log())
    )


def test_huber():
    """When outliers are present, using a Huber loss should result in dramatically lower
    errors."""
    assert (
        _compute_pose_error_with_outlier(use_huber=True)
        < _compute_pose_error_with_outlier(use_huber=False) / 100.0
    )
