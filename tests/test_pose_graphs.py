"""Simple pose graph tests. Could use cleanup/refactoring."""

from typing import List

import jaxlie
from jax import numpy as jnp

import jaxfg


def test_pose_graph_gauss_newton():
    pose_variables = [
        jaxfg.geometry.SE2Variable(),
        jaxfg.geometry.SE2Variable(),
    ]

    factors: List[jaxfg.core.FactorBase] = [
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[0],
            mu=jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(
                diagonal=[1.0, 1.0, 1.0]
            ),
        ),
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[1],
            mu=jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(
                diagonal=jnp.ones(3)
            ),
        ),
        jaxfg.geometry.BetweenFactor.make(
            variable_T_world_a=pose_variables[0],
            variable_T_world_b=pose_variables[1],
            T_a_b=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(
                diagonal=jnp.ones(3)
            ),
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
    assert graph.compute_joint_nll(initial_assignments) > graph.compute_joint_nll(
        solution_assignments
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
    assert jnp.all(
        solution_assignments.get_value(pose_variables[1]).parameters()
        == solution_assignments.get_stacked_value(
            jaxfg.geometry.SE2Variable
        ).parameters()[1]
    )


def test_pose_graph_levenberg_marquardt():
    pose_variables = [
        jaxfg.geometry.SE2Variable(),
        jaxfg.geometry.SE2Variable(),
    ]

    factors: List[jaxfg.core.FactorBase] = [
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[0],
            mu=jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(
                diagonal=jnp.ones(3)
            ),
        ),
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[1],
            mu=jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(
                diagonal=jnp.ones(3)
            ),
        ),
        jaxfg.geometry.BetweenFactor.make(
            variable_T_world_a=pose_variables[0],
            variable_T_world_b=pose_variables[1],
            T_a_b=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(
                diagonal=jnp.ones(3)
            ),
        ),
    ]

    graph = jaxfg.core.StackedFactorGraph.make(factors)
    initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(
        pose_variables
    )

    solution_assignments = graph.solve(
        initial_assignments,
        solver=jaxfg.solvers.LevenbergMarquardtSolver(
            linear_solver=jaxfg.sparse.ConjugateGradientSolver()
        ),
    )
    assert graph.compute_joint_nll(initial_assignments) > graph.compute_joint_nll(
        solution_assignments
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
    assert jnp.all(
        solution_assignments.get_value(pose_variables[1]).parameters()
        == solution_assignments.get_stacked_value(
            jaxfg.geometry.SE2Variable
        ).parameters()[1]
    )


def test_pose_graph_dogleg():
    pose_variables = [
        jaxfg.geometry.SE2Variable(),
        jaxfg.geometry.SE2Variable(),
    ]

    factors: List[jaxfg.core.FactorBase] = [
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[0],
            mu=jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(
                diagonal=jnp.ones(3)
            ),
        ),
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[1],
            mu=jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(
                diagonal=jnp.ones(3)
            ),
        ),
        jaxfg.geometry.BetweenFactor.make(
            variable_T_world_a=pose_variables[0],
            variable_T_world_b=pose_variables[1],
            T_a_b=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian.make_from_covariance(
                diagonal=jnp.ones(3)
            ),
        ),
    ]

    graph = jaxfg.core.StackedFactorGraph.make(factors)
    initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(
        pose_variables
    )

    solution_assignments = graph.solve(
        initial_assignments,
        solver=jaxfg.solvers.DoglegSolver(),
    )
    assert graph.compute_joint_nll(initial_assignments) > graph.compute_joint_nll(
        solution_assignments
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
    assert jnp.all(
        solution_assignments.get_value(pose_variables[1]).parameters()
        == solution_assignments.get_stacked_value(
            jaxfg.geometry.SE2Variable
        ).parameters()[1]
    )
