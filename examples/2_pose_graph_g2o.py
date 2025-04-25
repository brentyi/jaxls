"""Example for solving pose graph optimization problems loaded from `.g2o` files.

For a summary of options:

    python pose_graph_g2o.py --help

"""

import pathlib
from typing import Literal

import jax
import jaxlie
import jaxls
import numpy as onp
import tyro
import viser

import _g2o_utils

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def main(
    g2o_path: pathlib.Path = pathlib.Path(__file__).parent / "data/sphere2500.g2o",
    linear_solver: Literal[
        "conjugate_gradient", "cholmod", "dense_cholesky"
    ] = "conjugate_gradient",
) -> None:
    # Parse g2o file.
    with jaxls.utils.stopwatch("Reading g2o file"):
        g2o: _g2o_utils.G2OData = _g2o_utils.parse_g2o(g2o_path)
        jax.block_until_ready(g2o)

    # Analyze problem.
    with jaxls.utils.stopwatch("Analyzing problem"):
        problem = jaxls.LeastSquaresProblem(
            costs=g2o.costs, variables=g2o.pose_vars
        ).analyze()
        jax.block_until_ready(problem)

    with jaxls.utils.stopwatch("Making initial values"):
        init_vals = jaxls.VarValues.make(
            (
                pose_var.with_value(pose)
                for pose_var, pose in zip(g2o.pose_vars, g2o.initial_poses)
            )
        )
        jax.block_until_ready(init_vals)

    with jaxls.utils.stopwatch("Running solve"):
        solution_vals = problem.solve(
            init_vals, trust_region=None, linear_solver=linear_solver
        )
        jax.block_until_ready(solution_vals)

    with jaxls.utils.stopwatch("Running solve (again)"):
        solution_vals = problem.solve(
            init_vals, trust_region=None, linear_solver=linear_solver
        )
        jax.block_until_ready(solution_vals)

    # Plot
    server = viser.ViserServer()
    if isinstance(g2o.pose_vars[0], jaxls.SE2Var):

        def lift_to_SE3(pose: jaxlie.SE2) -> jaxlie.SE3:
            lifted_matrix = onp.eye(4) + onp.zeros((*pose.get_batch_axes(), 4, 4))
            lifted_matrix[:, :2, :2] = pose.rotation().as_matrix()
            lifted_matrix[:, :2, 3] = pose.translation()
            return jaxlie.SE3.from_matrix(lifted_matrix)

        init_poses = lift_to_SE3(init_vals.get_stacked_value(jaxls.SE2Var))
        opt_poses = lift_to_SE3(solution_vals.get_stacked_value(jaxls.SE2Var))

    elif isinstance(g2o.pose_vars[0], jaxls.SE3Var):
        init_poses = init_vals.get_stacked_value(jaxls.SE3Var)
        opt_poses = solution_vals.get_stacked_value(jaxls.SE3Var)
    else:
        assert False

    position_scale = 0.5

    server.scene.add_batched_axes(
        "/init_points",
        batched_wxyzs=onp.array(init_poses.wxyz_xyz[:, :4]),
        batched_positions=onp.array(init_poses.wxyz_xyz[:, 4:]) * position_scale,
        axes_length=0.1,
        axes_radius=0.01,
    )
    server.scene.add_batched_axes(
        "/opt_points",
        batched_wxyzs=onp.array(opt_poses.wxyz_xyz[:, :4]),
        batched_positions=onp.array(opt_poses.wxyz_xyz[:, 4:]) * position_scale,
        axes_length=0.1,
        axes_radius=0.01,
    )

    positions = onp.array(init_poses.wxyz_xyz[:, 4:]) * position_scale
    server.scene.add_line_segments(
        "/init_spline",
        points=onp.stack([positions[:-1], positions[1:]], axis=1),
        colors=(0, 0, 200),
    )

    positions = onp.array(opt_poses.wxyz_xyz[:, 4:]) * position_scale
    server.scene.add_line_segments(
        "/opt_spline",
        points=onp.stack([positions[:-1], positions[1:]], axis=1),
        colors=(0, 200, 0),
    )

    server.sleep_forever()


if __name__ == "__main__":
    tyro.cli(main)
