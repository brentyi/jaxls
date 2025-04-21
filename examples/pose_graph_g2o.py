"""Example for solving pose graph optimization problems loaded from `.g2o` files.

For a summary of options:

    python pose_graph_g2o.py --help

"""

import pathlib
from typing import Literal

import jax
import jaxls
import matplotlib.pyplot as plt
import tyro

import _g2o_utils

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def main(
    g2o_path: pathlib.Path = pathlib.Path(__file__).parent / "data/input_M3500_g2o.g2o",
    linear_solver: Literal[
        "conjugate_gradient", "cholmod", "dense_cholesky"
    ] = "conjugate_gradient",
) -> None:
    # Parse g2o file.
    with jaxls.utils.stopwatch("Reading g2o file"):
        g2o: _g2o_utils.G2OData = _g2o_utils.parse_g2o(g2o_path)
        jax.block_until_ready(g2o)

    # Making problem.
    with jaxls.utils.stopwatch("Analyzing problem"):
        problem = jaxls.LeastSquaresProblem(
            costs=g2o.costs, variables=g2o.pose_vars
        ).analyze()
        jax.block_until_ready(problem)

    with jaxls.utils.stopwatch("Making solver"):
        initial_vals = jaxls.VarValues.make(
            (
                pose_var.with_value(pose)
                for pose_var, pose in zip(g2o.pose_vars, g2o.initial_poses)
            )
        )
        jax.block_until_ready(initial_vals)

    with jaxls.utils.stopwatch("Running solve"):
        solution_vals = problem.solve(
            initial_vals, trust_region=None, linear_solver=linear_solver
        )
        jax.block_until_ready(solution_vals)

    with jaxls.utils.stopwatch("Running solve (again)"):
        solution_vals = problem.solve(
            initial_vals, trust_region=None, linear_solver=linear_solver
        )
        jax.block_until_ready(solution_vals)

    # Plot
    plt.figure()
    if isinstance(g2o.pose_vars[0], jaxls.SE2Var):
        plt.plot(
            *(initial_vals.get_stacked_value(jaxls.SE2Var).translation().T),
            c="r",
            label="Initial",
        )
        plt.plot(
            *(solution_vals.get_stacked_value(jaxls.SE2Var).translation().T),
            # Equivalent:
            # *(onp.array([solution_poses.get_value(v).translation() for v in pose_variables]).T),
            c="b",
            label="Optimized",
        )

    # Visualize 3D poses
    elif isinstance(g2o.pose_vars[0], jaxls.SE3Var):
        ax = plt.axes(projection="3d")
        ax.set_box_aspect((1.0, 1.0, 1.0))  # type: ignore
        ax.plot3D(  # type: ignore
            *(initial_vals.get_stacked_value(jaxls.SE3Var).translation().T),
            c="r",
            label="Initial",
        )
        ax.plot3D(  # type: ignore
            *(solution_vals.get_stacked_value(jaxls.SE3Var).translation().T),
            c="b",
            label="Optimized",
        )

    else:
        assert False

    plt.title(f"Optimization on {g2o_path.stem}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    tyro.cli(main)
