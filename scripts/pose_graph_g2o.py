"""Example for solving pose graph optimization problems loaded from `.g2o` files.

For a summary of options:

    python pose_graph_g2o.py --help

"""

import pathlib

import jax
import jaxfg2
import matplotlib.pyplot as plt
import tyro

import _g2o_utils


def main(
    g2o_path: pathlib.Path = pathlib.Path(__file__).parent / "data/input_M3500_g2o.g2o",
) -> None:
    # Parse g2o file.
    with jaxfg2.utils.stopwatch("Reading g2o file"):
        g2o: _g2o_utils.G2OData = _g2o_utils.parse_g2o(g2o_path)
        jax.block_until_ready(g2o)

    # Making graph.
    with jaxfg2.utils.stopwatch("Making graph"):
        graph = jaxfg2.StackedFactorGraph.make(factors=g2o.factors, vars=g2o.pose_vars)
        jax.block_until_ready(graph)

    with jaxfg2.utils.stopwatch("Making solver"):
        solver = jaxfg2.GaussNewtonSolver(
            verbose=True
        )  # , linear_solver=jaxfg2.ConjugateGradientSolver())
        initial_vals = jaxfg2.VarValues.make(g2o.pose_vars, g2o.initial_poses)

    with jaxfg2.utils.stopwatch("Running solve"):
        solver.solve(graph, initial_vals)

    with jaxfg2.utils.stopwatch("Running solve (again)"):
        solution_vals = solver.solve(graph, initial_vals)

    # Plot
    plt.figure()
    if isinstance(g2o.pose_vars[0], jaxfg2.SE2Var):
        plt.plot(
            *(initial_vals.get_stacked_value(jaxfg2.SE2Var).translation().T),
            c="r",
            label="Initial",
        )
        plt.plot(
            *(solution_vals.get_stacked_value(jaxfg2.SE2Var).translation().T),
            # Equivalent:
            # *(onp.array([solution_poses.get_value(v).translation() for v in pose_variables]).T),
            c="b",
            label="Optimized",
        )

    # Visualize 3D poses
    elif isinstance(g2o.pose_vars[0], jaxfg2.SE3Var):
        ax = plt.axes(projection="3d")
        ax.set_box_aspect((1, 1, 1))
        ax.plot3D(
            *(initial_vals.get_stacked_value(jaxfg2.SE3Var).translation().T),
            c="r",
            label="Initial",
        )
        ax.plot3D(
            *(solution_vals.get_stacked_value(jaxfg2.SE3Var).translation().T),
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
