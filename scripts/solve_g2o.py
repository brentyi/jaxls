"""Example for solving pose graph optimization problems loaded from `.g2o` files.

For a summary of options:


"""

import argparse
import dataclasses
import enum
import pathlib

import datargs
import matplotlib.pyplot as plt

import _g2o_utils
import jaxfg


class SolverType(enum.Enum):
    GAUSS_NEWTON = jaxfg.solvers.GaussNewtonSolver()
    LEVENBERG_MARQUARDT = jaxfg.solvers.LevenbergMarquardtSolver()

    @property
    def instance(self) -> jaxfg.solvers.NonlinearSolverBase:
        """Typed alias for `enum.value`."""
        assert isinstance(self.value, jaxfg.solvers.NonlinearSolverBase)
        return self.value


@datargs.argsclass(
    parser_params={"formatter_class": argparse.ArgumentDefaultsHelpFormatter}
)
@dataclasses.dataclass
class CliArgs:
    solver_type: SolverType = datargs.arg(
        default=SolverType.GAUSS_NEWTON,
        help="Nonlinear solver to use.",
    )
    g2o_path: pathlib.Path = datargs.arg(
        default=pathlib.Path(__file__).parent / "data/input_M3500_g2o.g2o",
        help="Path to g2o file.",
    )


def main(cli_args: CliArgs):
    # Read graph
    with jaxfg.utils.stopwatch("Reading g2o file"):
        g2o: _g2o_utils.G2OData = _g2o_utils.parse_g2o(cli_args.g2o_path)

    # Make factor graph and solve (twice!)
    with jaxfg.utils.stopwatch("Making factor graph"):
        graph = jaxfg.core.StackedFactorGraph.make(g2o.factors)

    with jaxfg.utils.stopwatch("Making initial poses"):
        initial_poses = jaxfg.core.VariableAssignments.make_from_dict(g2o.initial_poses)

    with jaxfg.utils.stopwatch("Solve"):
        solution_poses = graph.solve(
            initial_poses, solver=cli_args.solver_type.instance
        )

    with jaxfg.utils.stopwatch("Solve (again)"):
        solution_poses = graph.solve(
            initial_poses, solver=cli_args.solver_type.instance
        )

    # Plot
    plt.figure()

    if isinstance(next(iter(solution_poses.variables())), jaxfg.geometry.SE2Variable):
        # Visualize 2D poses
        plt.plot(
            *(
                initial_poses.get_stacked_value(jaxfg.geometry.SE2Variable)
                .translation()
                .T
            ),
            # Equivalent:
            # *(onp.array([initial_poses.get_value(v).translation() for v in pose_variables]).T),
            c="r",
            label="Dead-reckoned",
        )
        plt.plot(
            *(
                solution_poses.get_stacked_value(jaxfg.geometry.SE2Variable)
                .translation()
                .T
            ),
            # Equivalent:
            # *(onp.array([solution_poses.get_value(v).translation() for v in pose_variables]).T),
            c="b",
            label="Optimized",
        )
    elif isinstance(next(iter(solution_poses.variables())), jaxfg.geometry.SE3Variable):
        # Visualize 3D poses
        ax = plt.axes(projection="3d")
        ax.set_box_aspect((1, 1, 1))
        ax.plot3D(
            *(
                initial_poses.get_stacked_value(jaxfg.geometry.SE3Variable)
                .translation()
                .T
            ),
            c="r",
            label="Dead-reckoned",
        )
        ax.plot3D(
            *(
                solution_poses.get_stacked_value(jaxfg.geometry.SE3Variable)
                .translation()
                .T
            ),
            c="b",
            label="Optimized",
        )

    else:
        assert False

    plt.title(f"Optimization on {cli_args.g2o_path.stem}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(datargs.parse(CliArgs))
