"""Example for solving pose graph optimization problems loaded from `.g2o` files.

For a summary of options:

    python pose_graph_g2o.py --help

"""
import dataclasses
import enum
import pathlib
from typing import Dict, Optional

import _g2o_utils
import dcargs
import matplotlib.pyplot as plt

import jaxfg


class SolverType(enum.Enum):
    GAUSS_NEWTON = enum.auto()
    FIXED_ITERATION_GAUSS_NEWTON = enum.auto()
    LEVENBERG_MARQUARDT = enum.auto()
    DOGLEG = enum.auto()

    def get_solver(self) -> jaxfg.solvers.NonlinearSolverBase:
        """Get solver corresponding to an enum."""
        map: Dict[SolverType, jaxfg.solvers.NonlinearSolverBase] = {
            SolverType.GAUSS_NEWTON: jaxfg.solvers.GaussNewtonSolver(),
            SolverType.FIXED_ITERATION_GAUSS_NEWTON: jaxfg.solvers.FixedIterationGaussNewtonSolver(
                unroll=False
            ),
            SolverType.LEVENBERG_MARQUARDT: jaxfg.solvers.LevenbergMarquardtSolver(),
            SolverType.DOGLEG: jaxfg.solvers.DoglegSolver(),
        }
        return map[self]


@dataclasses.dataclass
class CliArgs:
    g2o_path: pathlib.Path = pathlib.Path(__file__).parent / "data/input_M3500_g2o.g2o"
    """Path to g2o file."""

    solver_type: SolverType = SolverType.GAUSS_NEWTON
    """Nonlinear solver to use."""

    huber_delta: Optional[float] = None
    """Threshold for huber losses; if not set, a standard least-squares objective is used."""


def main():
    # Parse CLI args
    cli_args = dcargs.parse(CliArgs)

    # Read graph
    with jaxfg.utils.stopwatch("Reading g2o file"):
        g2o: _g2o_utils.G2OData = _g2o_utils.parse_g2o(cli_args.g2o_path)

        # Use Huber loss if applicable.
        if cli_args.huber_delta is not None:

            def wrap_huber(factor: jaxfg.core.FactorBase) -> jaxfg.core.FactorBase:
                """Apply a Huber loss to an existing factor."""
                return dataclasses.replace(
                    factor,
                    noise_model=jaxfg.noises.HuberWrapper(
                        wrapped=factor.noise_model, delta=cli_args.huber_delta
                    ),
                )

            g2o = dataclasses.replace(g2o, factors=list(map(wrap_huber, g2o.factors)))

    # Make factor graph
    with jaxfg.utils.stopwatch("Making factor graph"):
        graph = jaxfg.core.StackedFactorGraph.make(g2o.factors)

    with jaxfg.utils.stopwatch("Making initial poses"):
        initial_poses = jaxfg.core.VariableAssignments.make_from_dict(g2o.initial_poses)

    # Time solver
    if not isinstance(
        cli_args.solver_type.get_solver(), jaxfg.solvers.FixedIterationGaussNewtonSolver
    ):
        # `max_iterations` field exists for all solvers but the fixed iteration GN
        with jaxfg.utils.stopwatch("Single-step JIT compile + solve"):
            solution_poses = graph.solve(
                initial_poses,
                solver=dataclasses.replace(
                    cli_args.solver_type.get_solver(), max_iterations=1
                ),
            )
            solution_poses.storage.block_until_ready()

        with jaxfg.utils.stopwatch("Single-step solve (already compiled)"):
            solution_poses = graph.solve(
                initial_poses,
                solver=dataclasses.replace(
                    cli_args.solver_type.get_solver(), max_iterations=1
                ),
            )
            solution_poses.storage.block_until_ready()

    with jaxfg.utils.stopwatch("Full solve"):
        solution_poses = graph.solve(
            initial_poses, solver=cli_args.solver_type.get_solver()
        )
        solution_poses.storage.block_until_ready()

    # Plot
    plt.figure()

    # Visualize 2D poses
    if isinstance(
        next(iter(solution_poses.get_variables())), jaxfg.geometry.SE2Variable
    ):
        plt.plot(
            *(
                initial_poses.get_stacked_value(jaxfg.geometry.SE2Variable)
                .translation()
                .T
            ),
            # Equivalent:
            # *(onp.array([initial_poses.get_value(v).translation() for v in pose_variables]).T),
            c="r",
            label="Initial",
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

    # Visualize 3D poses
    elif isinstance(
        next(iter(solution_poses.get_variables())), jaxfg.geometry.SE3Variable
    ):
        ax = plt.axes(projection="3d")
        ax.set_box_aspect((1, 1, 1))
        ax.plot3D(
            *(
                initial_poses.get_stacked_value(jaxfg.geometry.SE3Variable)
                .translation()
                .T
            ),
            c="r",
            label="Initial",
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
    main()
