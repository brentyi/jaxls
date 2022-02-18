import dataclasses
import pathlib
from typing import Dict, List

import jaxlie
import numpy as onp
from jax import numpy as jnp
from tqdm.auto import tqdm

import jaxfg


@dataclasses.dataclass
class G2OData:
    factors: List[jaxfg.core.FactorBase]
    initial_poses: Dict[jaxfg.geometry.LieVariableBase, jaxlie.MatrixLieGroup]


def parse_g2o(path: pathlib.Path, pose_count_limit: int = 100000) -> G2OData:
    """Parse a G2O file. Creates a list of factors and dictionary of initial poses."""

    with open(path) as file:
        lines = [line.strip() for line in file.readlines()]

    pose_variables: List[jaxfg.geometry.LieVariableBase] = []
    initial_poses: Dict[jaxfg.geometry.LieVariableBase, jaxlie.MatrixLieGroup] = {}

    factors: List[jaxfg.core.FactorBase] = []

    for line in tqdm(lines):
        parts = [part for part in line.split(" ") if part != ""]

        variable: jaxfg.geometry.LieVariableBase
        between: jaxlie.MatrixLieGroup

        if parts[0] == "VERTEX_SE2":
            if len(pose_variables) > pose_count_limit:
                continue

            # Create SE(2) variable
            _, index, x, y, theta = parts
            index = int(index)
            x, y, theta = map(float, [x, y, theta])
            assert len(initial_poses) == index
            variable = jaxfg.geometry.SE2Variable()
            initial_poses[variable] = jaxlie.SE2.from_xy_theta(x, y, theta)
            pose_variables.append(variable)

        elif parts[0] == "EDGE_SE2":
            # Create relative offset between pair of SE(2) variables
            before_index = int(parts[1])
            after_index = int(parts[2])

            if before_index > pose_count_limit or after_index > pose_count_limit:
                continue

            between = jaxlie.SE2.from_xy_theta(*(float(p) for p in parts[3:6]))

            precision_matrix_components = onp.array(list(map(float, parts[6:])))
            precision_matrix = onp.zeros((3, 3))
            precision_matrix[onp.triu_indices(3)] = precision_matrix_components
            precision_matrix = precision_matrix.T
            precision_matrix[onp.triu_indices(3)] = precision_matrix_components
            sqrt_precision_matrix = onp.linalg.cholesky(precision_matrix).T

            factors.append(
                jaxfg.geometry.BetweenFactor.make(
                    variable_T_world_a=pose_variables[before_index],
                    variable_T_world_b=pose_variables[after_index],
                    T_a_b=between,
                    noise_model=jaxfg.noises.Gaussian(
                        sqrt_precision_matrix=sqrt_precision_matrix
                    ),
                )
            )

        elif parts[0] == "VERTEX_SE3:QUAT":
            # Create SE(3) variable
            _, index, x, y, z, qx, qy, qz, qw = parts
            index = int(index)
            assert len(initial_poses) == index
            variable = jaxfg.geometry.SE3Variable()
            initial_poses[variable] = jaxlie.SE3(
                wxyz_xyz=onp.array(list(map(float, [qw, qx, qy, qz, x, y, z])))  # type: ignore
            )
            pose_variables.append(variable)

        elif parts[0] == "EDGE_SE3:QUAT":
            # Create relative offset between pair of SE(3) variables
            before_index = int(parts[1])
            after_index = int(parts[2])

            numerical_parts = list(map(float, parts[3:]))
            assert len(numerical_parts) == 7 + 21

            #  between = jaxlie.SE3.from_xy_theta(*(float(p) for p in parts[3:6]))

            xyz = numerical_parts[0:3]
            quaternion = numerical_parts[3:7]
            between = jaxlie.SE3.from_rotation_and_translation(
                rotation=jaxlie.SO3.from_quaternion_xyzw(onp.array(quaternion)),
                translation=onp.array(xyz),
            )

            precision_matrix = onp.zeros((6, 6))
            precision_matrix[onp.triu_indices(6)] = numerical_parts[7:]
            precision_matrix = precision_matrix.T
            precision_matrix[onp.triu_indices(6)] = numerical_parts[7:]

            sqrt_precision_matrix = onp.linalg.cholesky(precision_matrix).T

            factors.append(
                jaxfg.geometry.BetweenFactor.make(
                    variable_T_world_a=pose_variables[before_index],
                    variable_T_world_b=pose_variables[after_index],
                    T_a_b=between,
                    noise_model=jaxfg.noises.Gaussian(
                        sqrt_precision_matrix=sqrt_precision_matrix
                    ),
                )
            )
        else:
            assert False, f"Unexpected line type: {parts[0]}"

    # Anchor start pose
    factors.append(
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[0],
            mu=initial_poses[pose_variables[0]],
            noise_model=jaxfg.noises.DiagonalGaussian(
                jnp.ones(pose_variables[0].get_local_parameter_dim()) * 100.0
            ),
        )
    )

    return G2OData(factors=factors, initial_poses=initial_poses)


__all__ = ["G2OData", "parse_g2o"]
