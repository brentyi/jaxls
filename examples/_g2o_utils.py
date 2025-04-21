# from __future__ import annotations

import dataclasses
import pathlib
from typing import cast

import jax
import jaxlie
import jaxls
import numpy as onp
from tqdm.auto import tqdm


@dataclasses.dataclass
class G2OData:
    costs: list[jaxls.Cost]
    initial_poses: list[jaxlie.MatrixLieGroup]
    pose_vars: list[jaxls.Var]


def parse_g2o(path: pathlib.Path, pose_count_limit: int = 100000) -> G2OData:
    """Parse a G2O file. Creates a list of factors and dictionary of initial poses."""

    with open(path) as file:
        lines = [line.strip() for line in file.readlines()]

    var_count = 0
    factors = list[jaxls.Cost]()
    pose_variables = list[jaxls.Var]()
    initial_poses = list[jaxlie.MatrixLieGroup]()

    for line in tqdm(lines):
        parts = [part for part in line.split(" ") if part != ""]

        if parts[0] == "VERTEX_SE2":
            if len(pose_variables) > pose_count_limit:
                continue

            # Create SE(2) variable
            _, index, x, y, theta = parts
            index = int(index)
            x, y, theta = map(float, [x, y, theta])
            assert len(initial_poses) == index
            variable = jaxls.SE2Var(id=var_count)
            var_count += 1

            initial_poses.append(jaxlie.SE2.from_xy_theta(x, y, theta))
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

            factor = jaxls.Cost(
                # Passing in arrays like sqrt_precision_matrix as input makes
                # it possible vectorize factors.
                (
                    lambda values,
                    T_world_a,
                    T_world_b,
                    between,
                    sqrt_precision_matrix: sqrt_precision_matrix
                    @ (
                        (values[T_world_a].inverse() @ values[T_world_b]).inverse()
                        @ between
                    ).log()
                ),
                args=(
                    pose_variables[before_index],
                    pose_variables[after_index],
                    between,
                    cast(jax.Array, sqrt_precision_matrix),
                ),
                jac_mode="forward",
            )
            factors.append(factor)

        elif parts[0] == "VERTEX_SE3:QUAT":
            # Create SE(3) variable
            _, index, x, y, z, qx, qy, qz, qw = parts
            index = int(index)
            assert len(initial_poses) == index
            variable = jaxls.SE3Var(id=var_count)
            initial_poses.append(
                jaxlie.SE3(
                    wxyz_xyz=onp.array(list(map(float, [qw, qx, qy, qz, x, y, z])))  # type: ignore
                )
            )
            pose_variables.append(variable)
            var_count += 1

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

            factor = jaxls.Cost(
                # Passing in arrays like sqrt_precision_matrix as input makes
                # it possible for jaxfg vectorize factors.
                (
                    lambda values,
                    T_world_a,
                    T_world_b,
                    between,
                    sqrt_precision_matrix: sqrt_precision_matrix
                    @ (
                        (values[T_world_a].inverse() @ values[T_world_b]).inverse()
                        @ between
                    ).log()
                ),
                args=(
                    pose_variables[before_index],
                    pose_variables[after_index],
                    between,
                    cast(jax.Array, sqrt_precision_matrix),
                ),
                jac_mode="forward",
            )
            factors.append(factor)
        else:
            assert False, f"Unexpected line type: {parts[0]}"

    # Anchor start pose
    factor = jaxls.Cost(
        lambda var_values, start_pose: (
            var_values[start_pose].inverse() @ initial_poses[0]
        ).log(),
        args=(pose_variables[0],),
        jac_mode="reverse",
    )
    factors.append(factor)

    return G2OData(costs=factors, initial_poses=initial_poses, pose_vars=pose_variables)


__all__ = ["G2OData", "parse_g2o"]
