import dataclasses
from typing import Dict, List

import fannypack
import jaxlie
import matplotlib.pyplot as plt
import numpy as onp
from jax import numpy as jnp
from tqdm.auto import tqdm

import jaxfg

# config.update("jax_enable_x64", True)


with open("./data/parking-garage.g2o") as file:
    lines = [line.strip() for line in file.readlines()]

pose_variables = []
initial_poses_dict: Dict[jaxfg.geometry.SE3Variable, jaxlie.SE3] = {}

factors: List[jaxfg.core.FactorBase] = []

pose_count = 10000


for line in tqdm(lines):
    parts = line.split(" ")
    if parts[0] == "VERTEX_SE3:QUAT":
        _, index, x, y, z, qx, qy, qz, qw = parts
        index = int(index)

        if index >= pose_count:
            continue

        assert len(initial_poses_dict) == index

        variable = jaxfg.geometry.SE3Variable()

        initial_poses_dict[variable] = jaxlie.SE3(
            xyz_wxyz=onp.array(list(map(float, [x, y, z, qw, qx, qy, qz])))
        )

        pose_variables.append(variable)

    elif parts[0] == "EDGE_SE3:QUAT":
        before_index = int(parts[1])
        after_index = int(parts[2])

        if before_index >= pose_count:
            continue
        if after_index >= pose_count:
            continue
        # if after_index != before_index + 1:
        #     continue

        numerical_parts = list(map(float, parts[3:]))
        assert len(numerical_parts) == 7 + 21

        #  between = jaxlie.SE3.from_xy_theta(*(float(p) for p in parts[3:6]))

        xyz = numerical_parts[0:3]
        quaternion = numerical_parts[3:7]
        T_a_b = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3.from_quaternion_xyzw(onp.array(quaternion)),
            translation=onp.array(xyz),
        )

        information_matrix = onp.zeros((6, 6))
        information_matrix[onp.triu_indices(6)] = numerical_parts[7:]
        information_matrix = information_matrix.T
        information_matrix[onp.triu_indices(6)] = numerical_parts[7:]

        scale_tril_inv = onp.linalg.cholesky(information_matrix).T

        factors.append(
            jaxfg.geometry.BetweenFactor.make(
                variable_T_world_a=pose_variables[before_index],
                variable_T_world_b=pose_variables[after_index],
                T_a_b=T_a_b,
                scale_tril_inv=scale_tril_inv,
            )
        )

# Anchor start pose
factors.append(
    jaxfg.geometry.PriorFactor.make(
        variable=pose_variables[0],
        mu=initial_poses_dict[pose_variables[0]],
        scale_tril_inv=jnp.eye(6) * 100,
    )
)
print("Prior factor:", initial_poses_dict[pose_variables[0]])

print(f"Loaded {len(pose_variables)} poses and {len(factors)} factors")

print("Initial cost")

fannypack.utils.pdb_safety_net()
initial_poses = jaxfg.core.VariableAssignments.from_dict(initial_poses_dict)
graph = jaxfg.core.PreparedFactorGraph.from_factors(factors)

with jaxfg.utils.stopwatch("Compute residual"):
    print(jnp.sum(graph.compute_residual_vector(initial_poses) ** 2) * 0.5)


with jaxfg.utils.stopwatch("Solve"):
    solution_poses = graph.solve(
        initial_poses, solver=jaxfg.solvers.GaussNewtonSolver()
    )

with jaxfg.utils.stopwatch("Solve (#2)"):
    solution_poses = graph.solve(
        initial_poses, solver=jaxfg.solvers.GaussNewtonSolver()
    )

with jaxfg.utils.stopwatch("Converting storage to onp"):
    solution_poses = dataclasses.replace(
        solution_poses, storage=onp.array(solution_poses.storage)
    )


print("Plotting!")
plt.figure()
ax = plt.axes(projection="3d")

plt.title("Optimization on parking garage dataset")

ax.plot3D(
    *(onp.array([initial_poses.get_value(v).translation for v in pose_variables]).T),
    c="r",
    label="Dead-reckoned",
)
ax.plot3D(
    *(onp.array([solution_poses.get_value(v).translation for v in pose_variables]).T),
    c="b",
    label="Optimized",
)

plt.legend()
plt.show()
