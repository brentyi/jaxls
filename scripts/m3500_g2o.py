import dataclasses
from typing import Dict, List

import fannypack
import jax
import jaxlie
import matplotlib.pyplot as plt
import numpy as onp
from jax import numpy as jnp
from tqdm.auto import tqdm

import jaxfg

# config.update("jax_enable_x64", True)


with open("./data/input_M3500_g2o.g2o") as file:
    lines = [line.strip() for line in file.readlines()]

pose_variables = []
initial_poses_dict: Dict[jaxfg.geometry.SE2Variable, jaxlie.SE2] = {}

factors: List[jaxfg.core.FactorBase] = []

pose_count = 10000


for line in tqdm(lines):
    parts = line.split(" ")
    if parts[0] == "VERTEX_SE2":
        _, index, x, y, theta = parts
        index = int(index)

        if index >= pose_count:
            continue

        x, y, theta = map(float, [x, y, theta])

        assert len(initial_poses_dict) == index

        variable = jaxfg.geometry.SE2Variable()

        initial_poses_dict[variable] = jax.jit(jaxlie.SE2.from_xy_theta)(x, y, theta)

        pose_variables.append(variable)

    elif parts[0] == "EDGE_SE2":
        before_index = int(parts[1])
        after_index = int(parts[2])

        if before_index >= pose_count:
            continue
        if after_index >= pose_count:
            continue
        # if after_index != before_index + 1:
        #     continue

        between = jaxlie.SE2.from_xy_theta(*(float(p) for p in parts[3:6]))

        information_matrix_components = onp.array(list(map(float, parts[6:])))
        information_matrix = onp.zeros((3, 3))
        information_matrix[onp.triu_indices(3)] = information_matrix_components
        information_matrix = information_matrix.T
        information_matrix[onp.triu_indices(3)] = information_matrix_components
        scale_tril_inv = onp.linalg.cholesky(information_matrix).T

        # scale_tril_inv = jnp.array(onp.array(map(float, parts[6:6])))
        factors.append(
            jaxfg.geometry.BetweenFactor.make(
                variable_T_world_a=pose_variables[before_index],
                variable_T_world_b=pose_variables[after_index],
                T_a_b=between,
                scale_tril_inv=scale_tril_inv,
            )
        )
    else:
        assert False, f"Unexpected line type: {parts[0]}"

# Anchor start pose
factors.append(
    jaxfg.geometry.PriorFactor.make(
        variable=pose_variables[0],
        mu=initial_poses_dict[pose_variables[0]],
        scale_tril_inv=jnp.eye(3) * 100,
    )
)
print("Prior factor:", initial_poses_dict[pose_variables[0]])

print(f"Loaded {len(pose_variables)} poses and {len(factors)} factors")

print("Initial cost")

fannypack.utils.pdb_safety_net()
initial_poses = jaxfg.core.VariableAssignments.make_from_dict(initial_poses_dict)
graph = jaxfg.core.StackedFactorGraph.make(factors)

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

plt.title("Optimization on M3500 dataset, Olson et al. 2006")

plt.plot(
    *(initial_poses.get_stacked_value(jaxfg.geometry.SE2Variable).translation().T),
    # Equivalent:
    # *(onp.array([initial_poses.get_value(v).translation() for v in pose_variables]).T),
    c="r",
    label="Dead-reckoned",
)
plt.plot(
    *(solution_poses.get_stacked_value(jaxfg.geometry.SE2Variable).translation().T),
    # Equivalent:
    # *(onp.array([solution_poses.get_value(v).translation() for v in pose_variables]).T),
    c="b",
    label="Optimized",
)


# for i, v in enumerate(tqdm(pose_variables)):
#     x, y, cos, sin = initial_poses.get_value(v)
#     plt.arrow(x, y, cos * 0.1, sin * 0.1, width=0.05, head_width=0.1, color="r")
#     # plt.annotate(str(i), (x, y))
# for i, v in enumerate(tqdm(pose_variables)):
#     x, y, cos, sin = solution_poses.get_value(v)
#     plt.arrow(x, y, cos * 0.1, sin * 0.1, width=0.05, head_width=0.1, color="b")
# plt.annotate(str(i), (x, y))
# plt.plot(
#     [initial_poses[v][0] for v in pose_variables],
#     [initial_poses[v][1] for v in pose_variables],
#     label="Initial poses",
# )
# plt.plot(
#     [solution_poses[v][0] for v in pose_variables],
#     [solution_poses[v][1] for v in pose_variables],
#     label="Solution poses",
# )
plt.legend()
plt.show()
