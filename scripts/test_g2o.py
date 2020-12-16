import dataclasses
import time
from typing import List

import jax
import jax.profiler
import jaxfg
import jaxlie
import matplotlib.pyplot as plt
import numpy as onp
from jax import numpy as jnp
from jax.config import config
from tqdm.auto import tqdm

# config.update("jax_enable_x64", True)


with open("./data/input_M3500_g2o.g2o") as file:
    lines = [line.strip() for line in file.readlines()]

pose_variables = []
initial_poses: jaxfg.types.VariableAssignments = {}

factors: List[jaxfg.FactorBase] = []

pose_count = 10000


for line in tqdm(lines):
    parts = line.split(" ")
    if parts[0] == "VERTEX_SE2":
        _, index, x, y, theta = parts
        index = int(index)

        if index >= pose_count:
            continue

        x, y, theta = map(float, [x, y, theta])

        assert len(initial_poses) == index

        variable = jaxfg.SE2Variable()

        initial_poses[variable] = jax.jit(jaxlie.SE2.from_xy_theta)(
            x, y, theta
        ).xy_unit_complex

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

        delta = jax.jit(jaxlie.SE2.from_xy_theta)(
            *(float(p) for p in parts[3:6])
        ).xy_unit_complex

        q11, q12, q13, q22, q23, q33 = map(float, parts[6:])
        information_matrix = onp.array(
            [
                [q11, q12, q13],
                [q12, q22, q23],
                [q13, q23, q33],
            ]
        )
        scale_tril_inv = onp.linalg.cholesky(information_matrix).T

        # scale_tril_inv = jnp.array(onp.array(map(float, parts[6:6])))
        factors.append(
            jaxfg.BetweenFactor.make(
                before=pose_variables[before_index],
                after=pose_variables[after_index],
                delta=delta,
                scale_tril_inv=scale_tril_inv,
            )
        )

# Anchor start pose
factors.append(
    jaxfg.PriorFactor.make(
        variable=pose_variables[0],
        mu=initial_poses[pose_variables[0]],
        scale_tril_inv=jnp.eye(3) * 100.0,
    )
)
print("Prior factor:", initial_poses[pose_variables[0]])

print(f"Loaded {len(pose_variables)} poses and {len(factors)} factors")

print("Initial cost")

initial_poses = jaxfg.types.VariableAssignments.from_dict(initial_poses)
graph = jaxfg.FactorGraph().with_factors(*factors)

with jaxfg.utils.stopwatch("stacked"):
    errors_list = []
    for group_key, group in graph.factors_from_group.items():
        # Stack factors in our group
        factors_stacked: jaxfg.FactorBase = jax.tree_multimap(
            lambda *arrays: onp.stack(arrays, axis=0), *group
        )
        example_factor = next(iter(group))
        # Stack inputs to our factors
        values_indices = tuple([] for _ in range(len(example_factor.variables)))
        for factor in group:
            for i, variable in enumerate(factor.variables):
                storage_pos = initial_poses.storage_pos_from_variable[variable]
                values_indices[i].append(
                    onp.arange(
                        storage_pos, storage_pos + variable.get_parameter_dim()
                    ).reshape(variable.get_parameter_shape())
                )
        values_stacked = tuple(
            initial_poses.storage[onp.array(indices)] for indices in values_indices
        )
        # Vectorized error computation
        print(factors_stacked.scale_tril_inv[0])
        errors_list.append(
            jnp.einsum(
                "nij,nj->ni",
                factors_stacked.scale_tril_inv,
                jax.vmap(group_key.factor_type.compute_error)(
                    factors_stacked, *values_stacked
                ),
            ).flatten()
        )

    print(jnp.sum(jnp.concatenate(errors_list) ** 2) * 0.5)

# with jaxfg.utils.stopwatch("sequential"):
#     cost = 0.0
#
#     for factor in tqdm(list(graph.factors)):
#         cost = cost + jnp.sum(
#             (
#                 factor.scale_tril_inv
#                 @ factor.compute_error(
#                     *(
#                         initial_poses.get_value(variable)
#                         for variable in factor.variables
#                     )
#                 )
#             )
#             ** 2
#         )
#
#     print(cost * 0.5)

with jaxfg.utils.stopwatch("prepare"):
    prepared = graph.prepare()
with jaxfg.utils.stopwatch("prepared GN step"):
    prepared._gauss_newton_step(initial_poses)
with jaxfg.utils.stopwatch("prepared solve"):
    solution_poses = prepared.solve(initial_poses)

with jaxfg.utils.stopwatch("GN step JIT build"):
    graph._gauss_newton_step(initial_poses)

with jaxfg.utils.stopwatch("Solve"):
    graph.solve(initial_poses)

# print(solution_poses)

with jaxfg.utils.stopwatch("Converting storage to onp"):
    solution_poses = dataclasses.replace(
        solution_poses, storage=onp.array(solution_poses.storage)
    )


print("Plotting!")
plt.figure()

plt.title("Optimization on M3500 dataset, Olson et al. 2006")
plt.plot(
    *(onp.array([initial_poses.get_value(v)[:2] for v in pose_variables]).T),
    c="r",
    label="Dead-reckoned",
)
plt.plot(
    *(onp.array([solution_poses.get_value(v)[:2] for v in pose_variables]).T),
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
