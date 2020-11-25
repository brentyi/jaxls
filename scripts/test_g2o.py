import dataclasses
import time

import jax
import jaxfg
import matplotlib.pyplot as plt
import numpy as onp
from jax import numpy as jnp
from tqdm.auto import tqdm

with open("./data/input_M3500_g2o.g2o") as file:
    lines = [line.strip() for line in file.readlines()]

pose_variables = []
initial_poses: jaxfg.types.VariableAssignments = {}

factors = []

pose_count = 80000

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
        initial_poses[variable] = onp.array([x, y, onp.cos(theta), onp.sin(theta)])
        pose_variables.append(variable)

    elif parts[0] == "EDGE_SE2":
        before_index = int(parts[1])
        after_index = int(parts[2])

        if before_index >= pose_count:
            continue
        if after_index >= pose_count:
            continue

        delta = onp.array(list(map(float, parts[3:6])))

        q11, q12, q13, q22, q23, q33 = map(float, parts[6:])
        information_matrix = onp.array(
            [
                [q11, q12, q13],
                [q12, q22, q23],
                [q13, q23, q33],
            ]
        )
        scale_tril_inv = onp.linalg.cholesky(information_matrix)

        print(before_index, after_index)

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
        scale_tril_inv=jnp.eye(3),
    )
)

print(f"Loaded {len(pose_variables)} poses and {len(factors)} factors")


start_time = time.time()

initial_poses = jaxfg.types.VariableAssignments.from_dict(initial_poses)
solution_poses = jaxfg.FactorGraph().with_factors(*factors).solve(initial_poses)

print("====\nSolve time: ", time.time() - start_time, "\n====")
# print(solution_poses)

with jaxfg.utils.stopwatch("Converting storage to onp"):
    solution_poses = dataclasses.replace(
        solution_poses, storage=onp.array(solution_poses.storage)
    )


print("plotting!")
plt.figure()
# from tqdm.auto import tqdm
# print(onp.array(solution_poses.storage).shape)
# exit()
for i, v in enumerate(tqdm(pose_variables)):
    x, y, cos, sin = initial_poses.get_value(v)
    plt.arrow(x, y, cos * 0.1, sin * 0.1, width=0.05, head_width=0.1, color="r")
    # plt.annotate(str(i), (x, y))
for i, v in enumerate(tqdm(pose_variables)):
    x, y, cos, sin = solution_poses.get_value(v)
    plt.arrow(x, y, cos * 0.1, sin * 0.1, width=0.05, head_width=0.1, color="b")
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
