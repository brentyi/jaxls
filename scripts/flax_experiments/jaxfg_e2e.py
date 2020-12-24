# Some options
# > Nested optimization

import numpy as onp
from jaxfg import core, geometry, solvers, utils
from jaxlie import SE2

T_ab = SE2.from_xy_theta(1.0, 0.0, 0.2)
T_bc = SE2.from_xy_theta(1.0, 0.0, 0.2)

T_wa = SE2.identity()
T_wb = T_wa @ T_ab
T_wc = T_wb @ T_bc

A = geometry.SE2Variable()
B = geometry.SE2Variable()
C = geometry.SE2Variable()


def sample_se2_noise():
    return SE2.exp(onp.random.randn(SE2.tangent_dim) * 0.1)


graph = core.PreparedFactorGraph.from_factors(
    [
        geometry.PriorFactor.make(
            variable=A, mu=SE2.identity(), scale_tril_inv=onp.eye(3)
        ),
        geometry.BetweenFactor.make(
            before=A, after=B, delta=T_ab, scale_tril_inv=onp.eye(3)
        ),
        geometry.BetweenFactor.make(
            before=B, after=C, delta=T_bc, scale_tril_inv=onp.eye(3)
        ),
        # Noisy factors
        geometry.PriorFactor.make(
            variable=B, mu=T_wb @ sample_se2_noise(), scale_tril_inv=onp.eye(3)
        ),
        geometry.PriorFactor.make(
            variable=C, mu=T_wc @ sample_se2_noise(), scale_tril_inv=onp.eye(3)
        ),
    ]
)

ground_truth_assignments = core.VariableAssignments.from_dict(
    {
        A: T_wa,
        B: T_wb,
        C: T_wc,
    }
)

print(graph.compute_sum_squared_error(ground_truth_assignments))

graph.solve(ground_truth_assignments)
