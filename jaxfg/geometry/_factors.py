from typing import NamedTuple, Tuple

import jax_dataclasses
import jaxlie
import numpy as onp
from jax import numpy as jnp
from overrides import overrides

from .. import noises
from ..core._factor_base import FactorBase
from ._lie_variables import LieVariableBase

PriorValueTuple = Tuple[jaxlie.MatrixLieGroup]


@jax_dataclasses.pytree_dataclass
class PriorFactor(FactorBase[PriorValueTuple]):
    """Factor for defining a fixed prior on a frame.

    Residuals are computed as `(variable.inverse() @ mu).log()`.
    """

    mu: jaxlie.MatrixLieGroup

    @staticmethod
    def make(
        variable: LieVariableBase,
        mu: jaxlie.MatrixLieGroup,
        noise_model: noises.NoiseModelBase,
    ) -> "PriorFactor":
        return PriorFactor(
            variables=(variable,),
            mu=mu,
            noise_model=noise_model,
        )

    @overrides
    def compute_residual_vector(self, variable_values: PriorValueTuple) -> jnp.ndarray:

        T: jaxlie.MatrixLieGroup
        (T,) = variable_values

        # Equivalent to: return (variable_value.inverse() @ self.mu).log()
        return jaxlie.manifold.rminus(T, self.mu)

    @overrides
    def compute_residual_jacobians(
        self, variable_values: PriorValueTuple
    ) -> Tuple[jnp.ndarray, ...]:
        # Implementing this is optional!
        # Autodiff will handle Jacobians if we don't specify analytical ones.

        T: jaxlie.MatrixLieGroup
        (T,) = variable_values
        return (-onp.eye(type(T).tangent_dim),)


class BetweenValueTuple(NamedTuple):
    T_world_a: jaxlie.MatrixLieGroup
    T_world_b: jaxlie.MatrixLieGroup


@jax_dataclasses.pytree_dataclass
class BetweenFactor(FactorBase[BetweenValueTuple]):
    """Factor for defining a geometric relationship between frames `a` and `b`.

    Residuals are computed as `((T_world_a.inverse() @ T_world_b).inverse() @ T_a_b).log()`.
    """

    T_a_b: jaxlie.MatrixLieGroup

    @staticmethod
    def make(
        variable_T_world_a: LieVariableBase,
        variable_T_world_b: LieVariableBase,
        T_a_b: jaxlie.MatrixLieGroup,
        noise_model: noises.NoiseModelBase,
    ) -> "BetweenFactor":
        assert type(variable_T_world_a) is type(variable_T_world_b)
        assert variable_T_world_a.get_group_type() is type(T_a_b)

        return BetweenFactor(
            variables=(
                variable_T_world_a,
                variable_T_world_b,
            ),
            T_a_b=T_a_b,
            noise_model=noise_model,
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: BetweenValueTuple
    ) -> jnp.ndarray:
        T_world_a = variable_values.T_world_a
        T_world_b = variable_values.T_world_b

        return jaxlie.manifold.rminus(T_world_a.inverse() @ T_world_b, self.T_a_b)

    @overrides
    def compute_residual_jacobians(
        self, variable_values: BetweenValueTuple
    ) -> Tuple[jnp.ndarray, ...]:
        # Implementing this is optional!
        # Autodiff will handle Jacobians if we don't specify analytical ones.

        T_world_a = variable_values.T_world_a
        T_world_b = variable_values.T_world_b

        assert type(T_world_a) == type(T_world_b)
        group_cls = type(T_world_a)

        return (
            (T_world_a.inverse() @ T_world_b).inverse().adjoint(),
            -onp.eye(group_cls.tangent_dim),
        )
