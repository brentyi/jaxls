import dataclasses
from typing import NamedTuple, Tuple

import jax
import jaxlie
from jax import numpy as jnp
from overrides import overrides

from .. import noises, utils
from ..core._factor_base import FactorBase
from ._lie_variables import LieVariableBase

# jaxlie.MatrixLieGroup = TypeVar("jaxlie.MatrixLieGroup", bound=jaxlie.MatrixLieGroup)
PriorValueTuple = Tuple[jaxlie.MatrixLieGroup]


@dataclasses.dataclass
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

        value: jaxlie.MatrixLieGroup
        (value,) = variable_values

        # Equivalent to: return (variable_value.inverse() @ self.mu).log()
        return jaxlie.manifold.rminus(value, self.mu)

    @overrides
    def compute_residual_jacobians(
        self, variable_values: PriorValueTuple
    ) -> Tuple[jnp.ndarray]:
        (value,) = variable_values

        # Helper for using analytical `rplus` Jacobians
        #
        # Implementing this is totally optional -- we should get the same results even
        # with this function commented out!

        J_residual_wrt_value = jax.jacfwd(
            PriorFactor.compute_residual_vector, argnums=1
        )(self, (value,))[0].parameters()
        assert J_residual_wrt_value.shape == (
            value.tangent_dim,
            value.parameters_dim,
        )

        J_value_wrt_delta = jaxlie.manifold.rplus_jacobian_parameters_wrt_delta(
            transform=value
        )
        assert J_value_wrt_delta.shape == (
            value.parameters_dim,
            value.tangent_dim,
        )

        J_residual_wrt_delta = J_residual_wrt_value @ J_value_wrt_delta
        return (J_residual_wrt_delta,)


class BetweenValueTuple(NamedTuple):
    T_world_a: jaxlie.MatrixLieGroup
    T_world_b: jaxlie.MatrixLieGroup


@dataclasses.dataclass
class BetweenFactor(FactorBase[BetweenValueTuple]):
    """Factor for defining a geometric relationship between frames `a` and `b`.

    Residuals are computed as `((T_world_a @ T_a_b).inverse() @ T_world_b).log()`.
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

        # Equivalent to: return ((T_world_a @ self.T_a_b).inverse() @ T_world_b).log()
        return jaxlie.manifold.rminus(T_world_a @ self.T_a_b, T_world_b)

    @overrides
    def compute_residual_jacobians(
        self, variable_values: BetweenValueTuple
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        T_world_a = variable_values.T_world_a
        T_world_b = variable_values.T_world_b

        # Helper for using analytical `rplus` Jacobians
        #
        # Implementing this is totally optional -- we should get the same results even
        # with this function commented out!

        J_residual_wrt_a, J_residual_wrt_b = (
            J.parameters()
            for J in jax.jacfwd(BetweenFactor.compute_residual_vector, argnums=1)(
                self, BetweenValueTuple(T_world_a, T_world_b)
            )
        )

        J_a_wrt_delta, J_b_wrt_delta = jax.vmap(
            jaxlie.manifold.rplus_jacobian_parameters_wrt_delta
        )(utils.pytree_stack(T_world_a, T_world_b))

        return (J_residual_wrt_a @ J_a_wrt_delta, J_residual_wrt_b @ J_b_wrt_delta)
