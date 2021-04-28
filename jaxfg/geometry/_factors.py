import dataclasses
from typing import Generic, NamedTuple, Tuple, TypeVar

import jax
import jaxlie
from overrides import final, overrides

from .. import hints, utils
from ..core._factors import FactorBase
from ._lie_variables import LieVariableBase

LieGroupType = TypeVar("LieGroupType", bound=jaxlie.MatrixLieGroup)
PriorValueTuple = Tuple[LieGroupType]


@dataclasses.dataclass
class PriorFactor(FactorBase, Generic[LieGroupType]):
    """Factor for defining a fixed prior on a frame.

    Residuals are computed as `(variable.inverse() @ mu).log()`.
    """

    mu: LieGroupType

    @staticmethod
    def make(
        variable: LieVariableBase[LieGroupType],
        mu: LieGroupType,
        scale_tril_inv: hints.ScaleTrilInv,
    ) -> "PriorFactor[LieGroupType]":
        return PriorFactor(
            variables=(variable,),
            mu=mu,
            scale_tril_inv=scale_tril_inv,
        )

    @final
    @overrides
    def compute_residual_vector(self, variable_values: PriorValueTuple) -> hints.Array:

        value: LieGroupType
        (value,) = variable_values

        # Equivalent to: return (variable_value.inverse() @ self.mu).log()
        return jaxlie.manifold.rminus(value, self.mu)

    @final
    @overrides
    def compute_residual_jacobians(
        self, variable_values: PriorValueTuple
    ) -> Tuple[hints.Array]:

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


class BetweenVariableTuple(NamedTuple):
    variable_T_world_a: LieVariableBase
    variable_T_world_b: LieVariableBase


BetweenValueTuple = Tuple[LieGroupType, LieGroupType]


@dataclasses.dataclass
class BetweenFactor(FactorBase, Generic[LieGroupType]):
    """Factor for defining a geometric relationship between frames `a` and `b`.

    Residuals are computed as `((T_world_a @ T_a_b).inverse() @ T_world_b).log()`.
    """

    variables: BetweenVariableTuple
    T_a_b: LieGroupType

    @staticmethod
    def make(
        variable_T_world_a: LieVariableBase[LieGroupType],
        variable_T_world_b: LieVariableBase[LieGroupType],
        T_a_b: LieGroupType,
        scale_tril_inv: hints.ScaleTrilInv,
    ) -> "BetweenFactor[LieGroupType]":
        assert type(variable_T_world_a) is type(variable_T_world_b)
        assert variable_T_world_a.get_group_type() is type(T_a_b)

        return BetweenFactor(
            variables=BetweenVariableTuple(
                variable_T_world_a=variable_T_world_a,
                variable_T_world_b=variable_T_world_b,
            ),
            T_a_b=T_a_b,
            scale_tril_inv=scale_tril_inv,
        )

    @jax.jit
    @final
    @overrides
    def compute_residual_vector(
        self, variable_values: BetweenValueTuple
    ) -> hints.Array:
        T_world_a, T_world_b = variable_values
        # Equivalent to: return ((T_world_a @ self.T_a_b).inverse() @ T_world_b).log()
        return jaxlie.manifold.rminus(T_world_a @ self.T_a_b, T_world_b)

    @final
    @overrides
    def compute_residual_jacobians(
        self, variable_values: BetweenValueTuple
    ) -> Tuple[hints.Array, hints.Array]:
        T_world_a, T_world_b = variable_values

        # Helper for using analytical `rplus` Jacobians
        #
        # Implementing this is totally optional -- we should get the same results even
        # with this function commented out!

        J_residual_wrt_a, J_residual_wrt_b = (
            J.parameters()
            for J in jax.jacfwd(BetweenFactor.compute_residual_vector, argnums=1)(
                self, (T_world_a, T_world_b)
            )
        )

        J_a_wrt_delta, J_b_wrt_delta = jax.vmap(
            jaxlie.manifold.rplus_jacobian_parameters_wrt_delta
        )(utils.pytree_stack(T_world_a, T_world_b))

        return (J_residual_wrt_a @ J_a_wrt_delta, J_residual_wrt_b @ J_b_wrt_delta)
