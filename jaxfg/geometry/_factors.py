import dataclasses
from typing import TYPE_CHECKING, Generic, NamedTuple, Tuple, Type, TypeVar

import jax
import jaxlie
from jax import numpy as jnp
from overrides import overrides

from .. import types, utils
from ..core._factors import FactorBase
from ._lie_variables import LieVariableBase

LieGroupType = TypeVar("T", bound=jaxlie.MatrixLieGroup)


@dataclasses.dataclass(frozen=True)
class PriorFactor(FactorBase, Generic[LieGroupType]):
    """Factor for defining a fixed prior on a frame.

    Residuals are computed as `(variable.inverse() @ mu).log()`.
    """

    mu: jaxlie.MatrixLieGroup
    variable_type: Type[LieVariableBase]
    _static_fields = frozenset({"variable_type"})

    @staticmethod
    def make(
        variable: LieVariableBase,
        mu: jaxlie.MatrixLieGroup,
        scale_tril_inv: types.ScaleTrilInv,
    ):
        return PriorFactor(
            variables=(variable,),
            mu=mu,
            scale_tril_inv=scale_tril_inv,
            variable_type=type(variable),
        )

    @overrides
    def compute_residual_vector(
        self, variable_value: jaxlie.MatrixLieGroup
    ) -> jnp.ndarray:
        return (variable_value.inverse() @ self.mu).log()

    @overrides
    def compute_residual_jacobians(
        self, variable_value: jaxlie.MatrixLieGroup
    ) -> Tuple[jnp.ndarray]:

        # Helper for using analytical `rplus` Jacobians
        #
        # Implementing this is totally optional -- we should get the same results even
        # with this function commented out!

        J_residual_wrt_value = jax.jacfwd(
            PriorFactor.compute_residual_vector, argnums=1
        )(self, variable_value).parameters
        assert J_residual_wrt_value.shape == (
            variable_value.tangent_dim,
            variable_value.parameters_dim,
        )

        J_value_wrt_delta = jaxlie.manifold.rplus_jacobian_parameters_wrt_delta(
            transform=variable_value
        )
        assert J_value_wrt_delta.shape == (
            variable_value.parameters_dim,
            variable_value.tangent_dim,
        )

        J_residual_wrt_delta = J_residual_wrt_value @ J_value_wrt_delta
        return (J_residual_wrt_delta,)


class _BeforeAfterTuple(NamedTuple):
    variable_T_world_a: LieVariableBase
    variable_T_world_b: LieVariableBase


@dataclasses.dataclass(frozen=True)
class BetweenFactor(FactorBase, Generic[LieGroupType]):
    """Factor for defining a geometric relationship between frames `a` and `b`.

    Residuals are computed as `((T_world_a @ T_a_b).inverse() @ T_world_b).log()`.
    """

    variables: _BeforeAfterTuple
    T_a_b: jaxlie.MatrixLieGroup
    variable_type: Type[LieVariableBase]
    _static_fields = frozenset({"variable_type"})

    @staticmethod
    def make(
        variable_T_world_a: LieVariableBase,
        variable_T_world_b: LieVariableBase,
        T_a_b: jaxlie.MatrixLieGroup,
        scale_tril_inv: types.ScaleTrilInv,
    ):
        assert type(variable_T_world_a) is type(variable_T_world_b)
        assert variable_T_world_a.MatrixLieGroupType is type(T_a_b)

        return BetweenFactor(
            variables=_BeforeAfterTuple(
                variable_T_world_a=variable_T_world_a,
                variable_T_world_b=variable_T_world_b,
            ),
            T_a_b=T_a_b,
            scale_tril_inv=scale_tril_inv,
            variable_type=type(variable_T_world_a),
        )

    @jax.jit
    @overrides
    def compute_residual_vector(
        self, T_world_a: jaxlie.MatrixLieGroup, T_world_b: jaxlie.MatrixLieGroup
    ) -> jnp.ndarray:
        return ((T_world_a @ self.T_a_b).inverse() @ T_world_b).log()

    @overrides
    def compute_residual_jacobians(
        self, T_world_a: jaxlie.MatrixLieGroup, T_world_b: jaxlie.MatrixLieGroup
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        # Helper for using analytical `rplus` Jacobians
        #
        # Implementing this is totally optional -- we should get the same results even
        # with this function commented out!

        J_residual_wrt_a, J_residual_wrt_b = (
            J.parameters
            for J in jax.jacfwd(BetweenFactor.compute_residual_vector, argnums=(1, 2))(
                self, T_world_a, T_world_b
            )
        )

        J_a_wrt_delta, J_b_wrt_delta = jax.vmap(
            jaxlie.manifold.rplus_jacobian_parameters_wrt_delta
        )(utils.pytree_stack(T_world_a, T_world_b))

        return (J_residual_wrt_a @ J_a_wrt_delta, J_residual_wrt_b @ J_b_wrt_delta)
