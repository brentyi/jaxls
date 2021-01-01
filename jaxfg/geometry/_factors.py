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
    before: LieVariableBase
    after: LieVariableBase


@dataclasses.dataclass(frozen=True)
class BetweenFactor(FactorBase, Generic[LieGroupType]):
    """Factor for defining a geometric relationship between two frames.

    Nominally, we have:
        "before" -> `T_wb`
        "after" -> `T_wa`
        "between" -> `T_ba`

    Residuals are computed as `((T_wb @ T_ba).inverse() @ T_wa).log()`
    """

    variables: _BeforeAfterTuple
    between: jaxlie.MatrixLieGroup
    variable_type: Type[LieVariableBase]
    _static_fields = frozenset({"variable_type"})

    @staticmethod
    def make(
        before: LieVariableBase,
        after: LieVariableBase,
        between: types.VariableValue,
        scale_tril_inv: types.ScaleTrilInv,
    ):
        assert type(before) == type(after)
        return BetweenFactor(
            variables=_BeforeAfterTuple(before=before, after=after),
            between=between,
            scale_tril_inv=scale_tril_inv,
            variable_type=type(before),
        )

    @jax.jit
    @overrides
    def compute_residual_vector(
        self, before_value: jaxlie.MatrixLieGroup, after_value: jaxlie.MatrixLieGroup
    ) -> jnp.ndarray:
        # before is T_wb
        # between is T_ba
        # after is T_wa
        #
        # Our residual is the tangent-space difference between our actual computed and T_wa
        # transforms
        return ((before_value @ self.between).inverse() @ after_value).log()

    @overrides
    def compute_residual_jacobians(
        self, before_value: jaxlie.MatrixLieGroup, after_value: jaxlie.MatrixLieGroup
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        # Helper for using analytical `rplus` Jacobians
        #
        # Implementing this is totally optional -- we should get the same results even
        # with this function commented out!

        J_residual_wrt_before_value, J_residual_wrt_after_value = (
            J.parameters
            for J in jax.jacfwd(BetweenFactor.compute_residual_vector, argnums=(1, 2))(
                self, before_value, after_value
            )
        )

        J_before_value_wrt_delta = jaxlie.manifold.rplus_jacobian_parameters_wrt_delta(
            transform=before_value
        )

        J_after_value_wrt_delta = jaxlie.manifold.rplus_jacobian_parameters_wrt_delta(
            transform=after_value
        )

        return (
            J_residual_wrt_before_value @ J_before_value_wrt_delta,
            J_residual_wrt_after_value @ J_after_value_wrt_delta,
        )
