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
    def compute_error(self, variable_value: jaxlie.MatrixLieGroup):
        return (variable_value.inverse() @ self.mu).log()

    @overrides
    def compute_error_jacobians(
        self, variable_value: jaxlie.MatrixLieGroup
    ) -> Tuple[jnp.ndarray]:
        # We want: J_error_wrt_delta
        # Which is: J_error_wrt_value @ J_value_wrt_delta

        J_error_wrt_value = jax.jacfwd(PriorFactor.compute_error, argnums=1)(
            self, variable_value
        ).parameters
        assert J_error_wrt_value.shape == (
            variable_value.tangent_dim,
            variable_value.parameters_dim,
        )

        J_value_wrt_delta = jaxlie.manifold.rplus_jacobian_wrt_delta_at_zero(
            transform=variable_value
        ).parameters
        assert J_value_wrt_delta.shape == (
            variable_value.parameters_dim,
            variable_value.tangent_dim,
        )

        J_error_wrt_delta = J_error_wrt_value @ J_value_wrt_delta
        return (J_error_wrt_delta,)


class _BeforeAfterTuple(NamedTuple):
    before: LieVariableBase
    after: LieVariableBase


@dataclasses.dataclass(frozen=True)
class BetweenFactor(FactorBase, Generic[LieGroupType]):
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
    def compute_error(
        self, before_value: jaxlie.MatrixLieGroup, after_value: jaxlie.MatrixLieGroup
    ):
        # before is T_wb
        # between is T_ba
        # after is T_wa
        #
        # Our error is the tangent-space difference between our actual computed and T_wa
        # transforms
        return ((before_value @ self.between).inverse() @ after_value).log()
