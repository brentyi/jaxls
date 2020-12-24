import dataclasses
from typing import TYPE_CHECKING, Generic, NamedTuple, Type, TypeVar

import jax
import jaxlie
from overrides import overrides

from .. import types, utils
from ..core._factors import FactorBase

if TYPE_CHECKING:
    from ..core._variables import VariableBase

LieGroupType = TypeVar("T", bound=jaxlie.MatrixLieGroup)


@utils.hashable
@dataclasses.dataclass(frozen=True)
class PriorFactor(FactorBase, Generic[LieGroupType]):
    mu: jaxlie.MatrixLieGroup
    variable_type: Type["VariableBase[LieGroupType]"]
    _static_fields = frozenset({"variable_type"})

    @staticmethod
    def make(
        variable: "VariableBase[LieGroupType]",
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
        return self.variable_type.subtract_local(variable_value, self.mu)


class _BeforeAfterTuple(NamedTuple):
    before: "VariableBase[LieGroupType]"
    after: "VariableBase[LieGroupType]"


@utils.hashable
@dataclasses.dataclass(frozen=True)
class BetweenFactor(FactorBase, Generic[LieGroupType]):
    variables: _BeforeAfterTuple
    delta: jaxlie.MatrixLieGroup
    variable_type: Type["VariableBase[LieGroupType]"]
    _static_fields = frozenset({"variable_type"})

    @staticmethod
    def make(
        before: "VariableBase[LieGroupType]",
        after: "VariableBase[LieGroupType]",
        delta: types.VariableValue,
        scale_tril_inv: types.ScaleTrilInv,
    ):
        assert type(before) == type(after)
        return BetweenFactor(
            variables=_BeforeAfterTuple(before=before, after=after),
            delta=delta,
            scale_tril_inv=scale_tril_inv,
            variable_type=type(before),
        )

    @jax.jit
    @overrides
    def compute_error(
        self, before_value: jaxlie.MatrixLieGroup, after_value: jaxlie.MatrixLieGroup
    ):
        # before is T_wb
        # delta is T_ba
        # after is T_wa
        #
        # Our error is the tangent-space difference between our actual computed and T_wa
        # transforms
        return ((before_value @ self.delta).inverse() @ after_value).log()
