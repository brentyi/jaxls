import dataclasses
from typing import TYPE_CHECKING, Hashable, Type

if TYPE_CHECKING:
    from ..core._factors import FactorBase


@dataclasses.dataclass(frozen=True)
class GroupKey:
    """Key for grouping factors that can be computed in parallel."""

    factor_type: Type["FactorBase"]
    secondary_key: Hashable
