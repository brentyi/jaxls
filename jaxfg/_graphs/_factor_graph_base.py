import abc
import dataclasses
from typing import Dict, Generic, Set, TypeVar

from .._factors import FactorBase
from .._variables import RealVectorVariable, VariableBase

FactorGraphType = TypeVar("FactorGraphType", bound="FactorGraphBase")
FactorType = TypeVar("FactorType", bound=FactorBase)
VariableType = TypeVar("VariableType", bound=VariableBase)


@dataclasses.dataclass(frozen=True)
class FactorGraphBase(abc.ABC, Generic[FactorType, VariableType]):
    factors: Set[FactorType] = dataclasses.field(
        default_factory=lambda: set(), init=False
    )
    factors_from_variable: Dict[VariableType, Set[FactorType]] = dataclasses.field(
        default_factory=lambda: {}, init=False
    )

    def with_factors(self: FactorGraphType, *to_add: FactorType) -> FactorGraphType:
        """Generate a new graph with additional factors added.

        Existing graph is marked dirty and can no longer be used.
        """

        # Create shallow copy of self
        new_graph = dataclasses.copy.copy(self)

        # Mark self as dirty
        # (note that we can't mutate normally)
        object.__setattr__(self, "factors", None)
        object.__setattr__(self, "factors_from_variables", None)

        for factor in to_add:
            # Add factor to graph
            assert factor not in new_graph.factors
            new_graph.factors.add(factor)

            # Make constant-time variable=>factor lookup possible
            for v in factor.variables:
                if v not in new_graph.factors_from_variable:
                    new_graph.factors_from_variable[v] = set()
                new_graph.factors_from_variable[v].add(factor)

        # Return "new" graph
        return new_graph

    def without_factors(
        self: FactorGraphType, *to_remove: FactorType
    ) -> FactorGraphType:
        """Generate a new graph, with specified factors removed.

        Existing graph is marked dirty and can no longer be used.
        """

        # Copy self
        new_graph = dataclasses.copy.copy(self)

        # Mark self as dirty
        self.__setattr__("factors", None)
        self.__setattr__("factors_from_variables", None)

        for factor in to_remove:
            # Add factor to graph
            assert factor in new_graph.factors
            new_graph.factors.remove(factor)

            # Remove variables from graph
            for v in factor.variables:
                new_graph.factors_from_variable[v].remove(factor)
                if len(new_graph.factors_from_variable[v]) == 0:
                    new_graph.factors_from_variable.pop(v)

        # Return "new" graph
        return new_graph
