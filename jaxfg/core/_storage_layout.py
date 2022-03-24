import dataclasses
from typing import Collection, DefaultDict, Dict, Iterable, List, Type

from ._variables import VariableBase


@dataclasses.dataclass
class StorageLayout:
    """Contains information about how the values of variables are stored in a flattened
    storage vector.

    Note that this is a vanilla dataclass -- not a PyTree. (in other words: all contents
    are static)
    """

    local_flag: bool
    """Set to `True` for local parameterization storage."""

    dim: int
    """Total dimension of storage vector."""

    index_from_variable: Dict[VariableBase, int]
    """Start index of each stored variable."""

    index_from_variable_type: Dict[Type[VariableBase], int]
    """Variable of the same type are stored together. Index to the first of a type."""

    count_from_variable_type: Dict[Type[VariableBase], int]
    """Number of variables of each type."""

    def get_variables(self) -> Collection[VariableBase]:
        """Variables. Storage indices are guaranteed to be in ascending order."""
        # Dictionaries from Python 3.7 retain insertion order
        return self.index_from_variable.keys()

    def get_variable_types(self) -> Collection[Type[VariableBase]]:
        """Variable types. Storage indices are guaranteed to be in ascending order."""
        # Dictionaries from Python 3.7 retain insertion order
        return self.index_from_variable_type.keys()

    @staticmethod
    def make(variables: Iterable[VariableBase], local: bool = False) -> "StorageLayout":
        """Determine storage indexing from a list of variables."""

        # Sort variables by type name before bucketing
        # As variables_from_type will keep its insertion order when calling
        # .items() this ensure we always have the same alphabetical order
        variables = sorted(variables, key=lambda x: str(type(x)))

        # Bucket variables by type
        variables_from_type: DefaultDict[
            Type[VariableBase], List[VariableBase]
        ] = DefaultDict(list)
        for variable in variables:
            variables_from_type[type(variable)].append(variable)

        # Assign block of storage vector for each variable
        index_from_variable: Dict[VariableBase, int] = {}
        index_from_variable_type: Dict[Type[VariableBase], int] = {}
        storage_index = 0
        for variable_type, variables in variables_from_type.items():
            index_from_variable_type[variable_type] = storage_index
            for variable in variables:
                index_from_variable[variable] = storage_index
                storage_index += (
                    variable.get_local_parameter_dim()
                    if local
                    else variable.get_parameter_dim()
                )

        return StorageLayout(
            local_flag=local,
            dim=storage_index,
            index_from_variable=index_from_variable,
            index_from_variable_type=index_from_variable_type,
            count_from_variable_type={
                k: len(v) for k, v in variables_from_type.items()
            },
        )
