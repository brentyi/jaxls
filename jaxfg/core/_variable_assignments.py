import dataclasses
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Tuple,
    Type,
    TypeVar,
)

import jax
import numpy as onp
from jax import numpy as jnp

from .. import types, utils

if TYPE_CHECKING:
    from ._variables import VariableBase


@dataclasses.dataclass(frozen=True)
class StorageMetadata:
    """Contains information about where the values of each variable are in a flattened
    storage vector.
    """

    local_flag: bool
    """Set to `True` for local parameterization storage."""

    dim: int
    """Total dimension of storage vector."""

    index_from_variable: Dict["VariableBase", int]
    """Start index of each stored variable."""

    index_from_variable_type: Dict[Type["VariableBase"], int]
    """Variable of the same type are stored together. Index to the first of a type."""

    count_from_variable_type: Dict[Type["VariableBase"], int]
    """Number of variables of each type."""

    @property
    def ordered_variables(self) -> Tuple["VariableBase"]:
        # Dictionaries from Python 3.7 retain insertion order
        return tuple(self.index_from_variable.keys())

    @staticmethod
    def from_variables(
        variables: Iterable["VariableBase"], local: bool = False
    ) -> "StorageMetadata":
        """Determine storage indexing from variable list."""

        # Bucket variables by type + sort deterministically!
        variables_from_type: DefaultDict[
            Type["VariableBase"], List["VariableBase"]
        ] = DefaultDict(list)
        for variable in sorted(variables):
            variables_from_type[type(variable)].append(variable)

        # Assign block of storage vector for each variable
        index_from_variable: Dict["VariableBase", int] = {}
        index_from_variable_type: Dict[Type["VariableBase"], int] = {}
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

        return StorageMetadata(
            local_flag=local,
            dim=storage_index,
            index_from_variable=index_from_variable,
            index_from_variable_type=index_from_variable_type,
            count_from_variable_type={
                k: len(v) for k, v in variables_from_type.items()
            },
        )


VariableValueType = TypeVar("T", bound=types.VariableValue)


@jax.partial(utils.register_dataclass_pytree, static_fields=("storage_metadata",))
@dataclasses.dataclass(frozen=True)
class VariableAssignments:
    storage: jnp.ndarray
    """Values of variables stacked. Can either be local or global parameterizations,
    depending on the value of `.storage_metadata.local_flag`."""

    storage_metadata: StorageMetadata
    """Metadata for how variables are stored."""

    @property
    def variables(self):
        """Helper for iterating over variables."""
        return self.storage_metadata.ordered_variables

    def get_value(self, variable: "VariableBase[T]") -> VariableValueType:
        """Get value corresponding to specific variable.  """
        index = self.storage_metadata.index_from_variable[variable]
        return type(variable).unflatten(
            self.storage[index : index + variable.get_parameter_dim()]
        )

    def get_stacked_value(
        self, variable_type: Type[VariableValueType]
    ) -> VariableValueType:
        """Get values of all variables corresponding to a specific type."""
        index = self.storage_metadata.index_from_variable_type[variable_type]
        count = self.storage_metadata.count_from_variable_type[variable_type]
        return jax.vmap(variable_type.unflatten)(
            self.storage[
                index : index + variable_type.get_parameter_dim() * count
            ].reshape((count, variable_type.get_parameter_dim()))
        )

    def __repr__(self):
        value_from_variable = {
            variable: self.get_value(variable) for variable in self.variables
        }
        k: "VariableBase"

        contents: str = "\n".join(
            [
                f"\t{i}.{k.__class__.__name__}: {v}"
                for i, (k, v) in enumerate(value_from_variable.items())
            ]
        )
        return f"VariableAssignments(\n{contents}\n)"

    @staticmethod
    def from_dict(
        assignments: Dict["VariableBase", types.VariableValue]
    ) -> "VariableAssignments":
        # Figure out how variables are stored
        storage_metadata = StorageMetadata.from_variables(assignments.keys())

        # Stack variable values in order
        storage = jnp.concatenate(
            [
                variable.flatten(assignments[variable])
                for variable in storage_metadata.ordered_variables
            ],
            axis=0,
        )
        assert storage.shape == (storage_metadata.dim,)

        return VariableAssignments(
            storage=storage,
            storage_metadata=storage_metadata,
        )

    @staticmethod
    def create_default(variables: Iterable["VariableBase"]) -> "VariableAssignments":
        """Create an assignments object with all parameters set to their defaults."""
        variable: "VariableBase"
        return VariableAssignments.from_dict(
            {variable: variable.get_default_value() for variable in variables}
        )

    @jax.jit
    def apply_local_deltas(
        self, local_delta_assignments: "VariableAssignments"
    ) -> "VariableAssignments":
        """Update variables on manifold."""

        # Check that inputs make sense
        assert not self.storage_metadata.local_flag
        assert local_delta_assignments.storage_metadata.local_flag

        # On-manifold retractions, one variable type at a time!
        new_storage = jnp.zeros_like(self.storage)
        variable_type: Type["VariableBase"]
        for variable_type in self.storage_metadata.index_from_variable_type.keys():

            # Get locations
            count = self.storage_metadata.count_from_variable_type[variable_type]
            storage_index = self.storage_metadata.index_from_variable_type[
                variable_type
            ]
            local_storage_index = (
                local_delta_assignments.storage_metadata.index_from_variable_type[
                    variable_type
                ]
            )
            dim = variable_type.get_parameter_dim()
            local_dim = variable_type.get_local_parameter_dim()

            # Get batched variables
            batched_xs = self.storage[
                storage_index : storage_index + dim * count
            ].reshape((count, dim))
            batched_deltas = local_delta_assignments.storage[
                local_storage_index : local_storage_index + local_dim * count
            ].reshape((count, local_dim))

            # Batched variable update
            new_storage = new_storage.at[
                storage_index : storage_index + dim * count
            ].set(
                variable_type.flatten(
                    jax.vmap(variable_type.add_local)(
                        variable_type.unflatten(batched_xs), batched_deltas
                    )
                ).flatten()
            )
        return dataclasses.replace(self, storage=new_storage)
