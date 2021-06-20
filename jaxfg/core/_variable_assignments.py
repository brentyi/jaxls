from typing import Dict, Iterable, Type, TypeVar

import jax
import jax_dataclasses
from jax import numpy as jnp

from .. import hints
from ._storage_metadata import StorageMetadata
from ._variables import VariableBase

VariableValueType = TypeVar("VariableValueType", bound=hints.VariableValue)


@jax_dataclasses.pytree_dataclass
class VariableAssignments:
    """Storage class that maps variables to values."""

    storage: hints.Array
    """Values of variables stacked and flattened. Can either be local or global
    parameterizations, depending on the value of `.storage_metadata.local_flag`."""

    storage_metadata: StorageMetadata = jax_dataclasses.static_field()
    """Metadata for how variables are stored."""

    @staticmethod
    def make_from_defaults(variables: Iterable[VariableBase]) -> "VariableAssignments":
        """Create an assignment object from the default values corresponding to each variable."""

        # Figure out how variables are stored
        storage_metadata = StorageMetadata.make(variables, local=False)

        # Stack variable values in order
        storage = jnp.concatenate(
            [
                jnp.tile(
                    jax.jit(variable_type.flatten)(variable_type.get_default_value()),
                    reps=(storage_metadata.count_from_variable_type[variable_type],),
                )
                for variable_type in storage_metadata.get_variable_types()
            ],
            axis=0,
        ).block_until_ready()
        assert storage.shape == (storage_metadata.dim,)

        return VariableAssignments(storage=storage, storage_metadata=storage_metadata)

    @staticmethod
    def make_from_dict(
        assignments: Dict[VariableBase, hints.VariableValue],
    ) -> "VariableAssignments":
        """Create an assignment object from a full set of assignments."""

        return VariableAssignments.make_from_partial_dict(
            assignments.keys(), assignments
        )

    @staticmethod
    def make_from_partial_dict(
        variables: Iterable[VariableBase],
        assignments: Dict[VariableBase, hints.VariableValue],
    ) -> "VariableAssignments":
        """Create an assignment object from a variables and assignments. Missing
        assignments are assigned the default variable values."""

        # Figure out how variables are stored
        storage_metadata = StorageMetadata.make(variables, local=False)

        # Stack variable values in order
        storage = jnp.concatenate(
            [
                jax.jit(variable.flatten)(
                    assignments[variable]
                    if assignments is not None and variable in assignments
                    else variable.get_default_value()
                )
                for variable in storage_metadata.get_variables()
            ],
            axis=0,
        ).block_until_ready()
        assert storage.shape == (storage_metadata.dim,)

        return VariableAssignments(storage=storage, storage_metadata=storage_metadata)

    def as_dict(self) -> Dict[VariableBase, hints.VariableValue]:
        """Grab assignments as a variable -> value dictionary."""
        return {v: self.get_value(v) for v in self.get_variables()}

    def __repr__(self):
        value_from_variable = {
            variable: self.get_value(variable) for variable in self.get_variables()
        }
        k: VariableBase

        contents: str = "\n".join(
            [
                f"    {i}.{k.__class__.__name__}: {v}"
                for i, (k, v) in enumerate(value_from_variable.items())
            ]
        )
        return f"VariableAssignments(\n{contents}\n)"

    def get_variables(self) -> Iterable[VariableBase]:
        """Helper for iterating over variables."""
        return self.storage_metadata.get_variables()

    def get_value(self, variable: VariableBase[VariableValueType]) -> VariableValueType:
        """Get value corresponding to specific variable."""
        index = self.storage_metadata.index_from_variable[variable]
        return type(variable).unflatten(
            self.storage[index : index + variable.get_parameter_dim()]
        )

    def get_stacked_value(
        self, variable_type: Type[VariableBase[VariableValueType]]
    ) -> VariableValueType:
        """Get values of all variables corresponding to a specific type."""
        index = self.storage_metadata.index_from_variable_type[variable_type]
        count = self.storage_metadata.count_from_variable_type[variable_type]
        return jax.vmap(variable_type.unflatten)(
            self.storage[
                index : index + variable_type.get_parameter_dim() * count
            ].reshape((count, variable_type.get_parameter_dim()))
        )

    def set_value(
        self, variable: VariableBase[VariableValueType], value: VariableValueType
    ) -> "VariableAssignments":
        """Update a value corresponding to specific variable."""

        index = self.storage_metadata.index_from_variable[variable]
        with jax_dataclasses.copy_and_mutate(self) as output:
            output.storage = (
                jnp.asarray(output.storage)  # In case storage vector is an onp array
                .at[index : index + type(variable).get_parameter_dim()]
                .set(type(variable).flatten(value))
            )
        return output

    @jax.jit
    def manifold_retract(
        self, local_delta_assignments: "VariableAssignments"
    ) -> "VariableAssignments":
        """Update variables on manifold."""

        # Check that inputs make sense
        assert not self.storage_metadata.local_flag
        assert local_delta_assignments.storage_metadata.local_flag

        # On-manifold retractions, one variable type at a time!
        new_storage = jnp.zeros_like(self.storage)
        variable_type: Type[VariableBase]
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
            batched_values_flat = self.storage[
                storage_index : storage_index + dim * count
            ].reshape((count, dim))
            batched_deltas = local_delta_assignments.storage[
                local_storage_index : local_storage_index + local_dim * count
            ].reshape((count, local_dim))

            # Batched variable update
            new_storage = new_storage.at[
                storage_index : storage_index + dim * count
            ].set(
                jax.vmap(variable_type.flatten)(
                    jax.vmap(variable_type.manifold_retract)(
                        jax.vmap(variable_type.unflatten)(batched_values_flat),
                        batched_deltas,
                    )
                ).flatten()
            )

        return jax_dataclasses.replace(self, storage=new_storage)
