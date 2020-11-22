import dataclasses
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterable,
    Mapping,
    Tuple,
    Type,
    Union,
)

import jax
import numpy as onp
from jax import numpy as jnp

if TYPE_CHECKING:
    from ._factors import FactorBase
    from ._variables import VariableBase

ScaleTril = Any  # jnp.ndarray
ScaleTrilInv = Any  # jnp.ndarray


@dataclasses.dataclass(frozen=True)
class GroupKey:
    factor_type: Type["FactorBase"]
    secondary_key: Hashable


@dataclasses.dataclass(frozen=True)
class VariableAssignments:
    storage: jnp.ndarray
    """Values of variables stacked."""

    storage_pos_from_variable: Dict["VariableBase", int]
    """Start index of each stored variable."""

    @property
    def variables(self):
        """Helper for iterating over variables."""
        return self.storage_pos_from_variable.keys()

    def get_value(self, variable: "VariableBase") -> jnp.ndarray:
        """Get value corresponding to specific variable.  """
        index = self.storage_pos_from_variable[variable]
        print(index, index + variable.parameter_dim)
        return self.storage[index : index + variable.parameter_dim]

    def to_dict(self) -> Dict["VariableBase", jnp.ndarray]:
        """Return a variable->value dictionary."""
        return {variable: self.get_value(variable) for variable in self.variables}

    def __repr__(self):
        k: "VariableBase"
        return str(
            {
                f"{i}.{k.__class__.__name__}": list(v)
                for i, (k, v) in enumerate(self.to_dict().items())
            }
        )

    @staticmethod
    def from_dict(
        assignments: Dict["VariableBase", jnp.ndarray]
    ) -> "VariableAssignments":
        storage_list = []
        storage_pos_from_variable: Dict[VariableBase, int] = {}
        storage_index = 0
        for variable in assignments.keys():
            value = assignments[variable]
            assert value.shape == (variable.parameter_dim,)
            storage_list.append(value)
            storage_pos_from_variable[variable] = storage_index

            storage_index += variable.parameter_dim

        values: onp.ndarray = onp.concatenate(storage_list)

        return VariableAssignments(
            storage=values,
            storage_pos_from_variable=storage_pos_from_variable,
        )

    @staticmethod
    def create_default(variables: Iterable["VariableBase"]) -> "VariableAssignments":
        """Create an assignments object with all parameters set to their defaults."""
        storage_pos_from_variable = {}
        storage_pos = 0
        storage_list = []

        for variable in variables:
            # Add to storage list
            storage_list.append(variable.get_default_value())
            assert storage_list[-1].shape == (variable.parameter_dim,)

            # Increment index counter
            storage_pos_from_variable[variable] = storage_pos
            storage_pos += variable.parameter_dim

        return VariableAssignments(
            storage=onp.concatenate(storage_list),
            storage_pos_from_variable=storage_pos_from_variable,
        )

    @staticmethod
    def flatten(v: "VariableAssignments") -> Tuple[Tuple, Any]:
        return (v.storage,), v.storage_pos_from_variable

    @staticmethod
    def unflatten(aux_data: Any, children: Tuple) -> "VariableAssignments":
        assert len(children) == 1 and isinstance(
            children[0], jnp.ndarray
        ), "VariableAssignments pytree node should not be nested"
        return VariableAssignments(
            storage=children[0], storage_pos_from_variable=aux_data
        )


# Register assignments as Pytree node so we can use it as inputs/outputs to compiled
# functions
jax.tree_util.register_pytree_node(
    VariableAssignments,
    flatten_func=VariableAssignments.flatten,
    unflatten_func=VariableAssignments.unflatten,
)

# VariableAssignments = Dict["VariableBase", Union[onp.ndarray, jnp.ndarray]]

__all__ = [
    "ScaleTril",
    "ScaleTrilInv",
    "VariableAssignments",
]
