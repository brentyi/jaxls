import dataclasses
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Dict,
    Hashable,
    Iterable,
    List,
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
    local: bool
    """True if values are in local parameterization. False if values are global."""

    storage: jnp.ndarray
    """Values of variables stacked."""

    storage_pos_from_variable: Dict["VariableBase", int]
    """Start index of each stored variable."""

    storage_pos_from_variable_type: Dict[Type["VariableBase"], int]
    """Variable of the same type are stored together. Index to the first of a type."""

    count_from_variable_type: Dict[Type["VariableBase"], int]
    """Number of variables of each type."""

    @property
    def variables(self):
        """Helper for iterating over variables."""
        return self.storage_pos_from_variable.keys()

    def get_value(self, variable: "VariableBase") -> jnp.ndarray:
        """Get value corresponding to specific variable.  """
        index = self.storage_pos_from_variable[variable]
        return self.storage[
            index : index
            + (
                variable.get_local_parameter_dim()
                if self.local
                else variable.get_parameter_dim()
            )
        ]

    def __repr__(self):
        value_from_variable = {
            variable: self.get_value(variable) for variable in self.variables
        }
        k: "VariableBase"
        return str(
            {
                f"{i}.{k.__class__.__name__}": list(v)
                for i, (k, v) in enumerate(value_from_variable.items())
            }
        )

    @staticmethod
    def from_dict(
        assignments: Dict["VariableBase", jnp.ndarray]
    ) -> "VariableAssignments":
        # Sort variables by type
        variable_from_variable_type: DefaultDict[
            Type["VariableBase"], List["VariableBase"]
        ] = DefaultDict(list)

        for variable in assignments.keys():
            variable_from_variable_type[type(variable)].append(variable)

        # Create
        storage_list = []
        storage_pos_from_variable: Dict["VariableBase", int] = {}
        storage_pos_from_variable_type: Dict[Type["VariableBase"], int] = {}
        storage_index = 0
        for variable_type, variables in variable_from_variable_type.items():
            storage_pos_from_variable_type[variable_type] = storage_index
            for variable in variables:
                value = assignments[variable]
                assert value.shape == (variable.get_parameter_dim(),)
                storage_list.append(value)
                storage_pos_from_variable[variable] = storage_index
                storage_index += variable.get_parameter_dim()

        values: onp.ndarray = onp.concatenate(storage_list)

        return VariableAssignments(
            local=False,
            storage=values,
            storage_pos_from_variable=storage_pos_from_variable,
            storage_pos_from_variable_type=storage_pos_from_variable_type,
            count_from_variable_type={
                k: len(v) for k, v in variable_from_variable_type.items()
            },
        )

    def create_local_deltas(self) -> "VariableAssignments":
        # Sort variables by type
        variable_from_variable_type: DefaultDict[
            Type["VariableBase"], List["VariableBase"]
        ] = DefaultDict(list)

        for variable in self.variables:
            variable_from_variable_type[type(variable)].append(variable)

        #
        # Create
        storage_pos_from_variable: Dict["VariableBase", int] = {}
        storage_pos_from_variable_type: Dict[Type["VariableBase"], int] = {}
        storage_index = 0
        for variable_type, variables in variable_from_variable_type.items():
            storage_pos_from_variable_type[variable_type] = storage_index
            for variable in variables:
                storage_pos_from_variable[variable] = storage_index
                storage_index += variable.get_local_parameter_dim()

        return VariableAssignments(
            local=True,
            storage=onp.zeros(storage_index),
            storage_pos_from_variable=storage_pos_from_variable,
            storage_pos_from_variable_type=storage_pos_from_variable_type,
            count_from_variable_type={
                k: len(v) for k, v in variable_from_variable_type.items()
            },
        )

    @staticmethod
    def create_default(variables: Iterable["VariableBase"]) -> "VariableAssignments":
        """Create an assignments object with all parameters set to their defaults."""
        variable: "VariableBase"
        return VariableAssignments.from_dict(
            {variable: variable.get_default_value() for variable in variables}
        )

    @staticmethod
    def flatten(v: "VariableAssignments") -> Tuple[Tuple, Any]:
        return (v.storage,), (
            v.local,
            v.storage_pos_from_variable,
            v.storage_pos_from_variable_type,
            v.count_from_variable_type,
        )

    @staticmethod
    def unflatten(aux_data: Any, children: Tuple) -> "VariableAssignments":
        assert len(children) == 1 and isinstance(
            children[0], jnp.ndarray
        ), "VariableAssignments pytree node should not be nested"
        return VariableAssignments(
            local=aux_data[0],
            storage=children[0],
            storage_pos_from_variable=aux_data[1],
            storage_pos_from_variable_type=aux_data[2],
            count_from_variable_type=aux_data[3],
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
