import abc
import dataclasses
from typing import Generic, Sequence, Tuple, Type, TypeVar, cast

import jax
import numpy as onp
from jax import numpy as jnp
from overrides import EnforceOverrides, final, overrides

from .. import hints
from ._variables import VariableBase

FactorType = TypeVar("FactorType", bound="FactorBase")

# If Python had better support for variadic generics, higher-kinded types we could write
# `FactorBase[VariableType1, VariableType2, ...]`, but in the absence of this we will just use
# `FactorBase[Tuple[VariableType1, ...], Tuple[ValueType1, ...]]`.
#
# Note that these TypeVars can generally be inferred implicitly from hinted methods.
FactorVariableTypes = TypeVar(
    "FactorVariableTypes",
    bound=Tuple[VariableBase, ...],
)
FactorVariableValues = TypeVar(
    "FactorVariableValues", bound=Tuple[hints.VariableValue, ...]
)

# Disable type-checking here
# > https://github.com/python/mypy/issues/5374
@dataclasses.dataclass  # type: ignore
class FactorBase(
    Generic[FactorVariableTypes, FactorVariableValues],
    abc.ABC,
    EnforceOverrides,
):
    variables: FactorVariableTypes
    """Variables connected to this factor."""

    scale_tril_inv: hints.ScaleTrilInv
    """Inverse square root of covariance matrix."""

    @final
    def get_residual_dim(self) -> int:
        """Error dimensionality."""
        # We can't use [0] here, because (for stacked factors) there might be a batch dimension!
        return self.scale_tril_inv.shape[-1]

    def __init_subclass__(cls, *args, **kwargs):
        """Register all factors as hashable PyTree nodes."""
        super().__init_subclass__(*args, **kwargs)

        jax.tree_util.register_pytree_node(
            cls, flatten_func=cls._flatten, unflatten_func=cls._unflatten
        )
        cls.__hash__ = object.__hash__

    @classmethod
    @final
    def _flatten(
        cls: Type[FactorType], v: FactorType
    ) -> Tuple[Tuple[hints.PyTree, ...], Tuple]:
        """Flatten a factor for use as a PyTree/parameter stacking."""
        v_dict = vars(v)
        array_data = {k: v for k, v in v_dict.items()}

        # Store variable types to make sure treedef hashes match
        aux_dict = {}
        aux_dict["variabletypes"] = tuple(type(variable) for variable in v.variables)
        array_data.pop("variables")

        return (
            tuple(array_data.values()),
            tuple(array_data.keys())
            + tuple(aux_dict.keys())
            + tuple(aux_dict.values()),
        )

    @classmethod
    @final
    def _unflatten(
        cls: Type[FactorType], treedef: Tuple, children: Tuple[hints.Array]
    ) -> FactorType:
        """Unflatten a factor for use as a PyTree/parameter stacking."""
        array_keys = treedef[: len(children)]
        aux = treedef[len(children) :]
        aux_keys = aux[: len(aux) // 2]
        aux_values = aux[len(aux) // 2 :]

        # Create new dummy variables
        aux_dict = dict(zip(aux_keys, aux_values))
        aux_dict["variables"] = tuple(V() for V in aux_dict.pop("variabletypes"))

        out = cls(**dict(zip(array_keys, children)), **aux_dict)  # type: ignore
        return out

    @abc.abstractmethod
    def compute_residual_vector(
        self, variable_values: FactorVariableValues
    ) -> hints.Array:
        """Compute factor error.

        Args:
            variable_values: Values of self.variables
        """

    def compute_residual_jacobians(
        self, variable_values: FactorVariableValues
    ) -> Sequence[hints.Array]:
        """Compute Jacobian of residual with respect to local parameterization.

        Uses `jax.jacfwd` by default, but can optionally be overriden.

        Args:
            variable_values: Values of variables to linearize around.
        """

        def compute_cost_with_local_delta(
            local_deltas: Sequence[hints.Array],
        ) -> hints.Array:
            # Suppressing:
            # - Need type annotation for 'variable_value'
            # - Argument 1 to "zip" has incompatible type "FactorVariableTypes"; expected "Iterable[<nothing>]
            perturbed_values = tuple(  # type: ignore
                variable.manifold_retract(
                    x=variable_value,
                    local_delta=local_delta,
                )
                for variable, variable_value, local_delta in zip(
                    self.variables, variable_values, local_deltas  # type: ignore
                )
            )

            return self.compute_residual_vector(
                cast(FactorVariableValues, perturbed_values)
            )

        # Evaluate Jacobian when deltas are zero
        return jax.jacfwd(compute_cost_with_local_delta)(
            tuple(
                onp.zeros(variable.get_local_parameter_dim())
                for variable in self.variables
            )
        )


@dataclasses.dataclass
class LinearFactor(FactorBase):
    r"""Linearized factor, corresponding to the simple residual:
    $$
    r = ( \Sum_i A_i x_i ) - b_i
    $$
    """

    A_matrices: Tuple[hints.Array]
    b: hints.Array
    scale_tril_inv: hints.Array

    @final
    @overrides
    def compute_residual_vector(
        self, variable_values: Tuple[hints.VariableValue]
    ) -> hints.Array:
        linear_component = jnp.zeros_like(self.b)
        for A_matrix, value in zip(self.A_matrices, variable_values):
            linear_component = linear_component + A_matrix @ value
        return linear_component - self.b
