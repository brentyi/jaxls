import abc
import dataclasses
from typing import Generic, Sequence, Tuple, Type, TypeVar, cast, get_type_hints

import jax
import numpy as onp
from jax import numpy as jnp
from overrides import EnforceOverrides, final, overrides
from typing_utils import get_args, issubtype

from .. import hints
from ._variables import VariableBase

FactorType = TypeVar("FactorType", bound="FactorBase")

VariableValueTypes = TypeVar(
    "VariableValueTypes",
    bound=Tuple[hints.VariableValue, ...],
)

# Disable type-checking here
# > https://github.com/python/mypy/issues/5374
@dataclasses.dataclass  # type: ignore
class FactorBase(
    Generic[VariableValueTypes],
    abc.ABC,
    EnforceOverrides,
):
    variables: Tuple[VariableBase, ...]
    """Variables connected to this factor. 1-to-1, in-order correspondence with
    `VariableValueTypes`."""

    scale_tril_inv: hints.ScaleTrilInv
    """Inverse square root of covariance matrix."""

    @abc.abstractmethod
    def compute_residual_vector(
        self, variable_values: VariableValueTypes
    ) -> hints.Array:
        """Compute factor error.

        Args:
            variable_values: Values of self.variables
        """

    @final
    def get_residual_dim(self) -> int:
        """Error dimensionality."""
        # We can't use [0] here, because (for stacked factors) there might be a batch
        # dimension!
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

    def compute_residual_jacobians(
        self, variable_values: VariableValueTypes
    ) -> Tuple[hints.Array, ...]:
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
                self.build_variable_value_tuple(perturbed_values)
            )

        # Evaluate Jacobian when deltas are zero
        return jax.jacfwd(compute_cost_with_local_delta)(
            tuple(
                onp.zeros(variable.get_local_parameter_dim())
                for variable in self.variables
            )
        )

    def build_variable_value_tuple(
        self, variable_values: Tuple[hints.VariableValue, ...]
    ) -> VariableValueTypes:
        """Prepares and validates a raw tuple of variable values to be passed into
        `compute_residual_vector` or `compute_residual_jacobians`.

        Slightly sketchy: checks the type hinting on `compute_residual_vector` and if
        the user expects a named tuple, we wrap the input accordingly. Otherwise, we
        just cast and return the input."""

        assert isinstance(variable_values, tuple)

        output: VariableValueTypes

        try:
            value_type: Type[VariableValueTypes] = get_type_hints(
                self.compute_residual_vector
            )["variable_values"]
        except KeyError as e:
            raise NotImplementedError(
                f"Missing type hints for {type(self).__name__}.compute_residual_vector"
            ) from e

        # Function should be hinted with a tuple of some kind, but not `tuple` itself
        assert issubtype(value_type, tuple), value_type is not tuple

        # Heuristic: evaluates to `True` for NamedTuple types but `False` for
        # `Tuple[...]` types. Note that standard superclass checking approaches don't
        # work for NamedTuple types.
        if type(value_type) is type:
            # Hint is `NamedTuple`
            tuple_content_types = get_type_hints(value_type).values()
            output = value_type(*variable_values)
        else:
            # Hint is `typing.Tuple` annotation
            tuple_content_types = get_args(value_type)
            output = cast(VariableValueTypes, variable_values)

        # Validate type annotations
        assert len(variable_values) == len(tuple_content_types)
        for i, (value, expected_type) in enumerate(
            zip(variable_values, tuple_content_types)
        ):
            assert isinstance(value, expected_type), (
                f"Variable value type hint inconsistency: expected {expected_type} at, "
                f"position {i} but got {type(value)}."
            )

        return output
