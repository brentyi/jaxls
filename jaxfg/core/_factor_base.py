import abc
import dataclasses
from typing import Generic, Sequence, Tuple, Type, TypeVar, cast, get_type_hints

import jax
import numpy as onp
from jax import numpy as jnp
from overrides import EnforceOverrides, final
from typing_utils import get_args, issubtype

from .. import hints, noises, utils
from ._variable_assignments import VariableAssignments
from ._variables import VariableBase

FactorType = TypeVar("FactorType", bound="FactorBase")

VariableValueTuple = TypeVar(
    "VariableValueTuple",
    bound=Tuple[hints.VariableValue, ...],
)

T = TypeVar("T")


def _make_unhashable(x: T) -> T:
    """Helper for making an object unhashable."""

    def new_hash() -> int:
        assert False

    x.__hash__ = new_hash  # type: ignore
    return x


# Disable type-checking because of abstract class issues
# > https://github.com/python/mypy/issues/5374
@utils.register_dataclass_pytree
@dataclasses.dataclass  # type: ignore
class FactorBase(
    Generic[VariableValueTuple],
    abc.ABC,
    EnforceOverrides,
):
    variables: Tuple[VariableBase, ...] = dataclasses.field(
        metadata=utils.static_field(
            # To enable batch computations, we want to be able to stack many factors of
            # the same type. In order to do this, their treedefs must match: this is by
            # default not possible, as every factor will of course be attached to
            # different variables.
            #
            # To get around this, we erase the individual identities of the variables
            # before flattening each factor. Factors that have been flattened and then
            # restored at a JAX API boundary will therefore become unhashable. For
            # stacked factors, however, the specific variables connected to each factor
            # are retained implicitly in `FactorStack.value_indices`.
            treedef_from_value=lambda values: tuple(
                type(v) for v in cast(tuple, values)
            ),
            value_from_treedef=lambda treedef: tuple(
                _make_unhashable(t.__new__(t)) for t in treedef
            ),
        )
    )
    """Variables connected to this factor. 1-to-1, in-order correspondence with
    `VariableValueTuple`."""

    noise_model: noises.NoiseModelBase
    """Noise model."""

    @abc.abstractmethod
    def compute_residual_vector(
        self, variable_values: VariableValueTuple
    ) -> jnp.ndarray:
        """Compute factor error.

        Args:
            variable_values: Values of self.variables
        """

    def compute_residual_jacobians(
        self, variable_values: VariableValueTuple
    ) -> Tuple[jnp.ndarray, ...]:
        """Compute Jacobian of residual with respect to local parameterization.

        Uses `jax.jacfwd` by default, but can optionally be overriden.

        Args:
            variable_values: Values of variables to linearize around.
        """

        def compute_cost_with_local_delta(
            local_deltas: Sequence[jnp.ndarray],
        ) -> jnp.ndarray:
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

    def compute_whitened_residual_vector(self, variable_values: VariableValueTuple):
        pass

    def compute_whitened_residual_jacobians(
        self,
        variable_values: VariableValueTuple,
        residual_vector: hints.Array,
    ):
        pass

    def __init_subclass__(cls, *args, **kwargs):
        """Register all factors as hashable PyTree nodes."""
        super().__init_subclass__(*args, **kwargs)
        cls.__hash__ = object.__hash__

    @final
    def get_residual_dim(self) -> int:
        """Error dimensionality."""
        return self.noise_model.get_residual_dim()

    @final
    def get_variable_values_from_assignments(
        self, assignments: VariableAssignments
    ) -> VariableValueTuple:
        """Prepare a set of variable values corresponding to this factor, for use in
        `compute_residual_vector` or `compute_residual_jacobians`."""

        return self.build_variable_value_tuple(
            tuple(assignments.get_value(v) for v in self.variables)
        )

    @final
    def build_variable_value_tuple(
        self, variable_values: Tuple[hints.VariableValue, ...]
    ) -> VariableValueTuple:
        """Prepares and validates a raw tuple of variable values to be passed into
        `compute_residual_vector` or `compute_residual_jacobians`.

        Slightly sketchy: checks the type hinting on `compute_residual_vector` and if
        the user expects a named tuple, we wrap the input accordingly. Otherwise, we
        just cast and return the input."""

        assert isinstance(variable_values, tuple)

        output: VariableValueTuple

        try:
            value_type: Type[VariableValueTuple] = get_type_hints(
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
            tuple_content_types = tuple(get_type_hints(value_type).values())
            output = value_type(*variable_values)
        else:
            # Hint is `typing.Tuple` annotation
            tuple_content_types = get_args(value_type)
            output = cast(VariableValueTuple, variable_values)

        # Handle Ellipsis in type hints, eg `Tuple[SomeType, ...]`
        if len(tuple_content_types) == 2 and tuple_content_types[1] is Ellipsis:
            tuple_content_types = tuple_content_types[0:1] * len(variable_values)

        # Validate expected and received types
        assert len(variable_values) == len(tuple_content_types)
        for i, (value, expected_type) in enumerate(
            zip(variable_values, tuple_content_types)
        ):
            assert isinstance(value, expected_type), (
                f"Variable value type hint inconsistency: expected {expected_type} at, "
                f"position {i} but got {type(value)}."
            )

        return output
