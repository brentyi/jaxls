import abc
from typing import Generic, Tuple, Type, TypeVar, cast, get_args, get_type_hints

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from overrides import EnforceOverrides, final

from .. import hints, noises
from ._variable_assignments import VariableAssignments
from ._variables import VariableBase

FactorType = TypeVar("FactorType", bound="FactorBase")

VariableValueTuple = TypeVar(
    "VariableValueTuple",
    bound=Tuple[hints.VariableValue, ...],
)

T = TypeVar("T")


@jdc.pytree_dataclass
class _FactorBase:
    # For why we have two classes:
    # https://github.com/python/mypy/issues/5374#issuecomment-650656381

    variables: jdc.Static[Tuple[VariableBase, ...]]
    """Variables connected to this factor. 1-to-1, in-order correspondence with
    `VariableValueTuple`."""

    noise_model: noises.NoiseModelBase
    """Noise model."""


class FactorBase(_FactorBase, Generic[VariableValueTuple], abc.ABC, EnforceOverrides):
    # (1) Functions that must be overriden in subclasses.

    @abc.abstractmethod
    def compute_residual_vector(
        self, variable_values: VariableValueTuple
    ) -> jnp.ndarray:
        """Compute factor error.

        Args:
            variable_values: Values of self.variables
        """

    # (2) Function to override for analytical Jacobians. This is always optional.

    def compute_residual_jacobians(
        self, variable_values: VariableValueTuple
    ) -> Tuple[jnp.ndarray, ...]:
        """Compute Jacobian of residual with respect to local parameterization, by
        composing the residual computation Jacobian with the manifold retraction
        Jacobian.

        There are two options for specifying analytical Jacobians:
        1) Override this method directly,
        2) Override `manifold_retract_jacobian()` for variables and define a custom JVP
          method for `compute_residual_vector`.
        """

        # Some helpers for reshaping + concatenating Jacobians
        def reshape_leaves(tree: hints.Pytree, shape: Tuple[int, ...]) -> hints.Pytree:
            return jax.tree_map(lambda leaf: leaf.reshape(shape), tree)

        # To compute the Jacobian of the residual wrt the local parameters, we compose
        # (1) the residual wrt the variable parameters and (2) the variable parameters
        # wrt the local parameterization.
        assert len(self.variables) == len(variable_values)
        jacobians = jax.tree_map(
            jnp.dot,
            tuple(
                jnp.concatenate(jax.tree_leaves(tree), axis=-1)
                for tree in reshape_leaves(
                    jax.jacfwd(self.compute_residual_vector)(variable_values),
                    (self.get_residual_dim(), -1),
                )
            ),
            tuple(
                jnp.concatenate(
                    reshape_leaves(
                        self.variables[i].manifold_retract_jacobian(variable_values[i]),
                        (-1, v.get_local_parameter_dim()),
                    ),
                    axis=0,
                )
                for i, v in enumerate(self.variables)
            ),
        )

        return jacobians

    # (3) Shared implementations.

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
        # assert issubclass(value_type, tuple), value_type is not tuple

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

    @final
    def anonymize_variables(self: FactorType) -> FactorType:
        """Returns a copy of this factor with all variables replaced with their
        canonical instances. Used for factor stacking.
        """
        v: VariableBase
        return jdc.replace(
            self, variables=tuple(type(v).canonical_instance() for v in self.variables)
        )
