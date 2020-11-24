import abc
import dataclasses
import types
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    NamedTuple,
    Tuple,
    Type,
    TypeVar,
)

import jax
import numpy as onp
from jax import numpy as jnp
from overrides import overrides

from . import _types as _types

if TYPE_CHECKING:
    from . import RealVectorVariable, VariableBase


FactorType = TypeVar("FactorType", bound="FactorBase")


@dataclasses.dataclass(frozen=True)
class FactorBase(abc.ABC):
    variables: Tuple["VariableBase"]
    """Variables connected to this factor. Immutable. (currently assumed but unenforced)"""

    scale_tril_inv: _types.ScaleTrilInv
    """Inverse square root of covariance matrix."""

    # Use default object hash rather than dataclass one
    __hash__ = object.__hash__

    @property
    def error_dim(self) -> int:
        """Error dimensionality."""
        return self.scale_tril_inv.shape[0]

    def __init_subclass__(cls, **kwargs):
        """Register all factors as PyTree nodes."""
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node(
            cls, flatten_func=cls.flatten, unflatten_func=cls.unflatten
        )

    @classmethod
    def flatten(
        cls: Type[FactorType], v: FactorType
    ) -> Tuple[Tuple[jnp.ndarray], Tuple[str]]:
        """Flatten a factor for use as a PyTree/parameter stacking."""
        v_dict = dataclasses.asdict(v)
        v_dict.pop("variables")
        return tuple(v_dict.values()), tuple(v_dict.keys())

    @classmethod
    def unflatten(
        cls: Type[FactorType], aux_data: Tuple[str], children: Tuple[jnp.ndarray]
    ) -> FactorType:
        """Unflatten a factor for use as a PyTree/parameter stacking."""
        return cls(variables=tuple(), **dict(zip(aux_data, children)))

    def group_key(self) -> _types.GroupKey:
        """Get unique key for grouping factors.

        Args:

        Returns:
            _types.GroupKey:
        """
        v: "VariableBase"
        return _types.GroupKey(
            factor_type=self.__class__,
            secondary_key=(
                tuple((type(v), v.parameter_dim) for v in self.variables),
                self.error_dim,
            ),
        )

    @abc.abstractmethod
    def compute_error(self, *args: jnp.ndarray):
        """compute_error.

        Args:
            *args (jnp.ndarray): Arguments
        """


@dataclasses.dataclass(frozen=True)
class LinearFactor(FactorBase):
    """Linearized factor, corresponding to the simple residual:
    $$
    r = ( \Sum_i A_i x_i ) - b_i
    $$
    """

    A_matrices: Tuple[jnp.ndarray]
    b: jnp.ndarray
    scale_tril_inv: jnp.ndarray

    # Use default object hash rather than dataclass one
    __hash__ = object.__hash__

    @overrides
    def compute_error(self, *variable_values: jnp.ndarray):
        linear_component = jnp.zeros_like(self.b)
        for A_matrix, value in zip(self.A_matrices, variable_values):
            linear_component = linear_component + A_matrix @ value
        return linear_component - self.b

    @classmethod
    def linearize_from_factor(
        cls, factor: FactorBase, assignments: _types.VariableAssignments
    ) -> "LinearFactor":
        """Produce a LinearFactor object by linearizing an arbitrary factor.

        Args:
            factor (FactorBase): Factor to linearize.
            assignments (_types.VariableAssignments): Assignments to linearize around.

        Returns:
            LinearFactor: Linearized factor.
        """
        A_from_variable: Dict[
            "RealVectorVariable", Callable[[jnp.ndarray], jnp.ndarray]
        ] = {}

        # Pull out only the assignments that we care about
        assignments = {variable: assignments[variable] for variable in factor.variables}

        for variable in factor.variables:

            def f(local_delta: jnp.ndarray) -> jnp.ndarray:
                # Make copy of assignments with updated variable value
                assignments_copy = assignments.copy()
                assignments_copy[variable] = variable.add_local(
                    x=assignments_copy[variable], local_delta=local_delta
                )

                # Return whitened error
                return factor.scale_tril_inv @ factor.compute_error(*())

            # Linearize around variable
            f_jvp = jax.jacfwd(f)(jnp.zeros(variable.local_parameter_dim))[1]
            A_from_variable[variable.local_delta_variable] = f_jvp

        error = factor.compute_error(assignments=assignments)
        return LinearFactor(
            A_from_variable=A_from_variable, b=-factor.scale_tril_inv @ error
        )

    def group_key(self) -> _types.GroupKey:
        """Get unique key for grouping factors.

        Args:

        Returns:
            _types.GroupKey:
        """
        return _types.GroupKey(
            factor_type=self.__class__,
            secondary_key=tuple(A.shape for A in self.A_matrices),
        )


class PriorFactor(FactorBase):
    def __init__(
        self,
        variable: "VariableBase",
        mu: jnp.ndarray,
        scale_tril_inv: _types.ScaleTrilInv,
    ):
        super().__init__(variables=(variable,), scale_tril_inv=scale_tril_inv)
        self.mu = mu

    @overrides
    def compute_error(self, assignments: _types.VariableAssignments):
        variable = self.variables[0]
        return variable.subtract_local(assignments[variable], self.mu)


class BetweenFactor(FactorBase):
    class _BeforeAfterTuple(NamedTuple):
        before: "VariableBase"
        after: "VariableBase"

    def __init__(
        self,
        before: "VariableBase",
        after: "VariableBase",
        delta: jnp.ndarray,
        scale_tril_inv: _types.ScaleTrilInv,
    ):
        self.variables: BetweenFactor._BeforeAfterTuple
        self.delta = delta
        super().__init__(
            variables=self._BeforeAfterTuple(before=before, after=after),
            scale_tril_inv=scale_tril_inv,
        )

    @overrides
    def compute_error(self, assignments: _types.VariableAssignments):
        before: "VariableBase" = self.variables.before
        after: "VariableBase" = self.variables.after
        return before.subtract_local(
            before.add_local(x=assignments[before], local_delta=self.delta),
            assignments[after],
        )
