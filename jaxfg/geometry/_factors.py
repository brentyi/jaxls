from typing import NamedTuple, Tuple

import jax_dataclasses
import jaxlie
from jax import numpy as jnp
from overrides import overrides

from .. import noises
from ..core._factor_base import FactorBase
from ._lie_variables import LieVariableBase

# To implement a factor, we start by defining what the types of the variables that
# connect to it are.
# This can be either a standard tuple or a named one (see BetweenFactor for the latter),
# and should be used to inherit from `FactorBase[some tuple type]`,  and as input to the
# factor's `compute_residual_vector` and `compute_residual_jacobians` methods.
PriorValueTuple = Tuple[jaxlie.MatrixLieGroup]


@jax_dataclasses.pytree_dataclass
class PriorFactor(FactorBase[PriorValueTuple]):
    """Factor for defining a fixed prior on a frame.

    Residuals are computed as `(variable.inverse() @ mu).log()`.
    """

    mu: jaxlie.MatrixLieGroup

    # Optional: it can be nice to define a factory method. In this case we constrain the
    # types a bit (for example, restrict connections to just a single variable) and call
    # the dataclass constructor.
    @staticmethod
    def make(
        variable: LieVariableBase,
        mu: jaxlie.MatrixLieGroup,
        noise_model: noises.NoiseModelBase,
    ) -> "PriorFactor":
        return PriorFactor(
            variables=(variable,),
            mu=mu,
            noise_model=noise_model,
        )

    # Required: we define the residual corresponding to this factor type.
    @overrides
    def compute_residual_vector(self, variable_values: PriorValueTuple) -> jnp.ndarray:

        T: jaxlie.MatrixLieGroup
        (T,) = variable_values

        # Equivalent to: return (variable_value.inverse() @ self.mu).log()
        return jaxlie.manifold.rminus(T, self.mu)

    # Optional: we specify an analytical Jacobian.
    @overrides
    def compute_residual_jacobians(
        self, variable_values: PriorValueTuple
    ) -> Tuple[jnp.ndarray]:

        T: jaxlie.MatrixLieGroup
        (T,) = variable_values
        return (-jnp.eye(type(T).tangent_dim),)


# Our BetweenFactor implementation is pretty similar to PriorFactor, but requires two
# variables instead of just one. Named tuples are supported automatically, and can help
# keep our code a bit tidier.
class BetweenValueTuple(NamedTuple):
    T_world_a: jaxlie.MatrixLieGroup
    T_world_b: jaxlie.MatrixLieGroup


@jax_dataclasses.pytree_dataclass
class BetweenFactor(FactorBase[BetweenValueTuple]):
    """Factor for defining a geometric relationship between frames `a` and `b`.

    Residuals are computed as `((T_world_a.inverse() @ T_world_b).inverse() @ T_a_b).log()`.
    """

    T_a_b: jaxlie.MatrixLieGroup

    @staticmethod
    def make(
        variable_T_world_a: LieVariableBase,
        variable_T_world_b: LieVariableBase,
        T_a_b: jaxlie.MatrixLieGroup,
        noise_model: noises.NoiseModelBase,
    ) -> "BetweenFactor":
        assert type(variable_T_world_a) is type(variable_T_world_b)
        assert variable_T_world_a.get_group_type() is type(T_a_b)

        return BetweenFactor(
            variables=(
                variable_T_world_a,
                variable_T_world_b,
            ),
            T_a_b=T_a_b,
            noise_model=noise_model,
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: BetweenValueTuple
    ) -> jnp.ndarray:
        T_world_a = variable_values.T_world_a
        T_world_b = variable_values.T_world_b

        return jaxlie.manifold.rminus(T_world_a.inverse() @ T_world_b, self.T_a_b)

    @overrides
    def compute_residual_jacobians(
        self, variable_values: BetweenValueTuple
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Implementing this is optional!
        # Autodiff will handle Jacobians if we don't specify analytical ones.

        T_world_a = variable_values.T_world_a
        T_world_b = variable_values.T_world_b

        assert type(T_world_a) == type(T_world_b)
        group_cls = type(T_world_a)

        return (
            (T_world_a.inverse() @ T_world_b).inverse().adjoint(),
            -jnp.eye(group_cls.tangent_dim),
        )
