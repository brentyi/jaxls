import dataclasses
from typing import Type

import numpy as onp
from jax import numpy as jnp
from overrides import overrides

import jaxfg


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class State:
    params: jnp.ndarray
    """Length-4 array. (position, velocity), each 2D."""

    @overrides
    def __repr__(self) -> str:
        position = jnp.round(self.position, 5)
        velocity = jnp.round(self.velocity, 5)
        return f"(pos: {position}, vel: {velocity})"

    @staticmethod
    def make(position: jnp.ndarray, velocity: jnp.ndarray) -> "State":
        return State(
            params=jnp.concatenate(
                [
                    position,
                    velocity,
                ],
                axis=-1,
            )
        )

    @property
    def position(self):
        return self.params[..., :2]

    @property
    def velocity(self):
        return self.params[..., 2:]


class StateVariable(jaxfg.core.VariableBase):  # type: ignore
    """State of our system. A length-4 array: po"""

    @staticmethod
    @overrides
    def get_parameter_dim() -> int:
        return 4

    @staticmethod
    @overrides
    def get_local_parameter_dim() -> int:
        return 4

    @staticmethod
    @overrides
    def get_default_value() -> onp.ndarray:
        return State(params=onp.zeros(4))

    @staticmethod
    @overrides
    def add_local(x: State, local_delta: jnp.ndarray) -> State:
        return State(params=x.params + local_delta)

    @staticmethod
    @overrides
    def subtract_local(x: State, y: State) -> jnp.ndarray:
        return x.params - y.params

    @staticmethod
    @overrides
    def flatten(x: State) -> jnp.ndarray:
        return x.params

    @staticmethod
    @overrides
    def unflatten(flat: jnp.ndarray) -> State:
        return State(flat)


@dataclasses.dataclass(frozen=True)
class VisionFactor(jaxfg.core.FactorBase):
    predicted_position: jnp.ndarray

    @staticmethod
    def make(
        state_variable: StateVariable,
        position: jnp.ndarray,
        scale_tril_inv: jaxfg.types.ScaleTrilInv,
    ) -> "VisionFactor":
        return VisionFactor(
            variables=(state_variable,),
            scale_tril_inv=scale_tril_inv,
            predicted_position=position,
        )

    @overrides
    def compute_residual_vector(self, state_value: State):
        return state_value.position - self.predicted_position


@dataclasses.dataclass(frozen=True)
class DummyVelocityFactor(jaxfg.core.FactorBase):
    @staticmethod
    def make(
        state_variable: StateVariable,
    ) -> "DummyVelocityFactor":
        return DummyVelocityFactor(
            variables=(state_variable,),
            scale_tril_inv=onp.identity(2) * 2.0,
        )

    @overrides
    def compute_residual_vector(self, state_value: State):
        # Try to set velocities to some constant
        return state_value.velocity + 5.0


SPRING_CONSTANT = 0.05
DRAG_CONSTANT = 0.0075
POSITION_NOISE_STD = 1.0  # 0.1
VELOCITY_NOISE_STD = 2.0

SCALE_TRIL_INV = onp.diag(
    State.make(
        position=onp.ones(2) / POSITION_NOISE_STD,
        velocity=onp.ones(2) / VELOCITY_NOISE_STD,
    ).params
)
assert SCALE_TRIL_INV.shape == (4, 4)


@dataclasses.dataclass(frozen=True)
class DynamicsFactor(jaxfg.core.FactorBase):
    @staticmethod
    def make(
        before_variable: StateVariable,
        after_variable: StateVariable,
    ) -> "DynamicsFactor":
        return DynamicsFactor(
            variables=(before_variable, after_variable),
            scale_tril_inv=SCALE_TRIL_INV,
        )

    @overrides
    def compute_residual_vector(self, before_value: State, after_value: State):
        # Predict the state after our dynamics update
        spring_force = -SPRING_CONSTANT * before_value.position
        drag_force = (
            -DRAG_CONSTANT
            * jnp.sign(before_value.velocity)
            * (before_value.velocity ** 2)
        )
        pred_value = State.make(
            position=before_value.position + before_value.velocity,
            velocity=before_value.velocity + spring_force + drag_force,
        )

        # Compare predicted update vs variable
        return pred_value.params - after_value.params
