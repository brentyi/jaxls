import dataclasses

import jaxlie
from jax import numpy as jnp
from overrides import overrides

import jaxfg


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class State:
    pose: jaxlie.SE2
    velocities: jnp.ndarray
    """Length-2 array: (linear_vel, angular_vel)."""

    @overrides
    def __repr__(self) -> str:
        return (
            "("
            + ", ".join(
                [
                    f"{k}: {v:.5f}"
                    for k, v in {
                        "x": self.x,
                        "y": self.y,
                        "theta": self.theta,
                        "linear_vel": self.linear_vel,
                        "angular_vel": self.angular_vel,
                    }.items()
                ]
            )
            + ")"
        )

    @classmethod
    def make(
        cls,
        pose: jaxlie.SE2,
        linear_vel: jaxlie.types.Scalar,
        angular_vel: jaxlie.types.Scalar,
    ) -> "State":
        return cls(
            pose=pose,
            velocities=jnp.array(
                [linear_vel, angular_vel],
            )
        )

    @property
    def x(self):
        return self.pose.translation[0]

    @property
    def y(self):
        return self.pose.translation[1]

    @property
    def theta(self):
        return self.pose.rotation.as_radians()

    @property
    def linear_vel(self):
        return self.velocities[..., 0]

    @property
    def angular_vel(self):
        return self.velocities[..., 1]


class StateVariable(jaxfg.core.VariableBase):  # type: ignore
    """State of our system."""

    @staticmethod
    @overrides
    def get_parameter_dim() -> int:
        return 6  # (x, y, cos, sin, linear, angular)

    @staticmethod
    @overrides
    def get_local_parameter_dim() -> int:
        return 5  # (x, y, theta, linear, angular)

    @staticmethod
    @overrides
    def get_default_value() -> State:
        return State(pose=jaxlie.SE2.identity(), velocities=jnp.zeros(2))

    @staticmethod
    @overrides
    def manifold_retract(x: State, local_delta: jnp.ndarray) -> State:
        return State(
            pose=x.pose @ jaxlie.SE2.exp(local_delta[:3]),
            velocities=x.velocities + local_delta[3:5],
        )

    @staticmethod
    @overrides
    def manifold_inverse_retract(x: State, y: State) -> jnp.ndarray:
        return jnp.concatenate(
            ((x.pose.inverse() @ y.pose).log(), x.velocities - y.velocities), axis=-1
        )

    @staticmethod
    @overrides
    def flatten(x: State) -> jnp.ndarray:
        return jnp.concatenate([x.pose.parameters, x.velocities], axis=-1)

    @staticmethod
    @overrides
    def unflatten(flat: jnp.ndarray) -> State:
        return State(
            pose=jaxlie.SE2(xy_unit_complex=flat[..., :4]), velocities=flat[..., 4:6]
        )


@dataclasses.dataclass(frozen=True)
class VisionFactor(jaxfg.core.FactorBase):
    predicted_velocity: jnp.ndarray

    @staticmethod
    def make(
        state_variable: StateVariable,
        predicted_velocity: jnp.ndarray,
        scale_tril_inv: jaxfg.types.ScaleTrilInv,
    ) -> "VisionFactor":
        assert scale_tril_inv.shape[-2:] == (2, 2)
        return VisionFactor(
            variables=(state_variable,),
            scale_tril_inv=scale_tril_inv,
            predicted_velocity=predicted_velocity,
        )

    @overrides
    def compute_residual_vector(self, state_value: State):
        return state_value.velocities - self.predicted_velocity


def dynamics_forward(state: State) -> State:
    # Predict the state after our dynamics update
    return State(
        pose=state.pose
        @ jaxlie.SE2.from_rotation_and_translation(
            rotation=jaxlie.SO2.from_radians(state.angular_vel),
            translation=jnp.zeros(2).at[0].set(state.linear_vel),
        ),
        velocities=state.velocities,
    )


@dataclasses.dataclass(frozen=True)
class DynamicsFactor(jaxfg.core.FactorBase):
    @staticmethod
    def make(
        before_variable: StateVariable,
        after_variable: StateVariable,
        scale_tril_inv: jaxfg.types.ScaleTrilInv,
    ) -> "DynamicsFactor":
        assert scale_tril_inv.shape[-2:] == (5, 5)
        return DynamicsFactor(
            variables=(before_variable, after_variable),
            scale_tril_inv=scale_tril_inv,
        )

    @overrides
    def compute_residual_vector(self, before_value: State, after_value: State):
        pred_value = dynamics_forward(before_value)
        return StateVariable.manifold_inverse_retract(pred_value, after_value)


@dataclasses.dataclass(frozen=True)
class PriorFactor(jaxfg.core.FactorBase):
    mu: State

    @staticmethod
    def make(
        variable: StateVariable,
        mu: State,
        scale_tril_inv: jaxfg.types.ScaleTrilInv,
    ) -> "DynamicsFactor":
        assert scale_tril_inv.shape[-2:] == (5, 5)
        return PriorFactor(
            mu=mu,
            variables=(variable,),
            scale_tril_inv=scale_tril_inv,
        )

    @overrides
    def compute_residual_vector(self, value):
        return StateVariable.manifold_inverse_retract(value, self.mu)
