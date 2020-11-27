import abc
import contextlib
import dataclasses
import time
from typing import Generator, Tuple

import jax
import numpy as onp
import termcolor
from jax import numpy as jnp
from overrides import overrides


@contextlib.contextmanager
def stopwatch(label: str = "unlabeled block") -> Generator[None, None, None]:
    start_time = time.time()
    print(f"\n========")
    print(f"Running ({label})")
    yield
    print(f"{termcolor.colored(str(time.time() - start_time), attrs=['bold'])} seconds")
    print(f"========")


def immutable_dataclass(cls):
    """Decorator for defining immutable dataclasses."""

    # Hash based on object ID, rather than contents
    cls.__hash__ = object.__hash__

    return dataclasses.dataclass(cls, frozen=True)


def get_epsilon(x: jnp.ndarray) -> float:
    if x.dtype is jnp.dtype("float32"):
        return 1e-5
    elif x.dtype is jnp.dtype("float64"):
        return 1e-10
    else:
        assert False, f"Unexpected array type: {x.dtype}"


Matrix = jnp.ndarray
TangentVector = jnp.ndarray


class MatrixLieGroup(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def get_matrix_dim() -> int:
        """Get dimension of (square) matrix.

        Returns:
            int: Matrix dimensionality.
        """

    @staticmethod
    @abc.abstractmethod
    def get_tangent_dim() -> int:
        """Get dimensionality of tangent space.

        Args:

        Returns:
            int: Tangent space dimension.
        """

    @classmethod
    def identity(cls) -> Matrix:
        """Returns identity element.

        Args:

        Returns:
            Matrix:
        """
        return onp.eye(cls.get_matrix_dim())

    @staticmethod
    @abc.abstractmethod
    def exp(tangent: TangentVector) -> Matrix:
        """Computes `expm(wedge(tangent))`.

        Args:
            tangent (TangentVector): Input.

        Returns:
            Matrix: Output.
        """

    @staticmethod
    @abc.abstractmethod
    def log(x: Matrix) -> TangentVector:
        """Computes `logm(vee(tangent))`.

        Args:
            x (Matrix): Input.

        Returns:
            TangentVector: Output.
        """

    @staticmethod
    @abc.abstractmethod
    def inverse(x: Matrix) -> Matrix:
        """Computes the inverse of x.

        Args:
            x (Matrix): Input.

        Returns:
            Matrix: Output.
        """


class SO2(MatrixLieGroup):
    @staticmethod
    @overrides
    def get_matrix_dim() -> int:
        return 2

    @staticmethod
    @overrides
    def get_tangent_dim() -> int:
        return 1

    @staticmethod
    @overrides
    def exp(tangent: TangentVector) -> Matrix:
        assert tangent.shape == (1,)
        theta = tangent[0]
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return jnp.array(
            [
                [cos, -sin],
                [sin, cos],
            ]
        )

    @staticmethod
    @overrides
    def log(x: Matrix) -> TangentVector:
        assert x.shape == (2, 2)
        return jnp.arctan2(x[1, 0, None], x[0, 0, None])

    @staticmethod
    @overrides
    def inverse(x: Matrix) -> Matrix:
        assert x.shape == (2, 2)
        return x.T


class SE2(MatrixLieGroup):
    @staticmethod
    @overrides
    def get_matrix_dim() -> int:
        return 3

    @staticmethod
    @overrides
    def get_tangent_dim() -> int:
        return 3

    @staticmethod
    @overrides
    def exp(tangent: TangentVector) -> Matrix:
        # See Gallier and Xu:
        # > https:///pdfs.semanticscholar.org/cfe3/e4b39de63c8cabd89bf3feff7f5449fc981d.pdf

        assert tangent.shape == (3,)

        def compute_taylor(theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            theta_sq = theta ** 2

            sin_over_theta = 1.0 - theta_sq / 6.0
            one_minus_cos_over_theta = 0.5 * theta - theta * theta_sq / 24.0
            return sin_over_theta, one_minus_cos_over_theta

        def compute_exact(theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            sin_over_theta = jnp.sin(theta) / theta
            one_minus_cos_over_theta = (1.0 - jnp.cos(theta)) / theta
            return sin_over_theta, one_minus_cos_over_theta

        theta = tangent[2]
        sin_over_theta, one_minus_cos_over_theta = jax.lax.cond(
            jnp.abs(theta) < get_epsilon(tangent),
            compute_taylor,
            compute_exact,
            operand=theta,
        )

        V = jnp.array(
            [
                [sin_over_theta, -one_minus_cos_over_theta],
                [one_minus_cos_over_theta, sin_over_theta],
            ]
        )

        T = jnp.eye(3)
        T = T.at[:2, :2].set(SO2.exp(tangent[2:3]))
        T = T.at[:2, 2].set(V @ tangent[:2])
        return T

    @staticmethod
    @overrides
    def log(x: Matrix) -> TangentVector:
        # See Gallier and Xu:
        # > https:///pdfs.semanticscholar.org/cfe3/e4b39de63c8cabd89bf3feff7f5449fc981d.pdf

        assert x.shape == (3, 3)

        theta = SO2.log(x[:2, :2])[0]

        cos = jnp.cos(theta)
        cos_minus_one = cos - 1.0
        half_theta = theta / 2.0
        half_theta_over_tan_half_theta = jax.lax.cond(
            jnp.abs(cos_minus_one) < get_epsilon(x),
            # First-order Taylor approximation
            lambda args: 1.0 - (args[0] ** 2) / 12.0,
            # Default
            lambda args: -(args[1] * jnp.sin(args[0])) / args[2],
            operand=(theta, half_theta, cos_minus_one),
        )

        V_inv = jnp.array(
            [
                [half_theta_over_tan_half_theta, half_theta],
                [-half_theta, half_theta_over_tan_half_theta],
            ]
        )

        tangent = jnp.zeros(3)
        tangent = tangent.at[:2].set(V_inv @ x[:2, 2])
        tangent = tangent.at[2].set(theta)
        return tangent

    @staticmethod
    @overrides
    def inverse(x: Matrix) -> Matrix:
        assert x.shape == (3, 3)
        R = x[:2, :2]
        t = x[:2, 2]
        inv = jnp.eye(3)
        inv = inv.at[:2, :2].set(R.T)
        inv = inv.at[:2, 2].set(-R.T @ t)
        assert inv.shape == (3, 3)
        return inv
