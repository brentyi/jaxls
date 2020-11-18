import abc
import contextlib
from typing import TYPE_CHECKING, Dict, Generator, Set, Tuple

from jax import numpy as jnp
from overrides import overrides

if TYPE_CHECKING:
    from . import LinearFactor


class VariableBase(abc.ABC):
    def __init__(self, parameter_dim: int):
        self.parameter_dim = parameter_dim

    @contextlib.contextmanager
    @abc.abstractmethod
    def local_parameterization(self) -> Generator["RealVectorVariable", None, None]:
        ...


class RealVectorVariable(VariableBase):
    @contextlib.contextmanager
    @overrides
    def local_parameterization(self) -> Generator["RealVectorVariable", None, None]:
        """No special manifold; local parameterization is just self."""
        yield self

    def compute_error_dual(
        self,
        factors: Set["LinearFactor"],
        error_from_factor: Dict["LinearFactor", jnp.ndarray] = None,
    ):
        dual = jnp.zeros(self.parameter_dim)
        if error_from_factor is None:
            for factor in factors:
                dual = dual + factor.A_from_variable[self].T @ factor.b
        else:
            for factor in factors:
                dual = dual + factor.A_from_variable[self].T @ error_from_factor[factor]
        return dual
