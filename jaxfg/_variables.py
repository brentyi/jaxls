import abc

import torch
from overrides import overrides


class Variable(abc.ABC):
    """Container for variable nodes in our factor graphs."""

    def __init__(self, value: torch.Tensor):
        self.value = value
        """torch.Tensor: Underlying value of this Variable."""

        if value is not None:
            self.validate()

    @abc.abstractmethod
    def validate(self):
        """Validate the contained value."""

    @abc.abstractmethod
    def compute_error(self, other: torch.Tensor) -> torch.Tensor:
        """Compute an error vector between this variable and another."""


class RealVectorVariable(Variable):
    @overrides
    def validate(self):
        assert len(self.value.shape) == 2, "Shape of value should be (N, dim)"

    @overrides
    def compute_error(self, other_value: torch.Tensor) -> torch.Tensor:
        assert other_value.shape == self.value.shape, "Shape of values must match!"
        return self.value - other_value
