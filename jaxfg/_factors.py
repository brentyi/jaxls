import abc
from typing import Any, Callable, Dict, Hashable, Iterable, List, Protocol, Tuple, Type

import fannypack
import torch
import torchfilter
from overrides import overrides

from ._variables import RealVectorVariable, Variable


class Factor:
    @abc.abstractmethod
    def evaluate_log_likelihood(self) -> torch.Tensor:
        """Evaluate the log-likelihood associated with a factor, conditioned on the
        current values of the associated variables.
        """
        # return self.evaluate_log_likelihood_parallel((self,))

    @classmethod
    def evaluate_log_likelihood_group(cls, group: Iterable["Factor"]) -> torch.Tensor:
        """Evaluate the log-likelihood of multiple factors and sum.
        This should generally parallelize computations.

        Args:
            factors (Iterable[Factor]): Factors to evaluate.

        Returns:
            torch.Tensor: Evaluated log-likelihood. Shape should be `(N,)`.
        """
        log_likelihoods: List[torch.Tensor] = []
        for factor in group:
            log_likelihoods.append(factor.evaluate_log_likelihood())
            assert len(log_likelihoods[-1].shape) == 1, "Shape should be `(N,)`"

        return torch.sum(
            torch.stack(log_likelihoods, dim=0),
            dim=0,
        )

    def group_key(self) -> Hashable:
        """Key used for determining which factors can be computed in parallel.

        Returns:
            Hashable: Key to organize factors by.
        """
        return self


class MultivariateNormalFromTensorProtocol(Protocol):
    """Protocol for functions that map tensors to multivariate normals.

    We represent distributions as `(mu, scale_tril)` pairs, where `scale_tril` is the
    lower-triangular Cholesky decomposition of our covariance matrix.

    Note that we add support for additional tensors passed in as keyword arguments; this
    is to make parallelism easier in the future.
    """

    def __call__(
        self, __x: torch.Tensor, **kwargs: Any
    ) -> Tuple[torch.Tensor, torchfilter.types.ScaleTrilTorch]:
        ...


class PriorFactor(Factor):
    """Simple factor for defining a multivariate normal prior on a variable node."""

    def __init__(
        self,
        node: Variable,
        prior_mean: torch.Tensor,
        prior_scale_tril: torchfilter.types.ScaleTrilTorch,
    ):
        N, variable_dim = prior_mean.shape
        assert prior_scale_tril.shape == (N, variable_dim, variable_dim)

        self.node = node
        self.prior_mean = prior_mean
        self.prior_scale_tril = prior_scale_tril

    @overrides
    def evaluate_log_likelihood(self) -> torch.Tensor:
        # Compute error
        error = self.node.compute_error(self.prior_mean)

        # Return log probability
        error_distribution = torch.distributions.MultivariateNormal(
            loc=error.new_zeros(error.shape[-1]),
            scale_tril=self.prior_scale_tril,
        )
        return error_distribution.log_prob(error)

    @overrides
    def evaluate_log_likelihood_group(
        cls, group: Iterable["PriorFactor"]
    ) -> torch.Tensor:
        # Gather some information about what our factors should look like
        example_factor = next(iter(group))
        node_type = type(example_factor.node)
        N, dim = example_factor.node.value.shape
        prior_mean_shape = example_factor.prior_mean.shape
        prior_scale_tril_shape = example_factor.prior_scale_tril.shape

        # Stack inputs
        node_values = []
        prior_means = []
        prior_scale_trils = []
        for factor in group:
            assert type(factor.node) == node_type

            assert factor.node.value.shape == (N, dim)
            node_values.append(factor.node.value)

            assert factor.prior_mean.shape == prior_mean_shape
            prior_means.append(factor.prior_mean)

            assert factor.prior_scale_tril.shape == prior_scale_tril_shape
            prior_scale_trils.append(factor.prior_scale_tril)

        # Create dummy node, factor
        factor_count = len(node_values)
        values_concat = torch.stack(node_values, dim=0).reshape((factor_count * N, dim))
        prior_means_concat = torch.stack(prior_means, dim=0).reshape(
            (factor_count * N, dim)
        )
        prior_scale_trils_concat = torch.stack(prior_scale_trils, dim=0).reshape(
            (factor_count * N, dim, dim)
        )

        # Compute log likelihoods and sum
        log_likelihood_concat = PriorFactor(
            node=node_type(values_concat),
            prior_mean=prior_means_concat,
            prior_scale_tril=prior_scale_trils_concat,
        ).evaluate_log_likelihood()
        assert log_likelihood_concat.shape == (factor_count * N,)

        return torch.sum(log_likelihood_concat.reshape((factor_count, N)), dim=0)

    @overrides
    def group_key(self) -> Hashable:
        # We can evaluate prior factors in parallel if they have the same dimension +
        # factor type
        return (PriorFactor, type(self.node), self.prior_mean.shape[-1])


class TransitionFactor(Factor):
    """Factor representing a transition function, with before/after nodes."""

    def __init__(
        self,
        before_node: Variable,
        after_node: Variable,
        transition_fn: MultivariateNormalFromTensorProtocol,
        *,
        transition_fn_kwargs: Dict[str, torch.Tensor] = {},
    ):
        assert (
            before_node.value.shape[0] == after_node.value.shape[0]
        ), "Batch size must match!"
        self.before_node = before_node
        self.after_node = after_node
        self.transition_fn = transition_fn
        self.transition_fn_kwargs = transition_fn_kwargs

    @overrides
    def evaluate_log_likelihood(self) -> torch.Tensor:
        # Run forward prediction
        pred_value: torch.Tensor
        pred_scale_tril: torchfilter.types.ScaleTrilTorch
        pred_value, pred_scale_tril = self.transition_fn(
            self.before_node.value, **self.transition_fn_kwargs
        )
        assert (
            pred_value.shape == self.after_node.value.shape
        ), "Predicted value shape mismatch"

        # Compute error
        error = self.after_node.compute_error(pred_value)

        # Return log probability
        zeros = pred_scale_tril.new_zeros((1,) * len(error.shape)).expand(error.shape)
        error_distribution = torch.distributions.MultivariateNormal(
            loc=zeros,
            scale_tril=pred_scale_tril,
        )
        return error_distribution.log_prob(error)

    @classmethod
    @overrides
    def evaluate_log_likelihood_group(
        cls, group: Iterable["TransitionFactor"]
    ) -> torch.Tensor:
        # Gather some information about what our factors look like
        example_factor = next(iter(group))
        before_node_type = type(example_factor.before_node)
        after_node_type = type(example_factor.after_node)
        transition_fn = example_factor.transition_fn
        N, _ = example_factor.before_node.value.shape

        # Stack inputs
        before_values = []
        after_values = []
        transition_fn_kwargs: fannypack.utils.SliceWrapper[
            Dict[str, List[torch.Tensor]]
        ] = fannypack.utils.SliceWrapper({})

        for factor in group:
            before_values.append(factor.before_node.value)
            after_values.append(factor.after_node.value)
            transition_fn_kwargs.append(factor.transition_fn_kwargs)

        # Create dummy node, factor
        factor_count = len(before_values)
        before_values_concat = torch.stack(before_values, dim=0).flatten(
            start_dim=0, end_dim=1
        )
        after_values_concat = torch.stack(after_values, dim=0).flatten(
            start_dim=0, end_dim=1
        )
        transition_fn_kwargs_concat = transition_fn_kwargs.map(
            lambda l: torch.stack(l, dim=0).flatten(start_dim=0, end_dim=1)
        )

        # Compute log likelihoods and sum
        log_likelihood_concat = TransitionFactor(
            before_node=before_node_type(before_values_concat),
            after_node=after_node_type(after_values_concat),
            transition_fn=transition_fn,
            transition_fn_kwargs=transition_fn_kwargs_concat,
        ).evaluate_log_likelihood()
        assert log_likelihood_concat.shape == (factor_count * N,)

        return torch.sum(log_likelihood_concat.reshape((factor_count, N)), dim=0)

    @overrides
    def group_key(self) -> Hashable:
        # We can evaluate transition factors in parallel when they use the same
        # transition function
        return (TransitionFactor, self.transition_fn)


class ObservationFactor(TransitionFactor):
    """Factor representing an observation made on a node.

    Implemented as a simple wrapper around `TransitionFactor`.
    """

    def __init__(
        self,
        node: Variable,
        observation: torch.Tensor,
        observation_fn: MultivariateNormalFromTensorProtocol,
        *,
        observation_fn_kwargs: Dict[str, torch.Tensor] = {},
        observation_type: Type[Variable] = RealVectorVariable,
    ):
        observation_node = observation_type(observation)
        super().__init__(
            before_node=node,
            after_node=observation_node,
            transition_fn=observation_fn,
            transition_fn_kwargs=observation_fn_kwargs,
        )
