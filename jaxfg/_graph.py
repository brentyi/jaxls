import abc
from typing import Dict, Hashable, Iterable, List, Set

import torch
from tqdm.auto import tqdm

from ._factors import Factor
from ._variables import Variable


class Solver(abc.ABC):
    """Abstract class for non-linear least squares solvers?"""


class FactorGraph:
    def __init__(self):
        self.factors: Dict[Hashable, Set[Factor]] = {}
        """Container for storing factors in our graph. Note that factors are organized
        by a group key, which dictates whether they can be computed in parallel or not."""

    def add_factors(self, *to_add: Factor):
        """Add factor(s) to our graph."""

        for factor in to_add:
            group_key = factor.group_key()

            # Create group if it doesn't exist
            if group_key not in self.factors:
                self.factors[group_key] = set()

            # Add factor to group
            group = self.factors[group_key]
            assert factor not in group, "Factor already added!"
            group.add(factor)

    def remove_factors(self, *to_remove: Factor):
        """Remove factor(s) from our graph."""

        for factor in to_remove:
            group_key = factor.group_key()

            # Remove factor from group
            group = self.factors[group_key]
            assert factor in group, "Factor not in graph!"
            group.remove(factor)

            # Remove group if empty
            if len(group) == 0:
                self.factors.pop(group_key)

    def evaluate_log_likelihood(self, grouped: bool = False):
        """Compute log-likelihood of each factor and sum.

        Args:
            grouped (bool, optional): If set, evaluates grouped factors in parallel.
                Defaults to `True`.
        """

        assert len(self.factors) > 0, "Graph cannot be empty"

        log_likelihoods: List[torch.Tensor] = []
        group: Set[Factor]
        for group in self.factors.values():
            if grouped:
                # Evaluate in parallel
                factor: Factor = next(iter(group))
                log_likelihoods.append(factor.evaluate_log_likelihood_group(group))
            else:
                # Don't evaluate
                for factor in group:
                    log_likelihoods.append(factor.evaluate_log_likelihood())

        log_likelihood = torch.sum(
            torch.stack(log_likelihoods, dim=0),
            dim=0,
        )
        assert len(log_likelihood.shape) == 1, "Shape of likelihoods should be (N,)"

        return log_likelihood

    def solve_map_inference(self, nodes: Iterable[Variable], num_iters=100, solver: Solver = None):
        """Solve MAP inference problem for input variables."""

        parameters = [node.value for node in nodes]

        for p in parameters:
            p.requires_grad = True

        optimizer = torch.optim.SGD(parameters, lr=1e-5)
        for i in tqdm(range(num_iters)):
            optimizer.zero_grad()
            nll_loss = -torch.mean(self.evaluate_log_likelihood())
            nll_loss.backward()
            optimizer.step()


__all__ = ["FactorGraph", "Solver"]
