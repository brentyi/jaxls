"""Cost groups must be ordered by what they are, not by where they live."""

import re

import jax
import jax.numpy as jnp

import jaxls
from jaxls._problem import _sort_key


class _Point(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(2)):
    pass


@jaxls.Cost.factory(name="pull_to_zero")
def _pull_to_zero(vals: jaxls.VarValues, var: _Point) -> jax.Array:
    return vals[var]


@jaxls.Cost.factory(name="pull_to_one")
def _pull_to_one(vals: jaxls.VarValues, var: _Point) -> jax.Array:
    return vals[var] - 1.0


def _group_key(cost: jaxls.Cost):
    batch_axes = cost._get_batch_axes()
    return (
        jax.tree.structure(cost),
        tuple(
            leaf.shape[len(batch_axes) :] if hasattr(leaf, "shape") else ()
            for leaf in jax.tree.leaves(cost)
        ),
    )


def test_sort_key_has_no_function_address() -> None:
    var = _Point(0)
    for cost in (_pull_to_zero(var), _pull_to_one(var)):
        key = _sort_key(_group_key(cost))
        assert re.search(r" at 0x[0-9a-fA-F]+", key) is None, (
            "the sort key still carries a function address, so the cost "
            "group order depends on allocation and differs between "
            f"processes: {key[:120]!r}"
        )


def test_sort_key_still_separates_costs_by_name() -> None:
    var = _Point(0)
    zero_key = _sort_key(_group_key(_pull_to_zero(var)))
    one_key = _sort_key(_group_key(_pull_to_one(var)))
    assert zero_key != one_key
    assert "pull_to_zero" in zero_key
    assert "pull_to_one" in one_key
