"""Tests for deprecated API patterns.

WARNING: This file tests deprecated API patterns that should NOT be used in new code.
These patterns are maintained for backwards compatibility only.

Preferred patterns:
- Use `Cost.factory` instead of `Cost.create_factory`
- Use `Cost(...)` directly instead of `Cost.make(...)`
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

import jaxls


class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array(0.0)):
    pass


def test_cost_create_factory_deprecated():
    """Test that Cost.create_factory raises a deprecation warning.

    DEPRECATED: Use Cost.factory instead.
    """
    var = ScalarVar(0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # DEPRECATED: Use @jaxls.Cost.factory instead
        @jaxls.Cost.create_factory  # type: ignore[attr-defined]
        def cost_fn(vals: jaxls.VarValues, var: ScalarVar, target: float) -> jax.Array:
            return jnp.array([vals[var] - target])

        # Verify deprecation warning was raised
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1
        assert any("Cost.factory" in str(x.message) for x in deprecation_warnings)

    # Verify the cost still works
    cost = cost_fn(var, 1.0)  # type: ignore
    problem = jaxls.LeastSquaresProblem(costs=[cost], variables=[var]).analyze()
    solution = problem.solve(verbose=False)
    assert jnp.abs(solution[var] - 1.0) < 1e-4


def test_cost_create_factory_with_kind_deprecated():
    """Test that Cost.create_factory with kind parameter raises deprecation warning.

    DEPRECATED: Use Cost.factory(kind=...) instead.
    """
    var = ScalarVar(0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # DEPRECATED: Use @jaxls.Cost.factory(kind="constraint_eq_zero") instead
        @jaxls.Cost.create_factory(kind="constraint_eq_zero")  # type: ignore
        def constraint_fn(
            vals: jaxls.VarValues, var: ScalarVar, target: float
        ) -> jax.Array:
            return jnp.array([vals[var] - target])

        # Verify deprecation warning was raised
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1
        assert any("Cost.factory" in str(x.message) for x in deprecation_warnings)

    # PREFERRED: This is the new pattern
    @jaxls.Cost.factory
    def cost_fn(vals: jaxls.VarValues, var: ScalarVar, target: float) -> jax.Array:
        return jnp.array([vals[var] - target])

    # Verify both costs work together
    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var, 2.0), constraint_fn(var, 1.0)],  # type: ignore
        variables=[var],
    ).analyze()
    solution = problem.solve(verbose=False)
    assert jnp.abs(solution[var] - 1.0) < 1e-4


def test_cost_make_deprecated():
    """Test that Cost.make raises a deprecation warning.

    DEPRECATED: Use Cost(...) directly instead of Cost.make(...).
    """
    var = ScalarVar(0)

    def my_cost_fn(vals: jaxls.VarValues, var: ScalarVar) -> jax.Array:
        return jnp.array([vals[var] - 1.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # DEPRECATED: Use Cost(...) directly
        cost = jaxls.Cost.make(  # type: ignore
            compute_residual=my_cost_fn,
            args=(var,),
        )

        # Verify deprecation warning was raised
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1

    # Verify the cost still works
    problem = jaxls.LeastSquaresProblem(costs=[cost], variables=[var]).analyze()
    solution = problem.solve(verbose=False)
    assert jnp.abs(solution[var] - 1.0) < 1e-4


def test_preferred_api_no_warnings():
    """Verify that the preferred API does not raise deprecation warnings.

    PREFERRED: Use Cost.factory and Cost(...) directly.
    """
    var = ScalarVar(0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # PREFERRED: Use Cost.factory decorator
        @jaxls.Cost.factory
        def cost_fn(vals: jaxls.VarValues, var: ScalarVar, target: float) -> jax.Array:
            return jnp.array([vals[var] - target])

        # PREFERRED: Use Cost.factory with kind parameter
        @jaxls.Cost.factory(kind="constraint_eq_zero")
        def constraint_fn(
            vals: jaxls.VarValues, var: ScalarVar, target: float
        ) -> jax.Array:
            return jnp.array([vals[var] - target])

        # PREFERRED: Use Cost(...) directly
        direct_cost = jaxls.Cost(
            compute_residual=lambda vals, var: jnp.array([vals[var]]),
            args=(var,),
        )

        # No deprecation warnings should be raised
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0

    # Verify all costs work
    problem = jaxls.LeastSquaresProblem(
        costs=[cost_fn(var, 2.0), constraint_fn(var, 1.0), direct_cost],
        variables=[var],
    ).analyze()
    solution = problem.solve(verbose=False)
    assert jnp.abs(solution[var] - 1.0) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
