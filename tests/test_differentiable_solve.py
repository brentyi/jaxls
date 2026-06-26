"""Contract tests for differentiable least-squares solving."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxlie
import pytest

import jaxls


class Vec3Var(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(3)):
    """Simple Euclidean variable used for differentiable solve tests."""


@jaxls.Cost.factory
def _euclidean_prior(
    vals: jaxls.VarValues,
    var: Vec3Var,
    target: jax.Array,
) -> jax.Array:
    return vals[var] - target


def _euclidean_loss(
    target: jax.Array,
    *,
    linear_solver: str = "dense_cholesky",
) -> jax.Array:
    var = Vec3Var(0)
    problem = jaxls.LeastSquaresProblem([_euclidean_prior(var, target)], [var]).analyze()
    solution = problem.solve_differentiable(
        linear_solver=linear_solver,
        verbose=False,
    )
    return jnp.sum(solution[var] ** 2)


@jaxls.Cost.factory
def _se2_prior(
    vals: jaxls.VarValues,
    var: jaxls.SE2Var,
    target: jaxlie.SE2,
) -> jax.Array:
    return (vals[var] @ target.inverse()).log()


def test_gradient_through_simple_euclidean_problem() -> None:
    """Gradient should match closed-form behavior for x*=target."""
    target = jnp.array([1.0, 2.0, 3.0])
    grad = jax.grad(_euclidean_loss)(target)

    assert jnp.allclose(grad, 2 * target, atol=1e-4)


def test_gradient_through_lie_group_problem() -> None:
    """Gradients should propagate through SE2 variables and tangent mapping."""
    var = jaxls.SE2Var(0)

    def loss_fn(params: jax.Array) -> jax.Array:
        target = jaxlie.SE2.from_xy_theta(params[0], params[1], params[2])
        problem = jaxls.LeastSquaresProblem([_se2_prior(var, target)], [var]).analyze()
        solution = problem.solve_differentiable(
            linear_solver="dense_cholesky",
            verbose=False,
        )
        return jnp.sum(solution[var].translation() ** 2)

    params = jnp.array([1.0, -2.0, 0.5])
    grad = jax.grad(loss_fn)(params)

    expected = jnp.array([2.0, -4.0, 0.0])
    assert jnp.allclose(grad, expected, atol=1e-4)


def test_jit_compatibility() -> None:
    """Differentiable solve should work under jax.jit."""

    @jax.jit
    def loss_and_grad(target: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _euclidean_loss(target), jax.grad(_euclidean_loss)(target)

    target = jnp.array([1.0, 2.0, 3.0])
    loss, grad = loss_and_grad(target)

    assert jnp.allclose(loss, jnp.sum(target**2), atol=1e-4)
    assert jnp.allclose(grad, 2 * target, atol=1e-4)


def test_vmap_compatibility() -> None:
    """Differentiable solve should support batched usage with vmap."""

    def loss_and_grad(target: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _euclidean_loss(target), jax.grad(_euclidean_loss)(target)

    targets = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ]
    )
    losses, grads = jax.vmap(loss_and_grad)(targets)

    assert jnp.allclose(losses, jnp.sum(targets**2, axis=1), atol=1e-4)
    assert jnp.allclose(grads, 2 * targets, atol=1e-4)


def test_linear_solver_parity() -> None:
    """CG and dense Cholesky should produce the same gradients."""
    target = jnp.array([1.0, 2.0, 3.0])

    grad_cg = jax.grad(
        lambda t: _euclidean_loss(t, linear_solver="conjugate_gradient")
    )(target)
    grad_dense = jax.grad(
        lambda t: _euclidean_loss(t, linear_solver="dense_cholesky")
    )(target)

    assert jnp.allclose(grad_cg, 2 * target, atol=1e-4)
    assert jnp.allclose(grad_dense, 2 * target, atol=1e-4)
    assert jnp.allclose(grad_cg, grad_dense, atol=1e-4)


def test_constraints_rejected() -> None:
    """Differentiable solve should reject constrained problems."""

    class Vec2Var(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(2)):
        pass

    @jaxls.Cost.factory(kind="constraint_eq_zero")
    def equality_constraint(vals: jaxls.VarValues, var: Vec2Var) -> jax.Array:
        return vals[var][0] - 1.0

    var = Vec2Var(0)
    problem = jaxls.LeastSquaresProblem([equality_constraint(var)], [var]).analyze()

    with pytest.raises(ValueError, match="[Cc]onstraint"):
        problem.solve_differentiable(verbose=False)


def test_cholmod_rejected_for_differentiable_solve() -> None:
    """Differentiable solve should reject CHOLMOD linear solver."""
    var = Vec3Var(0)
    target = jnp.array([1.0, 2.0, 3.0])
    problem = jaxls.LeastSquaresProblem([_euclidean_prior(var, target)], [var]).analyze()

    with pytest.raises(ValueError, match="CHOLMOD|cholmod"):
        problem.solve_differentiable(linear_solver="cholmod", verbose=False)
