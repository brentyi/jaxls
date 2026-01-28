"""Test suite for differentiable solving via the adjoint method.

This test file follows an incremental prototype approach:
1. Trivial case (identity Jacobian) - validate custom_vjp mechanics
2. Over-determined system (non-trivial Jacobian) - validate (J^T J)^{-1} solve pattern
3. Iterative solve (Gauss-Newton loop) - validate adjoint works with iterative forward pass
4. Integration with jaxls types - use actual VarValues, Cost, etc.
5. Full integration - complete solve_differentiable() API
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxlie
import pytest

import jaxls


class TestStep1TrivialCase:
    """Step 1: Trivial case - validate custom_vjp mechanics work.

    For min_x ||x - target||^2:
    - Solution: x* = target
    - Jacobian: J = I (identity)
    - dx*/dtarget = I
    """

    def test_trivial_identity_gradient(self):
        """Test that gradients flow through a trivial 'solve' with identity Jacobian."""
        from jaxls._differentiable_solve import _solve_trivial

        target = jnp.array([1.0, 2.0, 3.0])

        # Forward pass should return target
        result = _solve_trivial(target)
        assert jnp.allclose(result, target)

        # Gradient of sum(x*^2) w.r.t. target should be 2*target
        # since x* = target, d(sum(x*^2))/dtarget = 2*x* = 2*target
        grad_fn = jax.grad(lambda t: jnp.sum(_solve_trivial(t) ** 2))
        grad = grad_fn(target)
        expected_grad = 2 * target
        assert jnp.allclose(grad, expected_grad)

    def test_trivial_jit_compatible(self):
        """Test that the trivial solve works under JIT."""
        from jaxls._differentiable_solve import _solve_trivial

        target = jnp.array([1.0, 2.0, 3.0])

        @jax.jit
        def loss_and_grad(t):
            loss = jnp.sum(_solve_trivial(t) ** 2)
            return loss, jax.grad(lambda x: jnp.sum(_solve_trivial(x) ** 2))(t)

        loss, grad = loss_and_grad(target)
        assert jnp.allclose(loss, jnp.sum(target**2))
        assert jnp.allclose(grad, 2 * target)


class TestStep2OverdeterminedSystem:
    """Step 2: Over-determined system - validate (J^T J)^{-1} solve pattern.

    For min_x ||Ax - b||^2 where A is tall (m > n):
    - Solution: x* = (A^T A)^{-1} A^T b
    - Adjoint: solve (A^T A) u = g, then grad_b = -A @ u
    """

    def test_overdetermined_linear_gradient(self):
        """Test gradient through overdetermined linear least squares."""
        from jaxls._differentiable_solve import _solve_overdetermined

        # Create a tall matrix A (more rows than columns)
        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, (5, 3))  # 5 equations, 3 unknowns
        b = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # The solution should be (A^T A)^{-1} A^T b
        expected_x = jnp.linalg.lstsq(A, b, rcond=None)[0]
        result = _solve_overdetermined(A, b)
        assert jnp.allclose(result, expected_x, atol=1e-5)

        # Test gradients via finite differences
        def loss_fn(b_param):
            x = _solve_overdetermined(A, b_param)
            return jnp.sum(x**2)

        grad_autodiff = jax.grad(loss_fn)(b)

        # Finite difference check
        eps = 1e-5
        grad_fd = jnp.zeros_like(b)
        for i in range(len(b)):
            b_plus = b.at[i].add(eps)
            b_minus = b.at[i].add(-eps)
            grad_fd = grad_fd.at[i].set(
                (loss_fn(b_plus) - loss_fn(b_minus)) / (2 * eps)
            )

        # Gradients should match within reasonable tolerance
        # Note: there can be some difference due to numerical precision in lstsq
        assert jnp.allclose(grad_autodiff, grad_fd, rtol=0.15, atol=0.15)

    def test_overdetermined_gradient_wrt_matrix(self):
        """Test gradient through overdetermined system w.r.t. the matrix A."""
        from jaxls._differentiable_solve import _solve_overdetermined

        key = jax.random.PRNGKey(1)
        A = jax.random.normal(key, (5, 3))
        b = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def loss_fn(A_param):
            x = _solve_overdetermined(A_param, b)
            return jnp.sum(x**2)

        grad_autodiff = jax.grad(loss_fn)(A)

        # Finite difference check
        eps = 1e-5
        grad_fd = jnp.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A_plus = A.at[i, j].add(eps)
                A_minus = A.at[i, j].add(-eps)
                grad_fd = grad_fd.at[i, j].set(
                    (loss_fn(A_plus) - loss_fn(A_minus)) / (2 * eps)
                )

        # Gradients should be close - some tolerance needed due to numerical precision
        assert jnp.allclose(grad_autodiff, grad_fd, rtol=0.1, atol=0.5)


class TestStep3IterativeSolve:
    """Step 3: Iterative solve - validate adjoint works with iterative forward pass.

    For nonlinear: min_x ||f(x) - target||^2
    Use Gauss-Newton iterations in forward pass.
    Backward: use Jacobian at converged solution.
    """

    def test_nonlinear_iterative_gradient(self):
        """Test gradient through nonlinear least squares with iterative solve."""
        from jaxls._differentiable_solve import _solve_nonlinear_iterative

        # Simple nonlinear problem: min_x ||x^2 - target||^2
        # Solution: x* = sqrt(target) (for positive target)
        target = jnp.array([1.0, 4.0, 9.0])

        result = _solve_nonlinear_iterative(target)
        expected = jnp.sqrt(target)
        assert jnp.allclose(result, expected, atol=1e-4)

        # Test gradients
        # For loss = sum(x*^2) where x* = sqrt(target), we have:
        # loss = sum(target), so dL/dtarget = [1, 1, 1]
        def loss_fn(t):
            x = _solve_nonlinear_iterative(t)
            return jnp.sum(x**2)

        grad_autodiff = jax.grad(loss_fn)(target)
        expected_grad = jnp.ones_like(target)

        # The adjoint method should give exact analytical gradient
        assert jnp.allclose(grad_autodiff, expected_grad, atol=1e-5)

        # Finite difference as sanity check (less precise due to iterative solve)
        eps = 1e-5
        grad_fd = jnp.zeros_like(target)
        for i in range(len(target)):
            t_plus = target.at[i].add(eps)
            t_minus = target.at[i].add(-eps)
            grad_fd = grad_fd.at[i].set(
                (loss_fn(t_plus) - loss_fn(t_minus)) / (2 * eps)
            )

        # FD should be close but not exact (iterative solve introduces small errors)
        assert jnp.allclose(grad_autodiff, grad_fd, rtol=0.1)


class TestStep4JaxlsTypes:
    """Step 4: Integration with jaxls types.

    Use actual VarValues, Cost, BlockRowSparseMatrix from jaxls.
    """

    def test_simple_single_variable_gradient(self):
        """Test gradient through jaxls solve with a single variable."""
        # Define a simple variable type for testing
        class ScalarVar(
            jaxls.Var[jax.Array],
            default_factory=lambda: jnp.zeros(2),
        ):
            pass

        # Simple prior cost: min ||x - target||^2
        @jaxls.Cost.factory
        def prior_cost(
            vals: jaxls.VarValues, var: ScalarVar, target: jax.Array
        ) -> jax.Array:
            return vals[var] - target

        var = ScalarVar(0)
        target = jnp.array([3.0, 4.0])

        def loss_fn(target_param):
            costs = [prior_cost(var, target_param)]
            problem = jaxls.LeastSquaresProblem(costs, [var]).analyze()
            solution = problem.solve_differentiable(
                linear_solver="dense_cholesky",
                verbose=False,
            )
            # Loss is some function of the solution
            return jnp.sum(solution[var] ** 2)

        # The solution should be x* = target
        # So loss = ||target||^2, and grad = 2*target
        grad_autodiff = jax.grad(loss_fn)(target)
        expected_grad = 2 * target
        assert jnp.allclose(grad_autodiff, expected_grad, atol=1e-4)

    def test_lie_group_variable_gradient(self):
        """Test gradient through jaxls solve with SE2 variable."""

        @jaxls.Cost.factory
        def prior_cost(
            vals: jaxls.VarValues, var: jaxls.SE2Var, target: jaxlie.SE2
        ) -> jax.Array:
            return (vals[var] @ target.inverse()).log()

        var = jaxls.SE2Var(0)
        target_params = jnp.array([1.0, 2.0, 0.5])  # x, y, theta

        def loss_fn(params):
            target = jaxlie.SE2.from_xy_theta(params[0], params[1], params[2])
            costs = [prior_cost(var, target)]
            problem = jaxls.LeastSquaresProblem(costs, [var]).analyze()
            solution = problem.solve_differentiable(
                linear_solver="dense_cholesky",
                verbose=False,
            )
            # Loss is the log map norm
            return jnp.sum(solution[var].log() ** 2)

        # Test that gradients can be computed
        grad_autodiff = jax.grad(loss_fn)(target_params)

        # Finite difference check
        eps = 1e-5
        grad_fd = jnp.zeros_like(target_params)
        for i in range(len(target_params)):
            p_plus = target_params.at[i].add(eps)
            p_minus = target_params.at[i].add(-eps)
            grad_fd = grad_fd.at[i].set(
                (loss_fn(p_plus) - loss_fn(p_minus)) / (2 * eps)
            )

        # Gradients should be close - some difference expected due to numerical precision
        assert jnp.allclose(grad_autodiff, grad_fd, rtol=0.1, atol=0.15)


class TestStep5FullIntegration:
    """Step 5: Full integration with solve_differentiable() API."""

    def test_pose_graph_differentiable(self):
        """Test differentiable solve on a pose graph problem."""

        @jaxls.Cost.factory
        def prior_cost(
            vals: jaxls.VarValues, var: jaxls.SE2Var, target: jaxlie.SE2
        ) -> jax.Array:
            return (vals[var] @ target.inverse()).log()

        @jaxls.Cost.factory
        def between_cost(
            vals: jaxls.VarValues,
            var0: jaxls.SE2Var,
            var1: jaxls.SE2Var,
            delta: jaxlie.SE2,
        ) -> jax.Array:
            return ((vals[var0].inverse() @ vals[var1]) @ delta.inverse()).log()

        vars = [jaxls.SE2Var(i) for i in range(3)]

        # Parameterize the measurements
        measurement_offset = jnp.array([0.1, 0.0, 0.0])  # x, y, theta offset

        def loss_fn(offset):
            # Create noisy measurements that depend on offset
            delta_01 = jaxlie.SE2.from_xy_theta(1.0 + offset[0], offset[1], offset[2])
            delta_12 = jaxlie.SE2.from_xy_theta(1.0 + offset[0], offset[1], offset[2])

            costs = [
                prior_cost(vars[0], jaxlie.SE2.identity()),
                between_cost(vars[0], vars[1], delta_01),
                between_cost(vars[1], vars[2], delta_12),
            ]
            problem = jaxls.LeastSquaresProblem(costs, vars).analyze()
            solution = problem.solve_differentiable(
                linear_solver="dense_cholesky",
                verbose=False,
            )

            # Loss: how far the final pose is from expected
            final_pose = solution[vars[2]]
            # Expected: approximately (2 + 2*offset[0], 0, 0)
            return jnp.sum(final_pose.translation() ** 2)

        # Test that gradients can be computed
        grad = jax.grad(loss_fn)(measurement_offset)

        # Finite difference check
        eps = 1e-5
        grad_fd = jnp.zeros_like(measurement_offset)
        for i in range(len(measurement_offset)):
            p_plus = measurement_offset.at[i].add(eps)
            p_minus = measurement_offset.at[i].add(-eps)
            grad_fd = grad_fd.at[i].set(
                (loss_fn(p_plus) - loss_fn(p_minus)) / (2 * eps)
            )

        # Gradients should be close - some tolerance needed for numerical precision
        assert jnp.allclose(grad, grad_fd, rtol=0.05, atol=0.05)

    def test_jit_differentiable_solve(self):
        """Test that differentiable solve works under JIT."""

        class ScalarVar(
            jaxls.Var[jax.Array],
            default_factory=lambda: jnp.zeros(3),
        ):
            pass

        @jaxls.Cost.factory
        def prior_cost(
            vals: jaxls.VarValues, var: ScalarVar, target: jax.Array
        ) -> jax.Array:
            return vals[var] - target

        var = ScalarVar(0)

        @jax.jit
        def loss_and_grad(target):
            def loss_fn(t):
                costs = [prior_cost(var, t)]
                problem = jaxls.LeastSquaresProblem(costs, [var]).analyze()
                solution = problem.solve_differentiable(
                    linear_solver="dense_cholesky",
                    verbose=False,
                )
                return jnp.sum(solution[var] ** 2)

            return loss_fn(target), jax.grad(loss_fn)(target)

        target = jnp.array([1.0, 2.0, 3.0])
        loss, grad = loss_and_grad(target)

        assert jnp.allclose(loss, jnp.sum(target**2), atol=1e-4)
        assert jnp.allclose(grad, 2 * target, atol=1e-4)

    def test_vmap_differentiable_solve(self):
        """Test that differentiable solve works with vmap."""

        class ScalarVar(
            jaxls.Var[jax.Array],
            default_factory=lambda: jnp.zeros(2),
        ):
            pass

        @jaxls.Cost.factory
        def prior_cost(
            vals: jaxls.VarValues, var: ScalarVar, target: jax.Array
        ) -> jax.Array:
            return vals[var] - target

        var = ScalarVar(0)

        def single_loss_and_grad(target):
            def loss_fn(t):
                costs = [prior_cost(var, t)]
                problem = jaxls.LeastSquaresProblem(costs, [var]).analyze()
                solution = problem.solve_differentiable(
                    linear_solver="dense_cholesky",
                    verbose=False,
                )
                return jnp.sum(solution[var] ** 2)

            return loss_fn(target), jax.grad(loss_fn)(target)

        # Batch of targets
        targets = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # vmap over the batch dimension
        batched_loss_and_grad = jax.vmap(single_loss_and_grad)
        losses, grads = batched_loss_and_grad(targets)

        # Check results
        expected_losses = jnp.sum(targets**2, axis=1)
        expected_grads = 2 * targets

        assert jnp.allclose(losses, expected_losses, atol=1e-4)
        assert jnp.allclose(grads, expected_grads, atol=1e-4)

    def test_multiple_linear_solvers(self):
        """Test differentiable solve with different linear solvers."""

        class ScalarVar(
            jaxls.Var[jax.Array],
            default_factory=lambda: jnp.zeros(3),
        ):
            pass

        @jaxls.Cost.factory
        def prior_cost(
            vals: jaxls.VarValues, var: ScalarVar, target: jax.Array
        ) -> jax.Array:
            return vals[var] - target

        var = ScalarVar(0)
        target = jnp.array([1.0, 2.0, 3.0])

        def loss_fn(t, linear_solver):
            costs = [prior_cost(var, t)]
            problem = jaxls.LeastSquaresProblem(costs, [var]).analyze()
            solution = problem.solve_differentiable(
                linear_solver=linear_solver,
                verbose=False,
            )
            return jnp.sum(solution[var] ** 2)

        # Test with conjugate gradient
        grad_cg = jax.grad(lambda t: loss_fn(t, "conjugate_gradient"))(target)

        # Test with dense cholesky
        grad_dense = jax.grad(lambda t: loss_fn(t, "dense_cholesky"))(target)

        # Both should give same gradient
        expected_grad = 2 * target
        assert jnp.allclose(grad_cg, expected_grad, atol=1e-4)
        assert jnp.allclose(grad_dense, expected_grad, atol=1e-4)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_constraints_not_supported(self):
        """Test that constraints raise an error for differentiable solve."""

        class ScalarVar(
            jaxls.Var[jax.Array],
            default_factory=lambda: jnp.zeros(2),
        ):
            pass

        @jaxls.Cost.factory(kind="constraint_eq_zero")
        def constraint(vals: jaxls.VarValues, var: ScalarVar) -> jax.Array:
            return vals[var][0] - 1.0  # x[0] = 1

        var = ScalarVar(0)
        costs = [constraint(var)]
        problem = jaxls.LeastSquaresProblem(costs, [var]).analyze()

        with pytest.raises(ValueError, match="[Cc]onstraint"):
            problem.solve_differentiable(verbose=False)

    def test_cholmod_not_supported_in_backward(self):
        """Test that CHOLMOD linear solver warns or errors appropriately."""

        class ScalarVar(
            jaxls.Var[jax.Array],
            default_factory=lambda: jnp.zeros(2),
        ):
            pass

        @jaxls.Cost.factory
        def prior_cost(
            vals: jaxls.VarValues, var: ScalarVar, target: jax.Array
        ) -> jax.Array:
            return vals[var] - target

        var = ScalarVar(0)
        target = jnp.array([1.0, 2.0])

        def loss_fn(t):
            costs = [prior_cost(var, t)]
            problem = jaxls.LeastSquaresProblem(costs, [var]).analyze()
            solution = problem.solve_differentiable(
                linear_solver="cholmod",
                verbose=False,
            )
            return jnp.sum(solution[var] ** 2)

        # CHOLMOD should either work (with pure_callback) or raise an error
        # For now we expect it to raise an error since pure_callback VJP is complex
        with pytest.raises((ValueError, NotImplementedError)):
            jax.grad(loss_fn)(target)
