import jax
import jaxlie
import jaxls


def test_pose_graph_decorator_syntax():
    """Test pose graph optimization using decorator-based cost syntax."""
    vars = (jaxls.SE2Var(0), jaxls.SE2Var(1))

    @jaxls.Cost.factory
    def prior_cost(
        vals: jaxls.VarValues, var: jaxls.SE2Var, target: jaxlie.SE2
    ) -> jax.Array:
        """Prior cost for a pose variable. Penalizes deviations from the target"""
        return (vals[var] @ target.inverse()).log()

    @jaxls.Cost.factory
    def between_cost(
        vals: jaxls.VarValues, delta: jaxlie.SE2, var0: jaxls.SE2Var, var1: jaxls.SE2Var
    ) -> jax.Array:
        """'Between' cost for two pose variables. Penalizes deviations from the delta."""
        return ((vals[var0].inverse() @ vals[var1]) @ delta.inverse()).log()

    costs = [
        prior_cost(vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
        prior_cost(vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
        between_cost(jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0), vars[0], vars[1]),
    ]

    problem = jaxls.LeastSquaresProblem(costs, vars).analyze()
    solution = problem.solve()

    # The optimal solution should place pose 0 at approximately (1/3, 0, 0)
    # and pose 1 at approximately (5/3, 0, 0) to balance all costs
    pose0 = solution[vars[0]]
    pose1 = solution[vars[1]]

    # Check that poses are roughly correct
    assert abs(pose0.translation()[0] - 1 / 3) < 0.01
    assert abs(pose0.translation()[1]) < 0.01
    assert abs(pose0.rotation().as_radians()) < 0.01

    assert abs(pose1.translation()[0] - 5 / 3) < 0.01
    assert abs(pose1.translation()[1]) < 0.01
    assert abs(pose1.rotation().as_radians()) < 0.01


def test_pose_graph_direct_syntax():
    """Test pose graph optimization using direct cost syntax."""
    vars = (jaxls.SE2Var(0), jaxls.SE2Var(1))

    costs = [
        # Prior cost for pose 0.
        jaxls.Cost(
            lambda vals, var, init: (vals[var] @ init.inverse()).log(),
            (vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
        ),
        # Prior cost for pose 1.
        jaxls.Cost(
            lambda vals, var, init: (vals[var] @ init.inverse()).log(),
            (vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
        ),
        # "Between" cost.
        jaxls.Cost(
            lambda vals, delta, var0, var1: (
                (vals[var0].inverse() @ vals[var1]) @ delta.inverse()
            ).log(),
            (jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0), vars[0], vars[1]),
        ),
    ]

    problem = jaxls.LeastSquaresProblem(costs, vars).analyze()
    solution = problem.solve()

    # The optimal solution should place pose 0 at approximately (1/3, 0, 0)
    # and pose 1 at approximately (5/3, 0, 0) to balance all costs
    pose0 = solution[vars[0]]
    pose1 = solution[vars[1]]

    # Check that poses are roughly correct
    assert abs(pose0.translation()[0] - 1 / 3) < 0.01
    assert abs(pose0.translation()[1]) < 0.01
    assert abs(pose0.rotation().as_radians()) < 0.01

    assert abs(pose1.translation()[0] - 5 / 3) < 0.01
    assert abs(pose1.translation()[1]) < 0.01
    assert abs(pose1.rotation().as_radians()) < 0.01


def test_decorator_and_direct_syntax_equivalence():
    """Test that decorator and direct syntax produce the same results."""
    vars = (jaxls.SE2Var(0), jaxls.SE2Var(1))

    # Decorator syntax
    @jaxls.Cost.factory
    def prior_cost(
        vals: jaxls.VarValues, var: jaxls.SE2Var, target: jaxlie.SE2
    ) -> jax.Array:
        return (vals[var] @ target.inverse()).log()

    @jaxls.Cost.factory
    def between_cost(
        vals: jaxls.VarValues, delta: jaxlie.SE2, var0: jaxls.SE2Var, var1: jaxls.SE2Var
    ) -> jax.Array:
        return ((vals[var0].inverse() @ vals[var1]) @ delta.inverse()).log()

    costs_decorator = [
        prior_cost(vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
        prior_cost(vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
        between_cost(jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0), vars[0], vars[1]),
    ]

    # Direct syntax
    costs_direct = [
        jaxls.Cost(
            lambda vals, var, init: (vals[var] @ init.inverse()).log(),
            (vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
        ),
        jaxls.Cost(
            lambda vals, var, init: (vals[var] @ init.inverse()).log(),
            (vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
        ),
        jaxls.Cost(
            lambda vals, delta, var0, var1: (
                (vals[var0].inverse() @ vals[var1]) @ delta.inverse()
            ).log(),
            (jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0), vars[0], vars[1]),
        ),
    ]

    # Solve both problems
    solution_decorator = (
        jaxls.LeastSquaresProblem(costs_decorator, vars).analyze().solve()
    )
    solution_direct = jaxls.LeastSquaresProblem(costs_direct, vars).analyze().solve()

    # Check that solutions are approximately equal
    pose0_decorator = solution_decorator[vars[0]]
    pose0_direct = solution_direct[vars[0]]
    pose1_decorator = solution_decorator[vars[1]]
    pose1_direct = solution_direct[vars[1]]

    assert jax.numpy.allclose(
        pose0_decorator.translation(), pose0_direct.translation(), atol=1e-6
    )
    assert jax.numpy.allclose(
        pose0_decorator.rotation().as_radians(),
        pose0_direct.rotation().as_radians(),
        atol=1e-6,
    )
    assert jax.numpy.allclose(
        pose1_decorator.translation(), pose1_direct.translation(), atol=1e-6
    )
    assert jax.numpy.allclose(
        pose1_decorator.rotation().as_radians(),
        pose1_direct.rotation().as_radians(),
        atol=1e-6,
    )
