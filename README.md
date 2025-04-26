# jaxls

[![pyright](https://github.com/brentyi/jaxls/actions/workflows/pyright.yml/badge.svg)](https://github.com/brentyi/jaxls/actions/workflows/pyright.yml)

**`jaxls`** is a library for solving sparse [NLLS](https://en.wikipedia.org/wiki/Non-linear_least_squares) and [IRLS](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares) problems in JAX.
These are common in classical robotics and computer vision.

To install on Python 3.12+:

```bash
pip install "git+https://github.com/brentyi/jaxls.git[examples]"
```

### Overviews

We provide a factor graph interface for specifying and solving least squares
problems. **`jaxls`** takes advantage of structure in graphs: repeated cost and
variable types are vectorized, and sparsity of adjacency is translated into
sparse matrix operations.

Supported:

- Automatic sparse Jacobians.
- Optimization on manifolds.
  - Examples provided for SO(2), SO(3), SE(2), and SE(3).
- Nonlinear solvers: Levenberg-Marquardt and Gauss-Newton.
- Linear subproblem solvers:
  - Sparse iterative with Conjugate Gradient.
    - Preconditioning: block and point Jacobi.
    - Inexact Newton via Eisenstat-Walker.
    - Recommended for most problems.
  - Dense Cholesky.
    - Fast for small problems.
  - Sparse Cholesky, on CPU. (CHOLMOD)

`jaxls` borrows heavily from libraries like
[GTSAM](https://gtsam.org/), [Ceres Solver](http://ceres-solver.org/),
[minisam](https://github.com/dongjing3309/minisam),
[SwiftFusion](https://github.com/borglab/SwiftFusion),
and [g2o](https://github.com/RainerKuemmerle/g2o).

### Pose graph example

```python
import jaxls
import jaxlie
```

**Defining variables.** Each variable is given an integer ID. They don't need to
be contiguous.

```
pose_vars = [jaxls.SE2Var(0), jaxls.SE2Var(1)]
```

**Defining costs (decorator).** The recommended way to define a cost is to
transform a function into a cost factory using the `@jaxls.Cost.create_factory`
decorator. The transformed function will return a `jaxls.Cost` object.

```python
# Defining cost types.

@jaxls.Cost.create_factory
def prior_cost(
    vals: jaxls.VarValues, var: jaxls.SE2Var, target: jaxlie.SE2
) -> jax.Array:
    """Prior cost for a pose variable. Penalizes deviations from the target"""
    return (vals[var] @ target.inverse()).log()

@jaxls.Cost.create_factory
def between_cost(
    vals: jaxls.VarValues, delta: jaxlie.SE2, var0: jaxls.SE2Var, var1: jaxls.SE2Var
) -> jax.Array:
    """'Between' cost for two pose variables. Penalizes deviations from the delta."""
    return ((vals[var0].inverse() @ vals[var1]) @ delta.inverse()).log()
```

```python
# Instantiating costs.
costs = [
    prior_cost(vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
    prior_cost(vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
    between_cost(jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0), vars[0], vars[1]),
]
```

**Defining costs (directly).** Costs can also be instantiated directly using a
callable cost function and a set of arguments. Under-the-hood, the `create_factory`
decorator constructs objects that look like this:

```python
# Costs take two main arguments:
# - A callable with signature `(jaxls.VarValues, *Args) -> jax.Array`.
# - A tuple of arguments: the type should be `tuple[*Args]`.
#
# All arguments should be PyTree structures. Variable types within the PyTree
# will be automatically detected.
costs = [
    # Cost on pose 0.
    jaxls.Cost(
        lambda vals, var, init: (vals[var] @ init.inverse()).log(),
        (pose_vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
    ),
    # Cost on pose 1.
    jaxls.Cost(
        lambda vals, var, init: (vals[var] @ init.inverse()).log(),
        (pose_vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
    ),
    # Cost between poses.
    jaxls.Cost(
        lambda vals, var0, var1, delta: (
            (vals[var0].inverse() @ vals[var1]) @ delta.inverse()
        ).log(),
        (pose_vars[0], pose_vars[1], jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0)),
    ),
]
```

Costs with similar structure, like the first two in this example, will be
vectorized under-the-hood.

Batched inputs can also be manually constructed, and are detected by inspecting
the shape of variable ID arrays in the input.

**Solving optimization problems.** To create the optimization problem, analyze
it, and solve:

```python
problem = jaxls.LeastSquaresProblem(costs, pose_vars).analyze()
solution = problem.solve()
print("All solutions", solution)
print("Pose 0", solution[pose_vars[0]])
print("Pose 1", solution[pose_vars[1]])
```

### CHOLMOD setup

By default, we use an iterative linear solver. This requires no extra
dependencies. For problems with strong supernodal structure or where our
preconditioners are ineffective, a direct solver can be much faster.

For Cholesky factorization via CHOLMOD, we rely on SuiteSparse:

```bash
# Option 1: via conda.
conda install conda-forge::suitesparse

# Option 2: via apt.
sudo apt install -y libsuitesparse-dev
```

You'll also need _scikit-sparse_:

```bash
pip install scikit-sparse
```
