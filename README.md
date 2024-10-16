# jaxls

[![pyright](https://github.com/brentyi/jaxls/actions/workflows/pyright.yml/badge.svg)](https://github.com/brentyi/jaxls/actions/workflows/pyright.yml)

**`jaxls`** is a library for nonlinear least squares in JAX.

We provide a factor graph interface for specifying and solving least squares
problems. We accelerate optimization by analyzing the structure of graphs:
repeated factor and variable types are vectorized, and the sparsity of adjacency
in the graph is translated into sparse matrix operations.

Use cases are primarily in least squares problems that are (1) sparse and (2)
inefficient to solve with gradient-based methods.

Currently supported:

- Automatic sparse Jacobians.
- Optimization on manifolds.
  - Examples provided for SO(2), SO(3), SE(2), and SE(3).
- Nonlinear solvers: Levenberg-Marquardt and Gauss-Newton.
- Linear subproblem solvers:
  - Sparse iterative with Conjugate Gradient.
    - Preconditioning: block and point Jacobi.
    - Inexact Newton via Eisenstat-Walker.
  - Sparse direct with Cholesky / CHOLMOD, on CPU.
  - Dense Cholesky for smaller problems.

For the first iteration of this library, written for [IROS 2021](https://github.com/brentyi/dfgo), see
[jaxfg](https://github.com/brentyi/jaxfg). `jaxls` is a rewrite that is faster
and easier to use. For additional references, see inspirations like
[GTSAM](https://gtsam.org/), [Ceres Solver](http://ceres-solver.org/),
[minisam](https://github.com/dongjing3309/minisam),
[SwiftFusion](https://github.com/borglab/SwiftFusion),
[g2o](https://github.com/RainerKuemmerle/g2o).

### Installation

`jaxls` supports `python>=3.12`:

```bash
pip install git+https://github.com/brentyi/jaxls.git
```

**Optional: CHOLMOD dependencies**

By default, we use an iterative linear solver. This requires no extra
dependencies. For some problems, like those with banded matrices, a direct
solver can be much faster.

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

**Defining factors.** Factors are defined using a callable cost function and a
set of arguments.

```python
# Factors take two arguments:
# - A callable with signature `(jaxls.VarValues, *Args) -> jax.Array`.
# - A tuple of arguments: the type should be `tuple[*Args]`.
#
# All arguments should be PyTree structures. Variable types within the PyTree
# will be automatically detected.
factors = [
    # Cost on pose 0.
    jaxls.Factor(
        lambda vals, var, init: (vals[var] @ init.inverse()).log(),
        (pose_vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
    ),
    # Cost on pose 1.
    jaxls.Factor(
        lambda vals, var, init: (vals[var] @ init.inverse()).log(),
        (pose_vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
    ),
    # Cost between poses.
    jaxls.Factor(
        lambda vals, var0, var1, delta: (
            (vals[var0].inverse() @ vals[var1]) @ delta.inverse()
        ).log(),
        (pose_vars[0], pose_vars[1], jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0)),
    ),
]
```

Factors with similar structure, like the first two in this example, will be
vectorized under-the-hood.

**Solving optimization problems.** We can set up the optimization problem, solve
it, and print the solutions:

```python
graph = jaxls.FactorGraph.make(factors, pose_vars)
solution = graph.solve()
print("All solutions", solution)
print("Pose 0", solution[pose_vars[0]])
print("Pose 1", solution[pose_vars[1]])
```

### Limitations

There are many practical features that we don't currently support:

- GPU accelerated Cholesky factorization. (for CHOLMOD we wrap [scikit-sparse](https://scikit-sparse.readthedocs.io/en/latest/), which runs on CPU only)
- Covariance estimation / marginalization.
- Incremental solves.
- Analytical Jacobians.
