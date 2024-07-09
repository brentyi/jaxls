# jaxls

[![build](https://github.com/brentyi/jaxfg/actions/workflows/build.yml/badge.svg)](https://github.com/brentyi/jaxfg/actions/workflows/build.yml)
[![lint](https://github.com/brentyi/jaxfg/actions/workflows/lint.yml/badge.svg)](https://github.com/brentyi/jaxfg/actions/workflows/lint.yml)
[![mypy](https://github.com/brentyi/jaxfg/actions/workflows/mypy.yml/badge.svg)](https://github.com/brentyi/jaxfg/actions/workflows/mypy.yml)
[![codecov](https://codecov.io/gh/brentyi/jaxfg/branch/master/graph/badge.svg?token=RNJB7EFC8T)](https://codecov.io/gh/brentyi/jaxfg)

**`jaxls`** is a library for sparse nonlinear least squares in JAX.

We provide a factor graph interface for defining variables and costs. We
analyze the structure of the graph to automatically vectorize factor and
variable operations, and translate the sparsity of graph connections into
sparse matrix operations.

This is useful for optimization problems where gradient-based methods are slow
or ineffective. In robotics, common ones includes sensor fusion, SLAM, bundle
adjustment, optimal control, inverse kinematics, and motion planning.

Features:

- Sparse Jacobians via autodiff.
- Manifold support; SO(2), SO(3), SE(2), and SE(3) implementations included.
- Nonlinear solvers: Gauss-Newton and Levenberg-Marquardt.
- Linear solvers: sparse Cholesky (CPU only via CHOLMOD), Jacobi-preconditioned Conjugate Gradient.

For the first iteration of this library (written for [IROS 2021](https://github.com/brentyi/dfgo)), see [jaxfg](https://github.com/brentyi/jaxfg).
`jaxls` is faster and easier to use.

### Installation

`jaxls` supports `python>=3.12`.

For Cholesky factorization via CHOLMOD, `scikit-sparse` requires SuiteSparse:

```bash
# Via conda.
conda install conda-forge::suitesparse
# Via apt.
sudo apt update
sudo apt install -y libsuitesparse-dev
# Via brew.
brew install suite-sparse
```

Then, from your environment of choice:

```bash
git clone https://github.com/brentyi/jaxls.git
cd jaxls
pip install -e .
```

### Pose graph example

```python
import jaxls
import jaxlie
```

**Defining variables.** Each variable is given an integer ID. They don't need to be contiguous.

```
pose_vars = [jaxls.SE2Var(0), jaxls.SE2Var(1)]
```

**Defining factors.** Factors are defined using a callable cost function and a set of arguments.

```python
factors = [
    # Cost on pose 0.
    jaxfg2.Factor.make(
        lambda vals, var, init: (vals[var] @ init.inverse()).log(),
        (pose_vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
    ),
    # Cost on pose 1.
    jaxfg2.Factor.make(
        lambda vals, var, init: (vals[var] @ init.inverse()).log(),
        (pose_vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
    ),
    # Cost between poses.
    jaxfg2.Factor.make(
        lambda vals, var0, var1, delta: (
            (vals[var0].inverse() @ vals[var1]) @ delta.inverse()
        ).log(),
        (pose_vars[0], pose_vars[1], jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0)),
    ),
]
```

Factors with similar structure, like the first two in this example, will be vectorized under-the-hood.

**Solving optimization problems.** We can set up the optimization problem,
solve it, and print the solutions:

```python
graph = jaxfg2.FactorGraph.make(factors, pose_vars)
solution = graph.solve()
print("All solutions", solution)
print("Pose 0", solution[pose_vars[0]])
print("Pose 1", solution[pose_vars[1]])
```
