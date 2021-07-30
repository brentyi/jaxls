# jaxfg

**`jaxfg`** is a factor graph-based nonlinear least squares library for JAX.
Typical applications include sensor fusion, SLAM, bundle adjustment, optimal
control.

The premise: we provide a high-level interface for defining probability
densities as factor graphs. MAP inference reduces to nonlinear optimization,
which we accelerate by analyzing the structure of the graph. Repeated factor and
variable types have operations vectorized, and the sparsity of graph connections
is leveraged for sparse matrix operations.

Features:

- Autodiff-powered sparse Jacobians.
- Automatic vectorization for repeated factor and variable types.
- Manifold definition interface, with implementations provided for SO(2), SE(2),
  SO(3), and SE(3) Lie groups.
- Support for standard JAX function transformations: `jit`, `vmap`, `pmap`,
  `grad`, etc.
- Nonlinear optimizers: Gauss-Newton, Levenberg-Marquardt, Dogleg.
- Sparse linear solvers: conjugate gradient (Jacobi-preconditioned), sparse
  cholesky (via CHOLMOD).

Borrows heavily from a wide set of existing libraries, including:
[GTSAM](https://gtsam.org/), [Ceres Solver](http://ceres-solver.org/),
[minisam](https://github.com/dongjing3309/minisam),
[SwiftFusion](https://github.com/borglab/SwiftFusion),
[g2o](https://github.com/RainerKuemmerle/g2o). For additional technical
background, we defer to [GTSAM](https://gtsam.org/tutorials/intro.html).

### Installation

`scikit-sparse` require SuiteSparse:

```bash
sudo apt update
sudo apt install -y libsuitesparse-dev
```

Then, from your environment of choice:

```bash
git clone https://github.com/brentyi/jaxfg.git
cd jaxfg
pip install -e .
```

### Example scripts

Toy pose graph optimization:

```bash
python scripts/pose_graph_simple.py
```

Pose graph optimization from `.g2o` files:

```bash
python scripts/pose_graph_g2o.py  # For options, pass in a --help flag
```

![](./scripts/data/optimized_sphere2500.png)

### Engineering notes

We currently take a "make everything a dataclass" philosophy for software
engineering in this library. This is convenient for several reasons, but notably
makes it easy for objects to be registered as pytree nodes in JAX. See
[`jax_dataclasses`](https://github.com/brentyi/jax_dataclasses) for details on
this.

In XLA, JIT compilation needs to happen for each unique set of input shapes.
Modifying graph structures can thus introduce significant re-compilation
overheads. This is a core limitation, that restricts dynamic and online
applications.

### To-do

This library's still in development mode! Here's our TODO list:

- [x] Preliminary graph, variable, factor interfaces
- [x] Real vector variable types
- [x] Refactor into package
- [x] Nonlinear optimization for MAP inference
  - [x] Conjugate gradient linear solver
  - [x] CHOLMOD linear solver
    - [x] Basic implementation. JIT-able, but no vmap, pmap, or autodiff
          support.
    - [ ] Custom VJP rule? vmap support?
  - [x] Gauss-Newton implementation
  - [x] Termination criteria
  - [x] Damped least squares
  - [x] Dogleg
  - [x] Inexact Newton steps
  - [x] Revisit termination criteria
  - [x] Reduce redundant code
  - [ ] Robust losses
- [x] Marginalization
  - [x] Prototype using sksparse/CHOLMOD (works but fairly slow)
  - [ ] JAX implementation?
- [x] Validate g2o example
- [x] Performance
  - [x] More intentional JIT compilation
  - [x] Re-implement parallel factor computation
  - [x] Vectorized linearization
  - [x] Basic (Jacobi) CGLS preconditioning
- [x] Manifold optimization (mostly offloaded to
      [jaxlie](https://github.com/brentyi/jaxlie))
  - [x] Basic interface
  - [x] Manifold optimization on SO2
  - [x] Manifold optimization on SE2
  - [x] Manifold optimization on SO3
  - [x] Manifold optimization on SE3
- [ ] Usability + code health (low priority)
  - [x] Basic cleanup/refactor
    - [x] Better parallel factor interface
    - [x] Separate out utils, lie group helpers
    - [x] Put things in folders
  - [x] Resolve typing errors
  - [x] Cleanup/refactor (more)
  - [x] Package cleanup: dependencies, etc
  - [x] Add CI:
    - [x] mypy
    - [x] lint
    - [x] build
    - [ ] coverage
  - [ ] More comprehensive tests
  - [ ] Clean up docstrings
  - [ ] Come up with a better name than "jaxfg"
