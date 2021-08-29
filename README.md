# jaxfg

[![build](https://github.com/brentyi/jaxfg/actions/workflows/build.yml/badge.svg)](https://github.com/brentyi/jaxfg/actions/workflows/build.yml)
[![lint](https://github.com/brentyi/jaxfg/actions/workflows/lint.yml/badge.svg)](https://github.com/brentyi/jaxfg/actions/workflows/lint.yml)
[![mypy](https://github.com/brentyi/jaxfg/actions/workflows/mypy.yml/badge.svg)](https://github.com/brentyi/jaxfg/actions/workflows/mypy.yml)
[![codecov](https://codecov.io/gh/brentyi/jaxfg/branch/master/graph/badge.svg?token=RNJB7EFC8T)](https://codecov.io/gh/brentyi/jaxfg)

<!-- vim-markdown-toc GFM -->

* [Installation](#installation)
* [Example scripts](#example-scripts)
* [Development](#development)
* [Current limitations](#current-limitations)
* [To-do](#to-do)

<!-- vim-markdown-toc -->

**`jaxfg`** is a factor graph-based nonlinear least squares library for JAX.
Typical applications include sensor fusion, SLAM, bundle adjustment, optimal
control.

The premise: we provide a high-level interface for defining probability
densities as factor graphs. MAP inference reduces to nonlinear optimization,
which we accelerate by analyzing the structure of the graph. Repeated factor and
variable types have operations vectorized, and the sparsity of graph connections
is translated into sparse matrix operations.

Features:

- Autodiff-powered sparse Jacobians.
- Automatic vectorization for repeated factor and variable types.
- Manifold definition interface, with implementations provided for SO(2), SE(2),
  SO(3), and SE(3) Lie groups.
- Support for standard JAX function transformations: `jit`, `vmap`, `pmap`,
  `grad`, etc.
- Nonlinear optimizers: Gauss-Newton, Levenberg-Marquardt, Dogleg.
- Sparse linear solvers: conjugate gradient (Jacobi-preconditioned), sparse
  Cholesky (via CHOLMOD).

This library is released as part of our IROS 2021 paper (more info in our core
experiment repository [here](https://github.com/brentyi/dfgo)) and borrows
heavily from a wide set of existing libraries, including
[GTSAM](https://gtsam.org/), [Ceres Solver](http://ceres-solver.org/),
[minisam](https://github.com/dongjing3309/minisam),
[SwiftFusion](https://github.com/borglab/SwiftFusion), and
[g2o](https://github.com/RainerKuemmerle/g2o). For technical background and
concepts, GTSAM has a
[great set of tutorials](https://gtsam.org/tutorials/intro.html).

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

### Development

If you're interested in extending this library to define your own factor graphs,
we'd recommend first familiarizing yourself with:

1. Pytrees in JAX:
   https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html
2. Python dataclasses: https://docs.python.org/3/library/dataclasses.html
   - We currently take a "make everything a dataclass" philosophy for software
     engineering in this library. This is convenient for several reasons, but
     notably makes it easy for objects to be registered as pytree nodes. See
     [`jax_dataclasses`](https://github.com/brentyi/jax_dataclasses) for details
     on this.
3. Type annotations: https://docs.python.org/3/library/typing.html
   - We rely on generics (`typing.Generic` and `typing.TypeVar`) particularly
     heavily. If you're familiar with C++ this should come very naturally
     (~templates).
4. Explicit decorators for overrides/inheritance:
   https://github.com/mkorpela/overrides
   - The `@overrides` and `@final` decorators signal which methods are being
     and/or shouldn't be overridden. The same goes for `@abc.abstractmethod`.

From there, we have a few references for defining your own factor graphs,
factors, and manifolds:

- The [simple pose graph script](./scripts/pose_graph_simple.py) includes the
  basics of setting up and solving a factor graph.
- Our [Lie group variable definitions](./jaxfg/geometry/_lie_variables.py) can
  serve as a template for defining your own variable types and manifolds. The
  [base class](./jaxfg/core/_variables.py) also has comments on what needs to be
  overridden.
- Our
  [PriorFactor and BetweenFactor implementations](./jaxfg/geometry/_factors.py)
  can serve as a template for defining your own factor types. The
  [base class](./jaxfg/core/_factor_base.py) also has comments on what needs to
  be overridden.

### Current limitations

1. In XLA, JIT compilation needs to happen for each unique set of input shapes.
   Modifying graph structures can thus introduce significant re-compilation
   overheads; this can restrict applications that are dynamic or online.
2. Our marginalization implementation is not very good.

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
  - [ ] New name
