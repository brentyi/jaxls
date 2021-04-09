# jaxfg

Factor graph-based nonlinear optimization library for JAX.

Applications include sensor fusion, control, planning, SLAM. _Heavily_ inspired
by a wide set of existing libraries, including:
[Ceres Solver](http://ceres-solver.org/),
[g2o](https://github.com/RainerKuemmerle/g2o), [GTSAM](https://gtsam.org/),
[minisam](https://github.com/dongjing3309/minisam),
[SwiftFusion](https://github.com/borglab/SwiftFusion).

Features:

- Autodiff-powered (sparse) Jacobians.
- Automatic batching of factor computations.
- Out-of-the-box support for optimization on SO(2), SO(3), SE(2), and SE(3).
- 100% implemented in Python!

Current limitations:

- Linear solves are restricted to a preconditioned conjugate gradient approach.
  Much slower than direct methods (eg sparse Cholesky) when problems are
  ill-conditioned.
- JIT compilation adds significant startup overhead. This could likely be
  optimized (for example, by specifying more analytical Jacobians) but is mostly
  unavoidable with JAX/XLA. Currently limits online applications.
- Python >=3.7 only, due to features needed for generic types.

---

### Example scripts

Toy pose graph optimization:

```
python scripts/pose_graph_simple.py
```

Pose graph optimization from `.g2o` files:

```bash
python scripts/pose_graph_g2o.py --help
```

---

### To-do

- [x] Preliminary graph, variable, factor interfaces
- [x] Real vector variable types
- [x] Refactor into package
- [x] Non-linear optimization
  - [x] Very basic Gauss-Newton implementation
  - [x] Termination criteria
  - [x] Damped least squares
  - [x] Inexact Newton steps
  - [x] Revisit termination criteria
  - [ ] Reduce redundant code
  - [ ] Robust losses
- [x] MAP inference
- [ ] Marginalization
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
