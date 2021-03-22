# jaxfg

Factor graph-based nonlinear optimization library for JAX.

Applications include sensor fusion, control, planning, SLAM. _Heavily_ inspired
by a wide set of existing libraries, including:
[Ceres Solver](http://ceres-solver.org/),
[g2o](https://github.com/RainerKuemmerle/g2o), [GTSAM](https://gtsam.org/),
[minisam](https://github.com/dongjing3309/minisam), and
[SwiftFusion](https://github.com/borglab/SwiftFusion).

Features:

- Autodiff-powered Jacobians.
- Automatic batching of factor computations.
- Out-of-the-box support for optimization on SO(2), SO(3), SE(2), and SE(3).
- 100% implemented in Python!

Areas that could use improvement:

- Linear solves are restricted to a preconditioned conjugate gradient approach.
  Can be much slower than direct methods when problems are ill-conditioned.
- JIT compilation adds startup overhead. This is mostly unavoidable with
  JAX/XLA.
- Support for robust losses.

---

### Example scripts

Toy pose graph optimization:

```
scripts/pose_graph_simple.py
```

Pose graph optimization from `.g2o` files:

```bash
scripts/pose_graph_g2o.py --help
```

---

### To-do

- [x] Preliminary graph, variable, factor interfaces
- [x] Real vector variable types
- [x] Refactor into package
- [x] Linear factor graph
- [x] Non-linear factor graph
  - [x] Very basic Gauss-Newton implementation
  - [x] Termination criteria
  - [x] Damped least squares
  - [x] Inexact Newton steps
  - [x] Revisit termination criteria
  - [ ] Reduce redundant code
- [x] MAP inference
- [x] Compare g2o example
  - [x] Validate against minisam
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
  - [ ] Cleanup/refactor (more)
  - [ ] Package cleanup: dependencies, etc
  - [ ] Add CI:
    - [x] mypy
    - [x] lint
    - [ ] build
  - [ ] Tests
