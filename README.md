# jaxfg

Library for solving factor graph-based least squares problems with JAX. Heavily
influenced by [minisam](https://github.com/dongjing3309/minisam),
[GTSAM](https://gtsam.org/), and [g2o](https://github.com/RainerKuemmerle/g2o).

Applications include sensor fusion, optimal control, planning, SLAM, etc.

![M3500 results](scripts/data/optimized_m3500.png)

### To-do

- [x] Preliminary graph, variable, factor interfaces
- [x] Real vector variable types
- [x] Refactor into package
- [x] Linear factor graph
- [x] Non-linear factors graph
  - [x] Very basic Gauss-Newton implementation
  - [ ] Proper termination criteria
  - [ ] Damped least squares
  - [ ] Gradslam-style damped least squares?
- [x] MAP inference
- [x] Compare g2o example
  - [x] Validate against minisam
- [x] Performance
  - [x] More intentional JIT compilation
  - [x] Re-implement parallel factor computation
  - [x] Vectorized linearization
  - [x] Basic CGLS preconditioning
  - [ ] Better preconditioning and/or debugging -- still slow
- [x] Manifold optimization
  - [x] Basic interface
  - [x] Manifold optimization on SO2
    - [ ] Analytical JVP?
    - [ ] Analytical VJP?
  - [x] Manifold optimization on SE2
    - [ ] Analytical JVP?
    - [ ] Analytical VJP?
  - [ ] Manifold optimization on SO3
    - [ ] Analytical JVP?
    - [ ] Analytical VJP?
  - [ ] Manifold optimization on SE3
    - [ ] Analytical JVP?
    - [ ] Analytical VJP?
- [ ] Code health (low priority unti
  - [ ] Cleanup/refactor
    - [ ] Do it
      - [ ] Better parallel factor interface
      - [ ] Separate out utils, lie group helpers
      - [ ] Put things in folders
    - [ ] Do it again
  - [ ] Package cleanup: dependencies, etc
  - [ ] Add CI: mypy, flake8, pytest, etc.
  - [ ] Tests
