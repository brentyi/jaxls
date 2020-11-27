# jaxfg

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
- [ ] Manifold optimization
  - [x] Basic interface
  - [x] Manifold optimization on SO2
  - [x] Manifold optimization on SE2
  - [ ] Manifold optimization on SO3
  - [ ] Manifold optimization on SE3
- [ ] Code health (low priority unti
  - [ ] Cleanup/refactor
    - [ ] Do it
    - [ ] Do it again
  - [ ] Package cleanup: dependencies, etc
  - [ ] Add CI: mypy, flake8, pytest, etc.
  - [ ] Tests
