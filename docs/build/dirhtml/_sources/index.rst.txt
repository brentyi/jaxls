jaxls
=====

|pyright|

jaxls is a solver for sparse, constrained, and/or non-Euclidean least squares
problems in JAX.

We provide an API for declaring least squares problems, which are then analyzed
for fast solve times: we automatically vectorize repeated cost and variable
operations, while translating sparse cost/variable relationships into sparse
matrix operations.

To install (Python >=3.10 minimum, >=3.12 recommended):

.. code-block:: bash

   # Core package only
   pip install "git+https://github.com/brentyi/jaxls.git"

   # Or, with dev dependencies
   pip install "git+https://github.com/brentyi/jaxls.git#egg=jaxls[dev,docs]"

Goals
-----

jaxls is designed to be:

- Lightweight and hackable, but fast. It aims to be practical for common
  scientific computing problems.
- Python-native. jaxls combines recent Python typing constructs with
  a functional, `PyTree <https://docs.jax.dev/en/latest/pytrees.html>`_-first
  implementation. Its API is type-safe, compatible with standard JAX
  `function transforms <https://docs.jax.dev/en/latest/key-concepts.html#transformations>`_,
  and more concise than traditional optimization tools.


Features
--------

We currently support:

- Automatic sparse Jacobians, defined analytically or via autodiff.
  See :doc:`guide/advanced/custom_jacobians` and :doc:`design/sparse_matrices`.
- Optimization on manifolds, including SO(2), SO(3), SE(2), SE(3).
  See :doc:`guide/advanced/non_euclidean`.
- Nonlinear solvers: Levenberg-Marquardt and Gauss-Newton.
- Linear subproblem solvers. See :doc:`guide/tips_and_gotchas` for selection guidance.

  - Sparse iterative with Conjugate Gradient (recommended for most problems).

    - Block and point Jacobi preconditioning; inexact Newton via Eisenstat-Walker.

  - Dense Cholesky (fast for small problems).
  - Sparse Cholesky on CPU (CHOLMOD).

- Augmented Lagrangian solver for constrained problems.
  See :doc:`guide/advanced/constraints`.

  - Automatic adaptive penalties for equalities and inequalities: ``h(x) = 0``, ``g(x) <= 0``.


Related projects
----------------

- `jaxlie <https://github.com/brentyi/jaxlie>`_: Lie groups in JAX, used by
  jaxls's built-in Lie group variable types.
- `PyRoki <https://github.com/chungmin99/pyroki>`_: kinematic optimization
  library for robots, built on jaxls.


Acknowledgements
----------------

jaxls is inspired by libraries like
`GTSAM <https://gtsam.org/>`_,
`Ceres Solver <http://ceres-solver.org/>`_,
`minisam <https://github.com/dongjing3309/minisam>`_,
`SwiftFusion <https://github.com/borglab/SwiftFusion>`_,
and `g2o <https://github.com/RainerKuemmerle/g2o>`_.

Algorithmic references:

- `Eisenstat & Walker (1996) <https://doi.org/10.1137/0917003>`_: adaptive inexact Newton tolerances for conjugate gradient.
- `Birgin & Martínez (2014) <https://doi.org/10.1137/1.9781611973365>`_: ALGENCAN-style augmented Lagrangian method.


.. toctree::
   :caption: Guide
   :hidden:
   :maxdepth: 1

   guide/basics
   guide/tips_and_gotchas
   guide/advanced/index

.. toctree::
   :caption: Examples
   :hidden:
   :maxdepth: 2

   examples/mechanics/index
   examples/vision/index
   examples/robotics/index
   examples/portfolio/index

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 1

   api/core
   api/lie_group_variables
   api/solver_config

.. toctree::
   :caption: Design Notes
   :hidden:
   :maxdepth: 1

   design/typed_api
   design/sparse_matrices
   design/traced_vs_static

.. |pyright| image:: https://github.com/brentyi/jaxls/actions/workflows/pyright.yml/badge.svg
   :alt: Pyright status
   :target: https://github.com/brentyi/jaxls/actions/workflows/pyright.yml
