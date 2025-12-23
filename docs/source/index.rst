jaxls
=====

|pyright|

jaxls is a library for solving sparse, constrained, and/or non-Euclidean
least squares problems in JAX.

These problems are common in robotics and computer vision for applications like
pose graph optimization, bundle adjustment, SLAM, camera calibration, inverse
kinematics, motion planning, and motion retargeting.

Install with:

.. code-block:: bash

   pip install "git+https://github.com/brentyi/jaxls.git"

   # Or, with examples:
   pip install "git+https://github.com/brentyi/jaxls.git#egg=jaxls[examples]"


Features
--------

We provide a factor graph interface for solving nonlinear least squares
problems. jaxls is designed to exploit structure in graphs: repeated cost
and variable types are vectorized, and sparsity of adjacency is translated into
sparse matrix operations.

Features include:

- Automatic sparse Jacobians, defined analytically or via autodiff.
- Optimization on manifolds (SO(2), SO(3), SE(2), SE(3)).
- Nonlinear solvers: Levenberg-Marquardt and Gauss-Newton.
- Linear subproblem solvers:

  - Sparse iterative with Conjugate Gradient (recommended for most problems).

    - Block and point Jacobi preconditioning; inexact Newton via Eisenstat-Walker.

  - Dense Cholesky (fast for small problems).
  - Sparse Cholesky on CPU (CHOLMOD).

- Augmented Lagrangian solver for constrained problems.

  - Automatic adaptive penalties for equalities and inequalities: ``h(x) = 0``, ``g(x) <= 0``.


.. toctree::
   :caption: Guide
   :hidden:
   :maxdepth: 1

   examples/guide/basics
   examples/guide/tips_and_gotchas
   examples/guide/advanced/index

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
   :caption: Other
   :hidden:
   :maxdepth: 1

   acknowledgements


.. |pyright| image:: https://github.com/brentyi/jaxls/actions/workflows/pyright.yml/badge.svg
   :alt: Pyright status
   :target: https://github.com/brentyi/jaxls/actions/workflows/pyright.yml
