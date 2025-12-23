Acknowledgements
================

**jaxls** borrows heavily from libraries like
`GTSAM <https://gtsam.org/>`_,
`Ceres Solver <http://ceres-solver.org/>`_,
`minisam <https://github.com/dongjing3309/minisam>`_,
`SwiftFusion <https://github.com/borglab/SwiftFusion>`_,
and `g2o <https://github.com/RainerKuemmerle/g2o>`_.

We also adapted details from these algorithmic references:

- Eisenstat & Walker, `Choosing the Forcing Terms in an Inexact Newton Method <https://doi.org/10.1137/0917003>`_, 1996.
  For adaptive inexact Newton tolerances in the conjugate gradient solver.

- Birgin & Martínez, `Practical Augmented Lagrangian Methods for Constrained Optimization <https://doi.org/10.1137/1.9781611973365>`_, 2014.
  For the ALGENCAN-style augmented Lagrangian implementation.
