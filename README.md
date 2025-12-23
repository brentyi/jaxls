# jaxls

[![pyright](https://github.com/brentyi/jaxls/actions/workflows/pyright.yml/badge.svg)](https://github.com/brentyi/jaxls/actions/workflows/pyright.yml)
[![docs](https://github.com/brentyi/jaxls/actions/workflows/docs.yml/badge.svg)](https://brentyi.github.io/jaxls/)

**`jaxls`** is a library for solving sparse, constrained, and/or non-Euclidean
least squares problems in JAX.

These problems are common in robotics and computer vision for applications like
pose graph optimization, bundle adjustment, SLAM, camera calibration, inverse
kinematics, motion planning, and motion retargeting.

**[Documentation](https://brentyi.github.io/jaxls/)**

### Installation

```bash
pip install "git+https://github.com/brentyi/jaxls.git"
```

### Quick example

```python
import jax
import jaxlie
import jaxls

# Define a cost function.
@jaxls.Cost.factory
def prior_cost(
    vals: jaxls.VarValues, var: jaxls.SE2Var, target: jaxlie.SE2
) -> jax.Array:
    return (vals[var] @ target.inverse()).log()

# Create variables and costs.
vars = [jaxls.SE2Var(0), jaxls.SE2Var(1)]
costs = [
    prior_cost(vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
    prior_cost(vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
]

# Solve.
solution = jaxls.LeastSquaresProblem(costs, vars).analyze().solve()
```

### In-the-wild examples

<table>
  <tr>
    <td>
      <a href="https://egoallo.github.io/">
        egoallo
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/brentyi/egoallo?style=social"
        />
      </a>
    </td>
    <td>
      For guiding diffusion model sampling with 2D observations / reprojection errors.
    </td>
  </tr>
  <tr>
    <td>
      <a href="https://videomimic.net/">
        videomimic
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/hongsukchoi/videomimic?style=social"
        />
      </a>
    </td>
    <td>
      For joint human-scene optimization and motion retargeting.
    </td>
  </tr>
  <tr>
    <td>
      <a href="https://pyroki-toolkit.github.io/">
        pyroki
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/chungmin99/pyroki?style=social"
        />
      </a>
    </td>
    <td>
      For solving inverse kinematics, motion planning, etc for robots.
    </td>
  </tr>
  <tr>
    <td>
      <a href="https://protomotions.github.io/tutorials/workflows/retargeting_pyroki.html">
        ProtoMotions
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/nvlabs/ProtoMotions?style=social"
        />
      </a>
    </td>
    <td>
      For trajectory-level humanoid motion retargeting (via PyRoki).
    </td>
  </tr>
</table>
