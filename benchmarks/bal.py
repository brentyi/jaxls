"""Bundle Adjustment in the Large (BAL) loader and reprojection cost.

Dataset: https://grail.cs.washington.edu/projects/bal/

Camera model: angle-axis rotation + translation + focal length + two radial
distortion parameters; projection onto the negative-z image plane.
"""

from __future__ import annotations

import bz2
import functools
import inspect
import time
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp

import jaxls


def _analyze(
    problem: jaxls.LeastSquaresProblem, schur_elimination: bool
) -> jaxls.AnalyzedLeastSquaresProblem:
    """analyze() with `schur_elimination` when supported. jaxls@main predates
    elimination; passing the flag through this shim lets the same benchmark
    scripts run against both versions for before/after comparisons."""
    if "schur_elimination" in inspect.signature(problem.analyze).parameters:
        return problem.analyze(schur_elimination=schur_elimination)
    return problem.analyze()


def run_k_iterations(
    problem: jaxls.AnalyzedLeastSquaresProblem,
    initial_vals: jaxls.VarValues,
    k: int,
    *,
    linear_solver: str,
    repeats: int = 3,
    lambda_initial: float | None = None,
    warmup_budget_s: float | None = None,
) -> tuple[float, float]:
    """Run exactly k LM iterations; return (accepted cost, min wall-clock).

    The shared measurement kernel for the matched-iteration benchmarks
    (`matched_iters.py`, `device_sweep.py`). Each rule guards against a real
    measurement bug: early termination is disabled so k is exact; one full
    warmup solve with `jax.block_until_ready` absorbs compilation and
    asynchronously dispatched device work; each timed point is the min of
    `repeats`. The reported cost is recomputed from the returned (accepted)
    solution — the solve summary's cost history also contains rejected
    proposals, which would not be honest to report.

    Run on a specific device by passing `jax.device_put`-placed inputs.
    """

    # `lambda_initial`: bundle-adjustment curvature scales put workable
    # damping around 1e1-1e3 (see results.md); the library default 5e-4 is
    # tuned for ~unit-curvature problems. When set, it is applied uniformly
    # to every method under comparison.
    extra = (
        {}
        if lambda_initial is None
        else {"trust_region": jaxls.TrustRegionConfig(lambda_initial=lambda_initial)}
    )

    def solve() -> tuple[jaxls.VarValues, jaxls.SolveSummary]:
        return problem.solve(
            initial_vals,
            linear_solver=linear_solver,  # type: ignore[arg-type]
            termination=jaxls.TerminationConfig(
                max_iterations=k, early_termination=False
            ),
            verbose=False,
            return_summary=True,
            **extra,
        )

    # Warmup (absorbs compilation). If it blows past `warmup_budget_s`, this
    # config is too slow to bother timing repeats — return the warmup's own
    # (cost, time). Guards against the pathological CPU full-CG-on-large-BA
    # cases whose single solve runs for minutes.
    w_start = time.perf_counter()
    warm_sol, _ = jax.block_until_ready(solve())
    warm_t = time.perf_counter() - w_start
    if warmup_budget_s is not None and warm_t > warmup_budget_s:
        cost = float(onp.asarray(problem._compute_cost_info(warm_sol).cost_total))
        return cost, warm_t

    elapsed = []
    sol = None
    for _ in range(repeats):
        start = time.perf_counter()
        sol, _ = jax.block_until_ready(solve())
        elapsed.append(time.perf_counter() - start)
    assert sol is not None

    final_cost = float(onp.asarray(problem._compute_cost_info(sol).cost_total))
    return final_cost, min(elapsed)


class CameraVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(9)):
    """BAL camera: angle-axis (3), translation (3), focal, k1, k2."""


class PointVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(3)):
    """3D landmark."""


def rodrigues(aa: jax.Array, x: jax.Array) -> jax.Array:
    """Rotate `x` by the angle-axis vector `aa`."""
    theta = jnp.linalg.norm(aa) + 1e-12
    k = aa / theta
    return (
        x * jnp.cos(theta)
        + jnp.cross(k, x) * jnp.sin(theta)
        + k * jnp.dot(k, x) * (1.0 - jnp.cos(theta))
    )


def project(camera: jax.Array, point: jax.Array) -> jax.Array:
    """BAL projection: world point -> 2D pixel coordinates."""
    p = rodrigues(camera[:3], point) + camera[3:6]
    proj = -p[:2] / p[2]
    f, k1, k2 = camera[6], camera[7], camera[8]
    r2 = jnp.sum(proj**2)
    return f * (1.0 + k1 * r2 + k2 * r2**2) * proj


def reprojection_cost(
    vals: jaxls.VarValues,
    camera: CameraVar,
    point: PointVar,
    observation: jax.Array,
) -> jax.Array:
    return project(vals[camera], vals[point]) - observation


def download_bal(name: str, dest: Path) -> Path:
    """Download and decompress a BAL problem file if not already present.

    `name` is the path under the BAL data root, e.g.
    "ladybug/problem-49-7776-pre.txt.bz2".
    """
    if dest.exists():
        return dest
    url = f"https://grail.cs.washington.edu/projects/bal/data/{name}"
    print(f"Downloading {url} ...")
    compressed, _ = urllib.request.urlretrieve(url)
    dest.write_bytes(bz2.decompress(Path(compressed).read_bytes()))
    return dest


@functools.lru_cache(maxsize=4)
def _parse_bal(
    path_str: str,
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray, onp.ndarray, onp.ndarray]:
    """Parse a BAL problem file into (cam_idx, pt_idx, obs, cameras, points).

    Cached: the slow per-observation Python parse runs once even when the
    same problem is built with several `analyze()` configurations."""
    with Path(path_str).open() as f:
        tokens = f.read().split()
    it = iter(tokens)
    n_cams, n_pts, n_obs = int(next(it)), int(next(it)), int(next(it))

    cam_idx = onp.empty(n_obs, dtype=onp.int64)
    pt_idx = onp.empty(n_obs, dtype=onp.int64)
    obs = onp.empty((n_obs, 2), dtype=onp.float64)
    for i in range(n_obs):
        cam_idx[i] = int(next(it))
        pt_idx[i] = int(next(it))
        obs[i, 0] = float(next(it))
        obs[i, 1] = float(next(it))

    cameras = onp.array([float(next(it)) for _ in range(9 * n_cams)]).reshape(n_cams, 9)
    points = onp.array([float(next(it)) for _ in range(3 * n_pts)]).reshape(n_pts, 3)
    return cam_idx, pt_idx, obs, cameras, points


def load_bal(
    path: Path,
    schur_elimination: bool = True,
) -> tuple[jaxls.AnalyzedLeastSquaresProblem, jaxls.VarValues]:
    """Parse a BAL problem file and build the jaxls problem + initial values."""
    cam_idx, pt_idx, obs, cameras, points = _parse_bal(str(path))
    n_cams, n_pts = cameras.shape[0], points.shape[0]

    problem = _analyze(
        jaxls.LeastSquaresProblem(
            costs=[
                jaxls.Cost(
                    reprojection_cost,
                    (
                        CameraVar(jnp.array(cam_idx)),
                        PointVar(jnp.array(pt_idx)),
                        jnp.array(obs),
                    ),
                    name="reprojection",
                )
            ],
            variables=[CameraVar(jnp.arange(n_cams)), PointVar(jnp.arange(n_pts))],
        ),
        schur_elimination,
    )
    initial_vals = jaxls.VarValues.make(
        [
            CameraVar(jnp.arange(n_cams)).with_value(jnp.array(cameras)),
            PointVar(jnp.arange(n_pts)).with_value(jnp.array(points)),
        ]
    )
    return problem, initial_vals


def make_toy_ba(
    seed: int = 0,
    n_cams: int = 30,
    n_pts: int = 700,
    obs_per_point: int = 6,
    schur_elimination: bool = True,
) -> tuple[jaxls.AnalyzedLeastSquaresProblem, jaxls.VarValues]:
    """Synthetic, well-conditioned BA problem. Small enough that a full dense
    solve is feasible, for the exactness anchor and the well-conditioned
    counter-case."""
    rng = onp.random.default_rng(seed)
    cams_gt = onp.concatenate(
        [
            rng.normal(0, 0.1, (n_cams, 3)),
            rng.normal(0, 1.0, (n_cams, 2)),
            rng.normal(10.0, 0.5, (n_cams, 1)),
            onp.full((n_cams, 1), 500.0),
            onp.zeros((n_cams, 2)),
        ],
        axis=1,
    )
    pts_gt = rng.normal(0, 2.0, (n_pts, 3))

    cam_idx = onp.concatenate(
        [rng.choice(n_cams, size=obs_per_point, replace=False) for _ in range(n_pts)]
    )
    pt_idx = onp.repeat(onp.arange(n_pts), obs_per_point)

    project_batch = jax.vmap(project)
    obs_clean = project_batch(jnp.array(cams_gt[cam_idx]), jnp.array(pts_gt[pt_idx]))
    obs = jnp.array(onp.asarray(obs_clean) + rng.normal(0, 1.0, (len(cam_idx), 2)))

    problem = _analyze(
        jaxls.LeastSquaresProblem(
            costs=[
                jaxls.Cost(
                    reprojection_cost,
                    (CameraVar(jnp.array(cam_idx)), PointVar(jnp.array(pt_idx)), obs),
                    name="reprojection",
                )
            ],
            variables=[CameraVar(jnp.arange(n_cams)), PointVar(jnp.arange(n_pts))],
        ),
        schur_elimination,
    )
    initial_vals = jaxls.VarValues.make(
        [
            CameraVar(jnp.arange(n_cams)).with_value(
                jnp.array(cams_gt + rng.normal(0, 0.01, cams_gt.shape))
            ),
            PointVar(jnp.arange(n_pts)).with_value(
                jnp.array(pts_gt + rng.normal(0, 0.05, pts_gt.shape))
            ),
        ]
    )
    return problem, initial_vals
