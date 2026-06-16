"""Analytic-Jacobian bundle adjustment: benchmark vs autodiff.

The BAL reprojection cost (rodrigues rotation -> translate -> perspective
divide -> radial distortion -> focal scale) is op-heavy; its Jacobian via
`jax.vmap(jacfwd(...))` expands to ~800 HLO ops per observation. This module
provides a hand-derived analytic Jacobian and benchmarks it against autodiff
across devices, measuring solve time, per-iteration time, compile time, and
HLO op/kernel count.

Profiling (see results.md "Performance ceiling") predicts the analytic
Jacobian helps op count / compile time / CPU more than GPU single-solve
wall-clock (which is dispatch-bound). This benchmark quantifies that rather
than assuming it.

The Jacobian is verified against autodiff to ~1e-8 before any timing.

Usage:
    uv run --extra dev --extra docs python benchmarks/analytic_jac_ba.py                  # default: ladybug49
    uv run --extra dev --extra docs python benchmarks/analytic_jac_ba.py --problem trafalgar138 --device gpu
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as onp  # noqa: E402

import jaxls  # noqa: E402

import bal  # noqa: E402
from bal import CameraVar, PointVar, project, rodrigues  # noqa: E402


# ---------------------------------------------------------------------------
# Analytic Jacobian of the BAL reprojection
# ---------------------------------------------------------------------------
#
# residual(camera, point) = project(camera, point) - obs, with
#   p_cam = R(aa) @ point + t                       (camera-frame point)
#   proj  = -p_cam[:2] / p_cam[2]
#   r2    = |proj|^2
#   d     = 1 + k1 r2 + k2 r2^2                      (radial distortion)
#   pix   = f * d * proj
#
# We need d(pix)/d(camera) [2x9] and d(pix)/d(point) [2x3]. By the chain
# rule everything reduces to: d(pix)/d(p_cam) [2x3] composed with
# d(p_cam)/d(camera) and d(p_cam)/d(point), plus the explicit columns for
# (f, k1, k2). The variables are Euclidean jax.Arrays, so the tangent is the
# plain perturbation and the cost ordering is [camera(9), point(3)].


def _dpix_dpcam(p_cam: jax.Array, f: jax.Array, k1: jax.Array, k2: jax.Array):
    """d(pix)/d(p_cam): (2, 3). Also returns (proj, d) for reuse."""
    x, y, z = p_cam[0], p_cam[1], p_cam[2]
    proj = jnp.array([-x / z, -y / z])
    # d(proj)/d(p_cam): (2,3)
    dproj = jnp.array(
        [
            [-1.0 / z, 0.0, x / z**2],
            [0.0, -1.0 / z, y / z**2],
        ]
    )
    r2 = proj @ proj
    d = 1.0 + k1 * r2 + k2 * r2**2
    # pix = f d proj; d(pix) = f (d dproj + proj (dd/dproj) dproj)
    # dd/dr2 = k1 + 2 k2 r2; dr2/dproj = 2 proj
    dd_dproj = (k1 + 2.0 * k2 * r2) * 2.0 * proj  # (2,)
    # d(pix)/d(proj) = f (d I + proj[:,None] * dd_dproj[None,:])   (2,2)
    dpix_dproj = f * (d * jnp.eye(2) + jnp.outer(proj, dd_dproj))
    dpix_dpcam = dpix_dproj @ dproj  # (2,3)
    return dpix_dpcam, proj, d, r2


def _drodrigues_daa(aa: jax.Array, x: jax.Array) -> jax.Array:
    """d(R(aa) x)/d(aa): (3, 3). Differentiates rodrigues w.r.t. the
    angle-axis vector. Uses autodiff on the small 3-vector function — this is
    a tiny graph (one rotation), unlike differentiating the full projection,
    and keeps the rotation derivative exact without a fragile hand expansion
    of the theta->0 limit."""
    return jax.jacobian(rodrigues, argnums=0)(aa, x)


def reprojection_cost(vals, cam, pt, obs):
    return project(vals[cam], vals[pt]) - obs


def analytic_jac(vals, cam, pt, obs):
    """Per-observation analytic Jacobian, shape (2, 12) = [d/d cam(9) | d/d pt(3)].

    Called inside `jax.vmap` by `_compute_jac_values`, so `vals[cam]` is a
    single (9,) camera and `vals[pt]` a single (3,) point."""
    camera = vals[cam]
    point = vals[pt]
    aa, t = camera[:3], camera[3:6]
    f, k1, k2 = camera[6], camera[7], camera[8]

    R_point = rodrigues(aa, point)
    p_cam = R_point + t
    proj = jnp.array([-p_cam[0] / p_cam[2], -p_cam[1] / p_cam[2]])
    dpix_dpcam, _, d, r2 = _dpix_dpcam(p_cam, f, k1, k2)

    # --- camera columns (9) ---
    # rotation (aa, 3): d(pix)/d(aa) = dpix_dpcam @ d(R point)/d(aa)
    dpcam_daa = _drodrigues_daa(aa, point)  # (3,3)
    dpix_daa = dpix_dpcam @ dpcam_daa  # (2,3)
    # translation (3): d(p_cam)/dt = I  ->  dpix_dpcam
    dpix_dt = dpix_dpcam  # (2,3)
    # focal (1): pix = f d proj -> d(pix)/df = d proj
    dpix_df = (d * proj)[:, None]  # (2,1)
    # k1 (1): d(pix)/dk1 = f proj * d(d)/dk1 = f proj r2
    dpix_dk1 = (f * proj * r2)[:, None]  # (2,1)
    # k2 (1): d(pix)/dk2 = f proj r2^2
    dpix_dk2 = (f * proj * r2**2)[:, None]  # (2,1)
    jac_cam = jnp.concatenate(
        [dpix_daa, dpix_dt, dpix_df, dpix_dk1, dpix_dk2], axis=1
    )  # (2,9)

    # --- point columns (3) ---
    # d(p_cam)/d(point) = R(aa); compose with dpix_dpcam
    dpcam_dpoint = jax.jacobian(rodrigues, argnums=1)(aa, point)  # (3,3) = R
    jac_pt = dpix_dpcam @ dpcam_dpoint  # (2,3)

    return jnp.concatenate([jac_cam, jac_pt], axis=1)  # (2,12)


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

_FILES = {
    "ladybug49": ("ladybug/problem-49-7776-pre.txt.bz2", "/tmp/ladybug-49.txt"),
    "trafalgar138": (
        "trafalgar/problem-138-44033-pre.txt.bz2",
        "/tmp/trafalgar-138.txt",
    ),
}


def build(problem_name: str, analytic: bool):
    name, dest = _FILES[problem_name]
    path = bal.download_bal(name, Path(dest))
    cam_idx, pt_idx, obs, cameras, points = bal._parse_bal(str(path))
    ncam, npt = cameras.shape[0], points.shape[0]
    kw = dict(jac_custom_fn=analytic_jac) if analytic else {}
    cost = jaxls.Cost(
        reprojection_cost,
        (CameraVar(jnp.array(cam_idx)), PointVar(jnp.array(pt_idx)), jnp.array(obs)),
        name="reprojection",
        **kw,
    )
    prob = jaxls.LeastSquaresProblem(
        [cost], [CameraVar(jnp.arange(ncam)), PointVar(jnp.arange(npt))]
    ).analyze(schur_elimination=True)
    init = jaxls.VarValues.make(
        [
            CameraVar(jnp.arange(ncam)).with_value(jnp.array(cameras)),
            PointVar(jnp.arange(npt)).with_value(jnp.array(points)),
        ]
    )
    return prob, init


# ---------------------------------------------------------------------------
# Verification + benchmark
# ---------------------------------------------------------------------------


def verify(problem_name: str) -> float:
    """Max abs diff between analytic and autodiff Jacobian values."""
    prob_ad, init = build(problem_name, analytic=False)
    prob_an, _ = build(problem_name, analytic=True)
    ci = prob_ad._compute_cost_info(init)
    A_ad = jax.jit(lambda v, c: prob_ad._compute_jac_values(v, c))(init, ci.jac_cache)
    A_an = jax.jit(lambda v, c: prob_an._compute_jac_values(v, c))(init, ci.jac_cache)
    d = 0.0
    for ba, bb in zip(A_ad.block_rows, A_an.block_rows):
        d = max(d, float(jnp.max(jnp.abs(ba.blocks_concat - bb.blocks_concat))))
    return d


def hlo_kernels(prob, init) -> tuple[int, int]:
    @jax.jit
    def solve():
        return prob.solve(
            init,
            linear_solver="dense_cholesky",
            termination=jaxls.TerminationConfig(
                max_iterations=30, early_termination=False
            ),
            return_summary=True,
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1e2),
        )

    txt = solve.lower().compile().as_text()
    return txt.count("fusion("), txt.count("custom-call(")


def bench(problem_name: str, device: str, repeats: int = 5) -> dict:
    dev = jax.devices(device)[0]
    out = {}
    for label, analytic in [("autodiff", False), ("analytic", True)]:
        prob, init = build(problem_name, analytic)
        prob = jax.device_put(prob, dev)
        init = jax.device_put(init, dev)

        def solve():
            return prob.solve(
                init,
                linear_solver="dense_cholesky",
                termination=jaxls.TerminationConfig(
                    max_iterations=30, early_termination=False
                ),
                return_summary=True,
                trust_region=jaxls.TrustRegionConfig(lambda_initial=1e2),
            )

        # Compile time = first call wall-time minus warm time.
        t0 = time.perf_counter()
        sol, summary = jax.block_until_ready(solve())
        cold = time.perf_counter() - t0
        cost = float(onp.asarray(prob._compute_cost_info(sol).cost_total))
        warm = 1e9
        for _ in range(repeats):
            s = time.perf_counter()
            jax.block_until_ready(solve())
            warm = min(warm, time.perf_counter() - s)
        # Per-step timestamps from a recorded solve (host float64). Entering
        # the recorder recompiles with the callback; warm up, then measure.
        with jaxls.record_iteration_times() as times:
            jax.block_until_ready(solve())
            times.clear()
            jax.block_until_ready(solve())
        fusions, customs = hlo_kernels(prob, init)
        n = int(summary.iterations)
        elapsed = [t - times[0] for t in times][: n + 1]
        # Running-best accepted cost; cost_history records proposals.
        ch = onp.asarray(summary.cost_history)[: n + 1].copy()
        for i in range(1, len(ch)):
            ch[i] = min(ch[i], ch[i - 1])
        # cost_history is only max_iterations long (init + max_iters-1 steps),
        # while the recorder has no cap; align both to the shorter length.
        m = min(len(ch), len(elapsed))
        ch = ch[:m]
        elapsed = elapsed[:m]
        out[label] = {
            "warm_ms": warm * 1e3,
            "per_iter_ms": warm / 30 * 1e3,
            "compile_s": cold - warm,
            "final_cost": cost,
            "fusions": fusions,
            "custom_calls": customs,
            "cost_history": ch.tolist(),
            "elapsed_s": elapsed,
        }
    return out


def plot(problem_name: str, device: str, res: dict) -> str:
    import matplotlib.pyplot as plt

    fig, (ax_step, ax_time) = plt.subplots(1, 2, figsize=(11, 4.2))
    style = {"autodiff": ("#d62728", "o", "--"), "analytic": ("#1f77b4", "^", "-")}
    for label, (color, marker, ls) in style.items():
        r = res[label]
        ch = r["cost_history"]
        steps = list(range(len(ch)))
        ax_step.plot(steps, ch, marker=marker, ms=4, ls=ls, color=color, label=label)
        ax_time.plot(
            r["elapsed_s"], ch, marker=marker, ms=4, ls=ls, color=color, label=label
        )
    for ax, xlab in [(ax_step, "outer LM step"), (ax_time, "wall-clock (s)")]:
        ax.set_yscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel("accepted cost")
        ax.legend()
    ax_time.set_xscale("symlog", linthresh=1e-2)
    ax_time.set_xlim(left=0)
    fig.suptitle(
        f"{problem_name} ({device}): analytic vs autodiff Jacobian — "
        "identical cost path, different speed"
    )
    fig.tight_layout()
    out = (
        Path(__file__).parent / "results" / f"analytic_jac_{problem_name}_{device}.png"
    )
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="ladybug49", choices=list(_FILES))
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    args = ap.parse_args()

    diff = verify(args.problem)
    print(f"Jacobian verification (analytic vs autodiff): max abs diff = {diff:.2e}")
    assert diff < 1e-6, "analytic Jacobian disagrees with autodiff!"

    res = bench(args.problem, args.device)
    print(f"\n{args.problem} on {args.device} (dense, lambda0=1e2, 30 iters):\n")
    hdr = f"{'metric':<16}{'autodiff':>14}{'analytic':>14}{'ratio':>9}"
    print(hdr)
    for key, fmt in [
        ("warm_ms", "{:.2f}"),
        ("per_iter_ms", "{:.3f}"),
        ("compile_s", "{:.2f}"),
        ("fusions", "{:.0f}"),
        ("custom_calls", "{:.0f}"),
        ("final_cost", "{:.1f}"),
    ]:
        a, b = res["autodiff"][key], res["analytic"][key]
        ratio = (a / b) if b else float("nan")
        print(f"{key:<16}{fmt.format(a):>14}{fmt.format(b):>14}{ratio:>8.2f}x")

    # Cost trajectory before/after: must be identical (same Jacobian math),
    # confirming analytic changes speed only, not the optimization path.
    print("\ncost trajectory (running-best accepted cost per outer step):")
    print(f"  {'step':>4}{'autodiff':>16}{'analytic':>16}")
    ad, an = res["autodiff"]["cost_history"], res["analytic"]["cost_history"]
    for i in range(0, len(ad), max(1, len(ad) // 10)):
        print(f"  {i:>4}{ad[i]:>16.4f}{an[i]:>16.4f}")
    max_traj_diff = max(abs(a - b) for a, b in zip(ad, an))
    print(f"  max |autodiff - analytic| over trajectory: {max_traj_diff:.2e}")

    path = plot(args.problem, args.device, res)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
