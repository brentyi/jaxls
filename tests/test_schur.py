"""Tests for variable elimination (Schur complement).

Elimination is automatic: `solve()` eliminates dominant block-diagonal
variable types for all three reduced-solve paths (dense Cholesky, CG, and
CHOLMOD sparse-direct on the reduced system). Covers:
- Exactness: the (automatic) reduced solve reproduces the full dense step.
- Multiple kept types, and multiple eliminated types.
- Automatic inference: what gets eliminated, and graceful fallbacks.
- Plan-builder validation of structurally invalid eliminations.
- float32 robustness of the reduced SPD solve.
"""

import sys
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import jaxls

if sys.version_info >= (3, 12):
    from jaxls import _problem, _schur
else:
    # On older interpreters the package runs from the transpiled sources.
    from jaxls._py310 import _problem, _schur  # type: ignore[no-redef]

jax.config.update("jax_enable_x64", True)


class CamVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(9)): ...


class BiasVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(2)): ...


class PointVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(3)): ...


class ColorVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(1)): ...


def _rodrigues(aa: jax.Array, x: jax.Array) -> jax.Array:
    theta = jnp.linalg.norm(aa) + 1e-12
    k = aa / theta
    return (
        x * jnp.cos(theta)
        + jnp.cross(k, x) * jnp.sin(theta)
        + k * jnp.dot(k, x) * (1.0 - jnp.cos(theta))
    )


def _reproject(cam: jax.Array, point: jax.Array) -> jax.Array:
    p = _rodrigues(cam[:3], point) + cam[3:6]
    proj = -p[:2] / p[2]
    f, k1, k2 = cam[6], cam[7], cam[8]
    r2 = jnp.sum(proj**2)
    return f * (1.0 + k1 * r2 + k2 * r2**2) * proj


def _make_ba_problem(
    seed: int = 0,
    n_cams: int = 4,
    n_pts: int = 12,
    with_bias: bool = False,
    extra_costs: list[jaxls.Cost] | None = None,
    extra_variables: list[jaxls.Var] | None = None,
    schur_elimination: Literal["auto", "off"] | tuple[type[jaxls.Var], ...] = "auto",
) -> tuple[jaxls.AnalyzedLeastSquaresProblem, jaxls.VarValues]:
    """Small synthetic bundle adjustment problem. Includes a kept-only cost
    group (camera priors) and an eliminated-only cost group (point priors).
    With `with_bias=True`, each observation also couples a second kept
    variable type, exercising multi-kept-slot groups and cross-type pairs in
    the reduced system. `extra_costs`/`extra_variables` are appended to the
    problem; extra variables are initialized to their type defaults."""
    rng = onp.random.default_rng(seed)
    cams_gt = onp.concatenate(
        [
            rng.normal(0, 0.1, (n_cams, 3)),
            rng.normal(0, 0.5, (n_cams, 2)),
            rng.normal(5.0, 0.2, (n_cams, 1)),
            onp.full((n_cams, 1), 500.0),
            onp.zeros((n_cams, 2)),
        ],
        axis=1,
    )
    pts_gt = rng.normal(0, 1.0, (n_pts, 3))

    cam_idx_list = []
    pt_idx_list = []
    for j in range(n_pts):
        for i in rng.choice(n_cams, size=3, replace=False):
            cam_idx_list.append(int(i))
            pt_idx_list.append(j)
    cam_idx = onp.array(cam_idx_list)
    pt_idx = onp.array(pt_idx_list)

    obs_list = []
    for i, j in zip(cam_idx, pt_idx):
        z = _reproject(jnp.array(cams_gt[i]), jnp.array(pts_gt[j]))
        obs_list.append(onp.asarray(z) + rng.normal(0, 1.0, 2))
    obs = jnp.array(onp.array(obs_list))

    costs = list[jaxls.Cost]()
    if with_bias:

        def residual_bias(
            vals: jaxls.VarValues,
            cam: CamVar,
            bias: BiasVar,
            pt: PointVar,
            z: jax.Array,
        ) -> jax.Array:
            return _reproject(vals[cam], vals[pt]) + vals[bias] - z

        costs.append(
            jaxls.Cost(
                residual_bias,
                (
                    CamVar(jnp.array(cam_idx)),
                    BiasVar(jnp.array(cam_idx)),
                    PointVar(jnp.array(pt_idx)),
                    obs,
                ),
            )
        )
        costs.append(
            jaxls.Cost(
                lambda vals, bias: 1e-2 * vals[bias],
                (BiasVar(jnp.arange(n_cams)),),
            )
        )
    else:

        def residual(
            vals: jaxls.VarValues, cam: CamVar, pt: PointVar, z: jax.Array
        ) -> jax.Array:
            return _reproject(vals[cam], vals[pt]) - z

        costs.append(
            jaxls.Cost(
                residual,
                (CamVar(jnp.array(cam_idx)), PointVar(jnp.array(pt_idx)), obs),
            )
        )
    # Kept-only cost group.
    costs.append(
        jaxls.Cost(
            lambda vals, cam, target: 1e-3 * (vals[cam] - target),
            (CamVar(jnp.arange(n_cams)), jnp.array(cams_gt)),
        )
    )
    # Eliminated-only cost group.
    costs.append(
        jaxls.Cost(
            lambda vals, pt, target: 1e-3 * (vals[pt] - target),
            (PointVar(jnp.arange(n_pts)), jnp.array(pts_gt)),
        )
    )

    if extra_costs is not None:
        costs.extend(extra_costs)

    variables: list[jaxls.Var] = [
        CamVar(jnp.arange(n_cams)),
        PointVar(jnp.arange(n_pts)),
    ]
    if with_bias:
        variables.append(BiasVar(jnp.arange(n_cams)))
    if extra_variables is not None:
        variables.extend(extra_variables)
    problem = jaxls.LeastSquaresProblem(costs, variables).analyze(
        schur_elimination=schur_elimination
    )

    init_list: list = [
        CamVar(jnp.arange(n_cams)).with_value(
            jnp.array(cams_gt + rng.normal(0, 0.02, cams_gt.shape))
        ),
        PointVar(jnp.arange(n_pts)).with_value(
            jnp.array(pts_gt + rng.normal(0, 0.05, pts_gt.shape))
        ),
    ]
    if with_bias:
        init_list.append(
            BiasVar(jnp.arange(n_cams)).with_value(
                jnp.array(rng.normal(0, 0.1, (n_cams, 2)))
            )
        )
    if extra_variables is not None:
        init_list.extend(extra_variables)
    return problem, jaxls.VarValues.make(init_list)


def _solve_kwargs(max_iterations: int = 10) -> dict:
    return dict(
        termination=jaxls.TerminationConfig(
            max_iterations=max_iterations, early_termination=False
        ),
        verbose=False,
        return_summary=True,
    )


def _solve_without_elimination(
    linear_solver: str,
    max_iterations: int = 10,
    **fixture_kwargs,
) -> tuple[jaxls.VarValues, jaxls.SolveSummary]:
    """Reference solve on the full system: rebuilds the (deterministic)
    fixture with `analyze(schur_elimination="off")` and solves it."""
    problem, init = _make_ba_problem(schur_elimination="off", **fixture_kwargs)
    assert problem._elimination is None
    return problem.solve(
        init,
        linear_solver=linear_solver,  # type: ignore[arg-type]
        **_solve_kwargs(max_iterations),
    )


def _cost_history(summary: jaxls.SolveSummary, n: int) -> onp.ndarray:
    return onp.asarray(summary.cost_history[:n])


def test_schur_dense_exactness() -> None:
    """dense_cholesky automatically eliminates the landmark block, and the
    reduced solve must trace the same LM path as the full dense solve."""
    problem, init = _make_ba_problem()
    assert _schur.infer_eliminate(problem) == (PointVar,)
    _, summary_full = _solve_without_elimination("dense_cholesky")
    _, summary_schur = problem.solve(
        init, linear_solver="dense_cholesky", **_solve_kwargs()
    )
    c_full = _cost_history(summary_full, 10)
    c_schur = _cost_history(summary_schur, 10)
    rel = onp.abs(c_full - c_schur) / onp.abs(c_full)
    assert rel.max() < 1e-6, f"max relative cost difference {rel.max()}"


def test_schur_dense_exactness_multiple_kept_types() -> None:
    """Exactness with two kept types per observation (cross-type blocks in
    the reduced system). Compared over the pre-convergence horizon; once LM
    starts rejecting steps near the optimum, last-bit differences can flip an
    accept decision and the trajectories separate (the single-step test below
    is the sharp check)."""
    problem, init = _make_ba_problem(with_bias=True, n_pts=18)
    assert _schur.infer_eliminate(problem) == (PointVar,)
    _, summary_full = _solve_without_elimination(
        "dense_cholesky", max_iterations=6, with_bias=True, n_pts=18
    )
    _, summary_schur = problem.solve(
        init, linear_solver="dense_cholesky", **_solve_kwargs(6)
    )
    c_full = _cost_history(summary_full, 6)
    c_schur = _cost_history(summary_schur, 6)
    rel = onp.abs(c_full - c_schur) / onp.abs(c_full)
    assert rel.max() < 1e-6, f"max relative cost difference {rel.max()}"


def test_schur_single_step_exactness() -> None:
    """The damped reduced step equals the explicit full dense solve, for a
    sweep of damping values, with multiple kept types in play."""
    problem, init = _make_ba_problem(with_bias=True)
    cost_info = problem._compute_cost_info(init)
    A = problem._compute_jac_values(init, cost_info.jac_cache)
    A_dense = A.to_dense()
    ATb = -A_dense.T @ cost_info.residual_vector

    plan = _schur.build_elimination_plan(problem, (PointVar,))
    factors = _schur.prepare_schur(plan, A, ATb, linear_solver="dense_cholesky")
    for lambd in (1e-4, 1e-2, 1.0):
        ATA = A_dense.T @ A_dense + lambd * jnp.eye(A_dense.shape[1])
        ref = jnp.linalg.solve(ATA, ATb)
        step = _schur.solve_schur_dense(factors, lambd)
        rel = float(jnp.linalg.norm(step - ref) / jnp.linalg.norm(ref))
        assert rel < 1e-8, f"lambda={lambd}: relative step difference {rel}"


try:
    import sksparse.cholmod  # noqa: F401

    _HAS_CHOLMOD = True
except Exception:
    _HAS_CHOLMOD = False

requires_cholmod = pytest.mark.skipif(
    not _HAS_CHOLMOD, reason="sksparse.cholmod not available"
)


def test_sparse_s_equals_dense_s() -> None:
    """The sparse COO assembly of S (summed) reproduces the dense S, for a
    sweep of damping values. Pure assembly check, no CHOLMOD needed."""
    import scipy.sparse

    problem, init = _make_ba_problem(with_bias=True)
    cost_info = problem._compute_cost_info(init)
    A = problem._compute_jac_values(init, cost_info.jac_cache)
    ATb = -A.to_dense().T @ cost_info.residual_vector

    plan = _schur.build_elimination_plan(problem, (PointVar,))
    assert plan.sparse_s_pattern is not None
    factors_dense = _schur.prepare_schur(plan, A, ATb, linear_solver="dense_cholesky")
    factors_sparse = _schur.prepare_schur(plan, A, ATb, linear_solver="cholmod")
    for lambd in (1e-4, 1e-2, 1.0):
        vinv = _schur._damped_vinv(factors_dense, lambd)
        S_dense = _schur._assemble_dense_S(factors_dense, lambd, vinv)

        vinv_s = _schur._damped_vinv(factors_sparse, lambd)
        s_values = _schur._assemble_S_values(factors_sparse, lambd, vinv_s)
        pat = plan.sparse_s_pattern
        S_sparse = scipy.sparse.coo_matrix(
            (onp.asarray(s_values), (onp.asarray(pat.rows), onp.asarray(pat.cols))),
            shape=(plan.reduced_dim, plan.reduced_dim),
        ).toarray()

        rel = float(
            onp.linalg.norm(S_sparse - onp.asarray(S_dense))
            / onp.linalg.norm(onp.asarray(S_dense))
        )
        assert rel < 1e-10, f"lambda={lambd}: sparse vs dense S relative diff {rel}"


@requires_cholmod
def test_schur_cholmod_single_step_exactness() -> None:
    """The CHOLMOD reduced step equals the explicit full dense solve. The
    CHOLMOD path regularizes by lambd + 1e-5, so the reference matches."""
    problem, init = _make_ba_problem(with_bias=True)
    cost_info = problem._compute_cost_info(init)
    A = problem._compute_jac_values(init, cost_info.jac_cache)
    A_dense = A.to_dense()
    ATb = -A_dense.T @ cost_info.residual_vector

    plan = _schur.build_elimination_plan(problem, (PointVar,))
    factors = _schur.prepare_schur(plan, A, ATb, linear_solver="cholmod")
    for lambd in (1e-4, 1e-2, 1.0):
        ATA = A_dense.T @ A_dense + (lambd + 1e-5) * jnp.eye(A_dense.shape[1])
        ref = jnp.linalg.solve(ATA, ATb)
        step = _schur.solve_schur_cholmod(factors, lambd)
        rel = float(jnp.linalg.norm(step - ref) / jnp.linalg.norm(ref))
        assert rel < 1e-8, f"lambda={lambd}: relative step difference {rel}"


@requires_cholmod
def test_schur_cholmod_converges() -> None:
    """End-to-end: Schur + CHOLMOD reaches the same optimum as Schur + dense
    over a full LM trajectory."""
    problem, init = _make_ba_problem()
    _, summary_dense = problem.solve(
        init, linear_solver="dense_cholesky", **_solve_kwargs(20)
    )
    _, summary_cholmod = problem.solve(
        init, linear_solver="cholmod", **_solve_kwargs(20)
    )
    cost_dense = _cost_history(summary_dense, 20)[-1]
    cost_cholmod = _cost_history(summary_cholmod, 20)[-1]
    rel = abs(cost_cholmod - cost_dense) / (abs(cost_dense) + 1e-12)
    assert rel < 1e-5, f"final cost mismatch: dense={cost_dense} cholmod={cost_cholmod}"


def test_schur_dense_exactness_multiple_elim_types() -> None:
    """Two eliminated types (in different costs) at once."""
    n_pts = 12
    rng = onp.random.default_rng(1)

    # Per-point color variables, observed independently (a cost may not
    # couple two eliminated variables, so colors get their own costs).
    color_obs = jnp.array(rng.normal(0.5, 0.1, (n_pts, 1)))
    color_costs = [
        jaxls.Cost(
            lambda vals, color, z: vals[color] - z,
            (ColorVar(jnp.arange(n_pts)), color_obs),
        )
    ]
    problem, init = _make_ba_problem(
        extra_costs=color_costs,
        extra_variables=[ColorVar(jnp.arange(n_pts))],
    )
    assert _schur.infer_eliminate(problem) == (PointVar, ColorVar)
    _, summary_full = _solve_without_elimination(
        "dense_cholesky",
        extra_costs=color_costs,
        extra_variables=[ColorVar(jnp.arange(n_pts))],
    )
    _, summary_schur = problem.solve(
        init, linear_solver="dense_cholesky", **_solve_kwargs()
    )
    c_full = _cost_history(summary_full, 10)
    c_schur = _cost_history(summary_schur, 10)
    rel = onp.abs(c_full - c_schur) / onp.abs(c_full)
    assert rel.max() < 1e-6, f"max relative cost difference {rel.max()}"


def test_schur_cg_converges() -> None:
    """The matrix-free reduced CG path reaches the same optimum. Compared on
    the cost of the returned (accepted) solutions, not the proposal
    history."""
    problem, init = _make_ba_problem()
    # CG trails the exact dense steps by a few outer iterations (its
    # Eisenstat-Walker tolerance tightens over the lambda trajectory); 20
    # iterations is enough for both to reach the optimum.
    sol_dense, _ = problem.solve(
        init, linear_solver="dense_cholesky", **_solve_kwargs(20)
    )
    sol_cg, _ = problem.solve(
        init, linear_solver="conjugate_gradient", **_solve_kwargs(20)
    )
    final_dense = float(onp.asarray(problem._compute_cost_info(sol_dense).cost_total))
    final_cg = float(onp.asarray(problem._compute_cost_info(sol_cg).cost_total))
    assert abs(final_cg - final_dense) / final_dense < 1e-3


def test_schur_cg_preconditioner_options() -> None:
    """The Schur CG path must honor `ConjugateGradientConfig.preconditioner`
    and converge with every option."""
    problem, init = _make_ba_problem()
    assert _schur.infer_eliminate(problem) == (PointVar,)
    # 20 iterations: see test_schur_cg_converges.
    sol_ref, _ = problem.solve(
        init, linear_solver="dense_cholesky", **_solve_kwargs(20)
    )
    ref = float(onp.asarray(problem._compute_cost_info(sol_ref).cost_total))
    # Unpreconditioned CG's convergence rate is bound by the raw conditioning
    # of the reduced system, so it lags the preconditioned options at any
    # fixed iteration budget; it gets a looser parity bound.
    preconditioners: tuple[
        tuple[Literal["block_jacobi", "point_jacobi"] | None, float], ...
    ] = (
        ("block_jacobi", 1e-2),
        ("point_jacobi", 1e-2),
        (None, 1e-1),
    )
    for preconditioner, tol in preconditioners:
        sol, _ = problem.solve(
            init,
            linear_solver=jaxls.ConjugateGradientConfig(preconditioner=preconditioner),
            **_solve_kwargs(20),
        )
        cost = float(onp.asarray(problem._compute_cost_info(sol).cost_total))
        assert abs(cost - ref) / ref < tol, f"preconditioner={preconditioner}"


def test_no_elimination_when_ineligible() -> None:
    """When the dominant type self-couples (pose-graph style), solves run on
    the full system without elimination."""
    costs = [
        jaxls.Cost(
            lambda vals, c0, c1: vals[c0] - vals[c1],
            (CamVar(jnp.arange(3)), CamVar(jnp.arange(1, 4))),
        ),
        jaxls.Cost(
            lambda vals, c0: vals[c0] - 1.0,
            (CamVar(jnp.array([0])),),
        ),
    ]
    problem = jaxls.LeastSquaresProblem(costs, [CamVar(jnp.arange(4))]).analyze()
    assert _schur.infer_eliminate(problem) == ()
    _, summary = problem.solve(None, linear_solver="dense_cholesky", **_solve_kwargs(5))
    costs_history = _cost_history(summary, 5)
    assert costs_history[-1] < costs_history[0]


def test_cholmod_receives_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """cholmod now solves the reduced (Schur) system sparse-directly, so the
    precomputed elimination plan IS handed to the solver."""
    problem, init = _make_ba_problem()
    assert problem._elimination is not None  # Plan was prebuilt by analyze().
    captured = {}
    real_solver = _problem.NonlinearSolver

    def spy_solver(*args, **kwargs):
        solver = real_solver(*args, **kwargs)
        captured["elimination"] = solver.elimination
        return solver

    monkeypatch.setattr(_problem, "NonlinearSolver", spy_solver)
    try:
        problem.solve(init, linear_solver="cholmod", **_solve_kwargs(2))
    except Exception:
        # CHOLMOD itself may be unavailable in this environment; the solver
        # is constructed (and the plan selected) before factorization.
        pass
    assert captured["elimination"] is not None, "cholmod did not receive the plan"


def test_solver_receives_prebuilt_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """By default the solver receives the plan that analyze() prebuilt."""
    problem, init = _make_ba_problem()
    assert problem._elimination is not None  # Plan was prebuilt by analyze().
    captured = {}
    real_solver = _problem.NonlinearSolver

    def spy_solver(*args, **kwargs):
        solver = real_solver(*args, **kwargs)
        captured["elimination"] = solver.elimination
        return solver

    monkeypatch.setattr(_problem, "NonlinearSolver", spy_solver)
    problem.solve(init, linear_solver="dense_cholesky", **_solve_kwargs(2))
    assert captured["elimination"] is not None


def test_analyze_flag_skips_plan() -> None:
    """analyze(schur_elimination="off") must not build a plan, and solves of
    the resulting problem run on the full system."""
    costs = [
        jaxls.Cost(
            lambda vals, cam, pt: vals[cam][:3] - vals[pt],
            (CamVar(jnp.array([0, 0])), PointVar(jnp.arange(2))),
        )
    ]
    variables: list[jaxls.Var] = [CamVar(jnp.array([0])), PointVar(jnp.arange(2))]
    with_plan = jaxls.LeastSquaresProblem(costs, variables).analyze()
    without_plan = jaxls.LeastSquaresProblem(costs, variables).analyze(
        schur_elimination="off"
    )
    assert with_plan._elimination is not None
    assert without_plan._elimination is None
    without_plan.solve(None, linear_solver="dense_cholesky", verbose=False)


def test_analyze_explicit_eliminate_tuple() -> None:
    """A tuple of variable types eliminates exactly those types, bypassing
    inference. Eliminating PointVar here matches what "auto" would pick, and the
    resulting solve still converges."""
    problem, init = _make_ba_problem(schur_elimination=(PointVar,))
    assert problem._elimination is not None
    assert [t.count for t in problem._elimination.elim_types]  # non-empty
    # The eliminated type is PointVar; cameras are kept.
    elim_dim = problem._tangent_dim - problem._elimination.reduced_dim
    assert elim_dim == 12 * PointVar.tangent_dim  # n_pts default = 12
    problem.solve(init, linear_solver="dense_cholesky", **_solve_kwargs(5))


def test_analyze_explicit_eliminate_rejects_bad_type() -> None:
    """An explicit tuple naming a non-block-diagonal type is rejected at
    analyze() time (build_elimination_plan raises). A cost couples two
    PointVars, so the point block is not block-diagonal; CamVar is kept."""
    costs = [
        jaxls.Cost(
            lambda vals, p0, p1: vals[p0] - vals[p1],
            (PointVar(jnp.array([0])), PointVar(jnp.array([1]))),
        ),
        jaxls.Cost(
            lambda vals, cam, pt: vals[cam][:3] - vals[pt],
            (CamVar(jnp.array([0])), PointVar(jnp.array([0]))),
        ),
    ]
    variables: list[jaxls.Var] = [CamVar(jnp.array([0])), PointVar(jnp.arange(2))]
    with pytest.raises(ValueError, match="couples multiple eliminated"):
        jaxls.LeastSquaresProblem(costs, variables).analyze(
            schur_elimination=(PointVar,)
        )


def test_analyze_rejects_invalid_schur_arg() -> None:
    """schur_elimination must be 'auto', 'off', or a tuple of types."""
    costs = [
        jaxls.Cost(
            lambda vals, cam, pt: vals[cam][:3] - vals[pt],
            (CamVar(jnp.array([0])), PointVar(jnp.array([0]))),
        )
    ]
    prob = jaxls.LeastSquaresProblem(
        costs, [CamVar(jnp.array([0])), PointVar(jnp.arange(1))]
    )
    with pytest.raises(ValueError, match="must be 'auto', 'off'"):
        prob.analyze(schur_elimination="sometimes")  # type: ignore[arg-type]


def test_plan_rejects_coupled_same_type() -> None:
    """A cost coupling two eliminated variables of the same type must be
    rejected by the plan builder (automatic inference never selects such a
    set; this guards the internal API)."""
    costs = [
        jaxls.Cost(
            lambda vals, p0, p1: vals[p0] - vals[p1],
            (PointVar(jnp.array([0])), PointVar(jnp.array([1]))),
        ),
        jaxls.Cost(
            lambda vals, cam, pt: vals[cam][:3] - vals[pt],
            (CamVar(jnp.array([0])), PointVar(jnp.array([0]))),
        ),
    ]
    analyzed = jaxls.LeastSquaresProblem(
        costs, [CamVar(jnp.array([0])), PointVar(jnp.arange(2))]
    ).analyze()
    # Automatic inference avoids the self-coupled point block (it falls back
    # to the camera block, which is eligible here).
    assert PointVar not in _schur.infer_eliminate(analyzed)
    with pytest.raises(ValueError, match="couples multiple eliminated"):
        _schur.build_elimination_plan(analyzed, (PointVar,))


def test_plan_rejects_coupled_different_types() -> None:
    """A cost coupling two different eliminated types must be rejected."""
    costs = [
        jaxls.Cost(
            lambda vals, pt, color: vals[pt][:1] - vals[color],
            (PointVar(jnp.array([0])), ColorVar(jnp.array([0]))),
        ),
        jaxls.Cost(
            lambda vals, cam, pt: vals[cam][:3] - vals[pt],
            (CamVar(jnp.array([0])), PointVar(jnp.array([0]))),
        ),
    ]
    analyzed = jaxls.LeastSquaresProblem(
        costs,
        [CamVar(jnp.array([0])), PointVar(jnp.array([0])), ColorVar(jnp.array([0]))],
    ).analyze()
    with pytest.raises(ValueError, match="couples multiple eliminated"):
        _schur.build_elimination_plan(analyzed, (PointVar, ColorVar))


def test_plan_rejects_absent_type() -> None:
    """Eliminating a type with no variables in the problem must raise."""
    problem, _ = _make_ba_problem()
    with pytest.raises(ValueError, match="no variables"):
        _schur.build_elimination_plan(problem, (ColorVar,))


def test_plan_rejects_eliminating_everything() -> None:
    """Eliminating every variable type must raise."""
    problem, _ = _make_ba_problem()
    with pytest.raises(ValueError, match="at least one"):
        _schur.build_elimination_plan(problem, (CamVar, PointVar))


def test_traced_problem_uses_prebuilt_plan() -> None:
    """The elimination plan is built at analyze() time, so solving a traced
    problem inside jax.jit works (the plan's index arrays are ordinary
    pytree leaves) instead of raising."""
    problem, init = _make_ba_problem()

    @jax.jit
    def solve_traced_problem(
        problem_traced: jaxls.AnalyzedLeastSquaresProblem,
        init_vals: jaxls.VarValues,
    ) -> jaxls.VarValues:
        return problem_traced.solve(
            init_vals,
            linear_solver="dense_cholesky",
            termination=jaxls.TerminationConfig(
                max_iterations=5, early_termination=False
            ),
            verbose=False,
        )

    sol = solve_traced_problem(problem, init)
    cost_before = float(onp.asarray(problem._compute_cost_info(init).cost_total))
    cost_after = float(onp.asarray(problem._compute_cost_info(sol).cost_total))
    assert cost_after < cost_before


def test_solve_spd_scaled_float64_exact() -> None:
    """The robust SPD solve must not perturb well-conditioned float64 solves."""
    rng = onp.random.default_rng(0)
    A = rng.normal(size=(20, 20))
    S = jnp.array(A @ A.T + 20 * onp.eye(20))
    x_gt = jnp.array(rng.normal(size=20))
    b = S @ x_gt
    x = _schur._solve_spd_scaled(S, b)
    assert float(jnp.linalg.norm(x - x_gt) / jnp.linalg.norm(x_gt)) < 1e-10


def test_solve_spd_scaled_float32_robust() -> None:
    """A numerically indefinite (catastrophically cancelled) float32 reduced
    system must produce finite output, and a well-conditioned float32 system
    must stay accurate."""
    rng = onp.random.default_rng(0)
    # Near-singular: a rank-deficient PSD matrix plus tiny noise that makes it
    # slightly indefinite in float32, as catastrophic cancellation does.
    u = rng.normal(size=(20, 3))
    S_bad = (u @ u.T).astype(onp.float32)
    S_bad += rng.normal(0, 1e-6, S_bad.shape).astype(onp.float32)
    S_bad = ((S_bad + S_bad.T) / 2).astype(onp.float32)
    b = rng.normal(size=20).astype(onp.float32)
    x = _schur._solve_spd_scaled(jnp.array(S_bad), jnp.array(b))
    assert bool(jnp.all(jnp.isfinite(x))), "robust SPD solve produced non-finite values"

    # Well-conditioned float32: still accurate to ~1e-3.
    A = rng.normal(size=(20, 20)).astype(onp.float32)
    S_good = jnp.array(A @ A.T + 20 * onp.eye(20, dtype=onp.float32))
    x_gt = jnp.array(rng.normal(size=20).astype(onp.float32))
    b_good = S_good @ x_gt
    x_good = _schur._solve_spd_scaled(S_good, b_good)
    rel = float(jnp.linalg.norm(x_good - x_gt) / jnp.linalg.norm(x_gt))
    # The float32 Tikhonov floor (~2e-3 relative) bounds the attainable
    # accuracy; this is the robustness/accuracy trade-off, by design.
    assert rel < 1e-2


def test_float32_end_to_end_no_nans() -> None:
    """Full float32 solve through the (automatic) direct Schur path: zero
    NaNs."""
    jax.config.update("jax_enable_x64", False)
    try:
        problem, init = _make_ba_problem()
        _, summary = problem.solve(
            init, linear_solver="dense_cholesky", **_solve_kwargs()
        )
        costs = onp.asarray(summary.cost_history[:10])
        assert not onp.any(onp.isnan(costs)), f"NaNs in float32 cost history: {costs}"
        assert costs[-1] < costs[0]
    finally:
        jax.config.update("jax_enable_x64", True)
