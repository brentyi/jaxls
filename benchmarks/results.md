# Variable elimination (Schur complement) — results

Branch `schur-elimination-v2`: a from-scratch reimplementation of variable
elimination for bundle adjustment, measured with the matched-iteration
methodology below. Elimination is automatic by default
(`analyze(schur_elimination="auto")`): `solve()` eliminates dominant
block-diagonal variable types (chosen from the problem structure, logged) and
all three linear solvers use the reduced system — dense Cholesky and CG solve
it densely / matrix-free, and cholmod factors it sparse-directly.
`schur_elimination="off"` opts out; the "full CG"/"full dense" baselines below
use it. Hardware: Apple Silicon CPU, float64
unless noted. Absolute times are hardware-dependent; the exactness results
and the matched-iteration *orderings* are the portable claims.

**Bottom line.** Variable elimination is correct (exact Newton steps),
robust (float32 with zero NaNs), and **beats the full-CG baseline at every
matched iteration count on real bundle adjustment, on both cost and
wall-clock**: at k=30 on Ladybug-49, Schur+dense is **13× faster to a lower
cost** and Schur+CG is **3–4× faster to the same cost**. On the
well-conditioned toy problem — where there is nothing to win — both Schur
paths still run slightly faster than full CG. Two Levenberg-Marquardt fixes
discovered during this work (see below) also substantially improved *all*
solvers, including the full-CG baseline itself.

## Methodology: matched outer iterations

"Time to convergence" is confounded by termination: a solver that stops
earlier looks faster for reasons unrelated to the linear algebra. Instead we
run **exactly k LM iterations** (early termination off), for a sweep of k,
and compare (accepted cost, wall-clock) at matched k. Timing rules: one full
warmup solve per configuration with `jax.block_until_ready` (absorbs
compilation and asynchronous dispatch), minimum of 3 repeats, and the
full-CG baseline doubles as a control line for the noise floor. Reported
costs are those of the **returned (accepted) solution** — the solve summary's
cost history also contains rejected proposals, which would not be honest to
report.

Reproduce: `uv run --extra dev --extra docs python benchmarks/device_sweep.py` (raw data saved to JSON;
`--replot` regenerates plots without re-running).

## Correctness — exact reduced steps (pass)

The direct Schur path solves the same damped normal equations as a full
dense Cholesky, exactly:

- Single damped steps match an explicit full dense solve to **~1e-10
  relative** across a sweep of damping values (`test_schur_single_step_exactness`),
  including problems with multiple kept variable types.
- LM cost trajectories match full dense to ~5e-9 relative on a small BA
  problem (`test_schur_dense_exactness`); visible in the toy table below,
  where Schur+dense and full dense produce identical costs at every k.
- On Ladybug-49, the reduced step satisfies the full-system normal equations
  to `|Hd-b|/|b| ~ 1e-11`, verified against a host-side sparse solve.

## Robustness — float32 (pass)

Forming `S = H_cc − W V⁻¹ Wᵀ` is a difference of nearly-equal SPD matrices
and cancels catastrophically in float32. The reduced solve applies Jacobi
(diagonal) scaling plus a precision-adaptive Tikhonov floor (~2e-3 relative
in float32, ~1e-15 in float64 so float64 stays exact). Ladybug-49 end-to-end
in float32 (`benchmarks/float32_check.py`):

- **0 NaNs**; best cost 26,785 (float32) vs 26,781 (float64) — a 1.5e-4
  relative gap.

## Performance — Ladybug-49 (real BAL: 49 cameras, 7,776 points, 31,843 observations)

> **Note:** the table below was timed before the column-scaler splice
> (see "The column-scaling math"). Under the final scaler, matched-k
> costs improve slightly (k=30: full CG 26,768, Schur+dense 26,746,
> Schur+CG 26,767) and the orderings are unchanged; post-splice timing
> runs coincided with other load on the machine (the unchanged
> full-dense control line ran 4× slow) and were discarded rather than
> published. Re-time with `uv run --extra dev --extra docs python benchmarks/device_sweep.py` on a
> quiet machine.

Accepted cost / wall-clock at matched k (float64, min of 3, CPU):

| k  | full CG (baseline) | Schur + dense       | Schur + CG          |
|---:|--------------------|---------------------|---------------------|
| 1  | 41,432 / 0.02 s    | 42,254 / 0.03 s         | 40,919 / 0.04 s       |
| 2  | 29,466 / 0.10 s    | 28,718 / 0.06 s         | 28,824 / 0.09 s       |
| 4  | 27,206 / 0.30 s    | 27,175 / 0.10 s         | 27,212 / 0.20 s       |
| 6  | 26,964 / 0.64 s    | 26,938 / 0.13 s         | 26,961 / 0.33 s       |
| 9  | 26,890 / 1.30 s    | 26,861 / 0.19 s         | 26,888 / 0.56 s       |
| 13 | 26,862 / 2.38 s    | 26,829 / 0.28 s         | 26,860 / 0.91 s       |
| 18 | 26,840 / 3.92 s    | 26,807 / 0.37 s         | 26,839 / 1.37 s       |
| 24 | 26,822 / 5.95 s    | 26,791 / 0.50 s         | 26,821 / 1.93 s       |
| 30 | 26,810 / 8.39 s    | **26,779 / 0.62 s**     | **26,809 / 2.57 s**   |

- **Schur+dense dominates at every k from k=2 on: lower cost *and* less
  time.** At k=30 it is **13× faster** than full CG and reaches a lower cost
  than full CG ever does; it already beats full CG's best-ever cost
  (26,810 @ 8.39 s) by k=18 (26,807 @ 0.37 s) — **~22× less wall-clock to
  full CG's best cost.**
- **Schur+CG tracks full CG's cost trajectory almost exactly at ~3× less
  wall-clock** (2.57 s vs 8.39 s at k=30). Per-iteration, both CG variants
  pay more as the Eisenstat-Walker forcing term tightens, but the reduced
  441-dim system needs far fewer inner iterations than the full 23,769-dim
  one.
- Why: each outer iteration, full CG re-solves the ill-conditioned full
  system from scratch with a runaway inner-iteration count near convergence.
  The Schur path eliminates the points analytically (block-diagonal V), and
  the per-iteration cost of the direct reduced solve is flat: ~2.5 ms of
  damping-independent preparation plus ~17 ms per damping value, dominated by
  the pair-block products for `W V⁻¹ Wᵀ`.

## Performance — Toy (30 cameras, 700 points): the well-conditioned counter-case

All solvers reach the optimum (5,913.72) by k≈4; there is nothing for
conditioning to win. At k=16: Schur+dense **0.04 s**, Schur+CG **0.09 s**,
full CG 0.13 s, full dense 6.4 s. So elimination costs nothing on easy
problems (it is in fact mildly faster here too), and full dense is ~150×
slower — the usual `O(n³)` cliff that makes it infeasible at real scale.

## Two Levenberg-Marquardt fixes that this work surfaced

> **Superseded in part — see "Final configuration" at the end.** Fix 1 (the
> spliced scaler) was later found to regress downstream workloads whose
> lambdas were tuned against the legacy convention (pyroki IK, M3500 pose
> graphs) and was **reverted**; the rejected-step termination fixes plus
> Nielsen lambda escalation recover the BAL behavior the splice was built
> for. Fix 2 stands.

Both were found by asking why *exact* Newton steps were being rejected on
raw BAL data while inexact CG steps were accepted. They live in the shared
LM loop and improve every solver:

1. **Jacobian column scaling: spliced Levenberg/Marquardt damping.** See
   the derivation below. *(Later reverted; see above.)*
2. **Predicted-cost double-scaling bug.** The trust-region quality ratio
   computed the predicted residual as `A_scaled @ (scaler * delta)`, which
   applies the column scaling twice; the physical prediction is
   `A_scaled @ delta`. This mildly distorted accept/reject decisions before,
   and would have been catastrophic with the corrected scaler.

### The column-scaling math (historical: the splice was reverted)

Each LM step solves `min ‖A S δ + r‖² + λ‖δ‖²` with `S = diag(s)` and
applies `Δ = S δ`; equivalently `(AᵀA + λ S⁻²) Δ = −Aᵀr`. So choosing
`s_i` chooses the per-coordinate damping `d_i = λ / s_i²`:

- **Levenberg** (`s = 1`, `d = λ`): one λ shared across curvatures that
  differ by orders of magnitude — on raw BAL, exact Newton steps are
  rejected for ~12 straight iterations at the default λ.
- **Marquardt/Moré** (`s = 1/n`, `d = λn²`; Ceres): fixes BA, but
  amplifies weak columns without bound and re-denominates every
  existing tuned λ — pyroki's collision benchmarks then converge to
  worse local minima.
- **Legacy jaxls** (`s = (2+n)/(1+n) ∈ (1, 2]`): ≈ Levenberg, and the
  convention downstream λs were tuned against.

The splice evaluated here was whichever scales a column less:

```
s(n) = min( (2+n)/(1+n) ,  (3/2)/n )      # evaluated, then reverted
```

A min of two decreasing curves switches at their intersection, so fixing
the crossover at unit norm — the scale the Marquardt branch normalizes
*to* — pins the numerator: `c = s_legacy(1) = 3/2`. Below unit norm this
is exactly the legacy scaler; above it, exactly scale-invariant damping
`d = (4/9) λn²`.

**Why it was reverted:** "below unit norm" turned out not to protect real
downstream problems — IK Jacobian columns (norms 6–11) and pose-graph
columns land *above* the crossover, so their tuned λs were silently
re-denominated by ~50×: pyroki IK-Beam fell from 99.8% to 94.6% success
(p99 error 0.13→16.7 mm) and the M3500 g2o example converged 5× worse.
Meanwhile the splice's own motivation dissolved: with the rejected-step
termination criteria gated (see "A real LM bug" below) and Nielsen lambda
escalation, the legacy scaler reaches *better* Ladybug-49 full-CG costs
than the splice ever did. See "Final configuration".

A consequence worth stating plainly: performance numbers measured before
these fixes (including those of the previous Schur implementation on the
`schur-variable-elimination` branch, which reported full CG plateauing at
33,137) are not comparable with this table. The honest comparison is
within-run, against the same LM loop — which is what the table shows.

## Caveats and the GPU question

- The direct reduced solve is dense in the kept block: `O(n_keep²)` memory
  and `O(n_keep³)` factorization. Fine to roughly 1–2k camera tangent
  dimensions; beyond that the reduced system should go through a sparse
  Cholesky (the co-visibility pattern is already computed by the plan).
- All numbers are CPU. On GPU, dense Cholesky is a weak spot, but
  **Schur+CG is matrix-free end-to-end** — its assembly uses
  `segment_sum` (contiguous reductions, no scatter atomics) and its
  advantage comes from the reduced system's lower inner-iteration count,
  which is sequential and cannot be parallelized away. So the GPU-favored
  path is Schur+CG; validate there before relying on it.
- float64 is recommended for the direct path; float32 works (0 NaNs) with
  accuracy bounded by the deliberate ~2e-3 Tikhonov floor.

## Reproduce

```bash
uv run --extra dev --extra docs python benchmarks/device_sweep.py       # BA cost/time matrix + plots
uv run --extra dev --extra docs python benchmarks/float32_check.py      # float32 robustness (Ladybug-49)
pytest tests/test_schur.py              # correctness/validation suite
```

BAL data downloads to /tmp automatically. Raw arrays are saved to
`benchmarks/results/*.json`; `device_sweep.py --replot` regenerates plots.

---

# CPU vs GPU study and GPU assembly optimization

Added 2026-06-12. The numbers above are Apple-Silicon CPU. This section
re-runs the matched-iteration study on a second machine across **both CPU
and GPU**, and documents a GPU-driven optimization of the per-iteration
assembly. Hardware: AMD Threadripper PRO 5955WX (16C/32T) + NVIDIA RTX 4090,
float64.

> **Config note:** the measurements in this part were taken with the
> (since-reverted) spliced scaler plus the termination fixes. The
> qualitative conclusions — assembly bottleneck, broadcast-vs-GEMM, the
> termination bug, Schur-vs-full-system gaps — are config-independent;
> the shipped-configuration BA tables live in "Final configuration" below.

Harness: `benchmarks/device_sweep.py` (same matched-k methodology: exactly k
LM iterations, early termination off, warmup + min-of-3, per device via
`jax.device_put`). It compares full CG, cholmod (CPU only), Schur+dense,
Schur+CG, and full dense (small problems only).

## Where the GPU time went: batched einsum vs broadcast

Profiling one Schur+dense outer iteration on Ladybug-49/GPU (per-phase
timing) showed the cost was *not* the Cholesky factorization but the
**assembly of the reduced matrix**:

| phase             | before  | after   |
|-------------------|--------:|--------:|
| `prepare_schur`   | 2.37 ms | 0.22 ms |
| `assemble_dense_S`| 7.75 ms | 0.55 ms |
| `solve_spd_scaled`| 1.56 ms | 1.56 ms |
| full inner step   | 8.66 ms | 2.10 ms |

The assembly is built from many small batched products
(`H_cc` Gram blocks `J^T J`, the cross blocks `W`, and the pair products of
`W V^{-1} W^T`). These were written as `jnp.einsum` / `lax.dot_general`.
Each has a **tiny contraction axis** — the residual dimension (2 for a
reprojection cost) or the eliminated-variable dimension (3 for a landmark).
On the 4090, XLA lowers such a batched GEMM to a kernel that is
**15–30× slower** than the mathematically identical
broadcast-multiply-sum. Measured on the 91,243-pair product of Ladybug-49:

| form                              | time    |
|-----------------------------------|--------:|
| `einsum('ptf,psf->pts', ...)`     | 5.13 ms |
| `lax.dot_general` (batched)       | 5.13 ms |
| `sum(a[...,None,:]*b[...,:,None,...], axis)` | **0.17 ms** |

The fix (`_batched_gram`, `_batched_outer_last` in `_schur.py`) replaces the
five matrix-producing einsums with the broadcast form. It is bit-identical
to the einsum on random inputs (max diff 0.0) and changes no results: the
single damped step still matches a from-scratch full dense solve to
**2.6e-14 relative**, and all 53 tests pass. The dense inner step is now
bottlenecked by the irreducible Cholesky solve, as it should be.

A dead end worth recording: replacing the pair-list assembly with one big
dense `camera×point` matmul is **slower** (8.4 ms), because the camera×point
occupancy is only ~8% on Ladybug — the dense matmul wastes >90% of its FLOPs
on structural zeros. The sparse pair structure must be kept.

## Ladybug-49 — before/after, both devices

Wall-clock at k=30 (float64, min of 3), Schur paths only (the methods the
optimization touches):

| method        | CPU before | CPU after | GPU before | GPU after |
|---------------|-----------:|----------:|-----------:|----------:|
| Schur + dense | 1.45 s     | **1.04 s** | 0.339 s    | **0.079 s** |
| Schur + CG    | 15.5 s     | (≈ same)¹  | 0.463 s    | **0.357 s** |

¹ The CPU Schur+CG re-run shared all 32 threads with another job and timed
slower (artifact, not a regression). The CG inner loop's matrix-*vector*
einsums were measured separately and are NOT GEMM-pathological — XLA already
lowers them well, so no broadcast rewrite applies there.
**GPU Schur+dense improves 4.3×; CPU Schur+dense 1.4×.**
Returned costs are identical to the pre-change run (≤1e-10).

## Cross-device picture (optimized)

Wall-clock at k=30 on Ladybug-49 (float64), all methods:

| method        | CPU      | GPU      |
|---------------|---------:|---------:|
| Schur + dense | **1.04 s** | **0.079 s** |
| Schur + CG    | ~14.8 s  | 0.357 s  |
| cholmod       | 4.16 s   | n/a      |
| full CG       | 33.0 s   | 0.88 s   |

- **Schur+dense is the fastest path on *both* devices** — including GPU,
  correcting the earlier prediction above that "dense Cholesky is a GPU weak
  spot, so Schur+CG is the GPU-favored path." With the assembly cost removed,
  the 441×441 Cholesky is cheap on the 4090 and the dense path wins outright.
- The GPU rehabilitates full CG (33 s → 0.88 s), but it is still ~11× slower
  than GPU Schur+dense.
- cholmod remains the best *full-system* CPU path, but Schur+dense beats it 4×.

## Trafalgar-138 — the gap at scale (165,899 observations)

Trafalgar-138 (138 cameras, 44,033 points, 165,899 observations) has a
**133,341-dim full system**; elimination reduces it to **1,242** dims. The
two slowest baselines were capped once their cost had visibly plateaued
(CPU full CG at k≤9, CPU Schur+CG at k≤13) — each further point cost
minutes of wall-clock to confirm an already-flat line.

### A real LM bug, found via the first Trafalgar run

The first run of this study showed every *exact-step* solver (Schur+dense
177,142; cholmod 178,707; full CG 179,039) plateauing well above where
inexact Schur+CG kept descending (173,653). Tracing one stalled solve
showed the mechanism: a **rejected** proposal whose (untaken) step was tiny
tripped the parameter-tolerance convergence criterion, which exits the
lambda-escalation loop before lambda can rise; on rejection lambda is kept,
so every subsequent outer iteration re-proposed the bit-identical step
forever. The cost criterion had already been gated to accepted proposals
(see the LM fixes above); the parameter criterion needed the same gate.
With the one-line fix (`converged_parameters ... & accepted`,
`_solvers.py`), **every solver converges to ~173.6k** — and the exact-step
methods now reach the *lowest* costs, as they should.

Wall-clock / accepted cost after the fix (float64; the two slowest
baselines capped once their behavior was established):

| method        | device | k  | time     | cost        |
|---------------|--------|---:|---------:|------------:|
| full CG       | CPU    |  9 | 97.4 s   | 174,420 |
| full CG       | GPU    | 30 | 9.55 s   | 173,660 |
| cholmod       | CPU    | 30 | 17.9 s   | 173,659 |
| Schur + dense | CPU    | 30 | 4.93 s   | **173,632** |
| Schur + dense | GPU    | 30 | **0.229 s** | **173,632** |
| Schur + CG    | CPU    | 30 | 354 s    | 173,653 |
| Schur + CG    | GPU    | 30 | 5.33 s   | 173,653 |

- **Schur vs naive CG at scale:** at matched k=9, CPU full CG needs 97.4 s
  (174,420) while GPU Schur+dense is at a comparable cost (174,989) in
  0.070 s — a **~1,400× wall-clock gap**, up from ~30× (CPU) / ~11× (GPU)
  on Ladybug-49. Even on the same GPU, Schur+dense reaches a lower cost
  than full CG ever does at 1/40th the wall-clock (0.23 s vs 9.6 s).
- **Schur+dense is now both the fastest and the most accurate** path on
  both devices (173,632). CPU Schur+CG produces the same trajectory as GPU
  Schur+CG ~50× slower — each CG matvec is a pass over 165,899
  observations, which the GPU parallelizes and the CPU serializes.
- cholmod again the best full-system path, and again beaten by Schur+dense
  on both time (3.6× at k=30) and cost.

---

# Final configuration (the shipped state)

Added 2026-06-13. After the cross-device study above, two downstream
regressions traced to the spliced column scaler forced a configuration
change; this section records the final state and its benchmark numbers.
Per-problem A/B tables are reproducible via the benchmark suite
(`python -m benchmarks.suite --gate` against a committed baseline).

**What shipped:**

- **Legacy column scaler** (`1/(1+n) + 1`). The splice silently
  re-denominated tuned lambdas for any problem with above-unit column
  norms: pyroki IK-Beam fell from 99.8% to 94.6% success (p99 error
  0.13 → 16.7 mm) and the M3500 g2o example converged 5× worse
  (679.8 vs 137.9). Both match `main` exactly under the final config.
- **Rejected-step termination gates + AL/lambda_max fixes + stable gain
  ratio** (see "A real LM bug" above; all solvers converge ~2% lower on
  Trafalgar with the gates in place).
- **Nielsen-style accelerating lambda escalation**: on each consecutive
  rejection within an outer step, the escalation factor itself doubles, so
  a grossly under-damped lambda recovers in O(log log) tries. This is the
  rejection-side half of H.B. Nielsen (1999), "Damping Parameter in
  Marquardt's Method", IMM-REP-1999-05, DTU; the decrease-on-acceptance
  rule remains jaxls's plain halving. A no-op when the first proposal is
  accepted, so tuned solves are unaffected.
- **Broadcast-form batched products** (`utils._batched_*`) in Schur
  assembly and the block-Jacobi preconditioner (5–30× over einsum on GPU).

**Lambda guidance for bundle adjustment:** BA curvature scales put
workable uniform damping around `lambda_initial ≈ 1e1–1e3` (a three-decade
plateau); the library default 5e-4 cold-starts the exact-step paths
through a long rejection sweep. The BA tables below pass
`lambda_initial=1e2` to *every* method, so the comparison isolates the
linear solver.

## Bundle adjustment, main vs this PR (lambda_initial=1e2, float64)

GPU, accepted cost / outer LM steps / wall-clock at the largest k within
a 25 s/solve budget; "(times out)" marks rows stopped by the budget. The
baseline is jaxls@main's full-system CG — what `solve()` runs on these
problems without this PR. (CPU numbers, cholmod, and the rest come from
`benchmarks/device_sweep.py`; main's full dense crashes on an int64/int32
index bug this PR fixes.)

| problem | full CG (main) | **Schur+dense (PR)** | Schur+CG (PR) |
|---|---|---|---|
| Toy | 5,913.89 / 16 it / 0.075 s | 5,913.72 / 16 it / **0.025 s** | 5,913.72 / 16 it / 0.032 s |
| Ladybug-49 | 27,394 / 30 it / 9.50 s | **26,779 / 30 it / 0.080 s** | 27,711 / 30 it / 0.50 s |
| Trafalgar-138 | 202,122 / 6 it / 31.3 s (times out) | **173,627 / 30 it / 0.234 s** | 173,627 / 30 it / 5.22 s |

See `results/ba_comparison_gpu.png`: cost vs wall-clock per problem, one
marker per LM step starting from the shared step-0 initial cost
(running-best cost, since the unconverged full-CG baseline is non-monotone
in matched k; symlog x so t=0 and the multi-second baseline tail are both
legible).

- **Ladybug-49:** Schur+dense reaches a lower cost than full CG ever does,
  **119× faster** at matched 30 steps.
- **Trafalgar-138:** full CG times out at 6 steps far from convergence
  (CPU exceeds 650 s *per solve* past the first step); **Schur+dense
  reaches 173,627 in 0.234 s** — >130× faster at a 14% lower cost. Each
  Schur+dense step is ~40× cheaper than a full-CG step here, so it both
  takes useful steps and takes them faster.
- Toy: everything agrees at the optimum in the same step count; Schur
  paths are fastest there too.

## Reproduce (cross-device)

```bash
uv run --extra dev --extra docs python benchmarks/device_sweep.py                  # CPU+GPU, all problems
uv run --extra dev --extra docs python benchmarks/device_sweep.py --devices gpu    # GPU only
```

Requires a CUDA jaxlib for the GPU rows; falls back to CPU-only otherwise.
Raw arrays in `benchmarks/results/device_*.json`.

---

# Performance ceiling: why single-solve time is what it is

Added 2026-06-14. After the GPU-assembly and LM-fix wins above, a focused
optimization pass (target: 2x on single-solve wall-clock) profiled where the
time actually goes. The conclusion is structural and worth recording so
future work starts from it rather than re-deriving it.

## The solve is host-dispatch-bound, not compute-bound

On Trafalgar-138 (GPU, float64, lambda_initial=1e2), a 30-iteration dense
solve takes ~256 ms = ~8.5 ms/iter. But:

- **Dispatch-only time (no `block_until_ready`) equals full time** (256 ms ==
  256 ms): the host enqueue loop, not the device, is the critical path. The
  GPU is mostly idle.
- The whole solve is **0.64 GFLOP** against **1.76 GB** of memory traffic —
  it runs at ~0.01% of the 4090's float64 peak. Nothing is compute-limited.
- Each iteration compiles to **~175 fusions + ~9 custom-calls ≈ 184 kernel
  launches**, run serially (~46 µs each). Per-iteration kernel count is
  nearly identical for toy (138) and Trafalgar (175) — the overhead is
  structural, which is why on small problems the linear solve is ~1% of
  wall-clock and the other ~99% is this fixed per-iteration chain.

So the bottleneck is **the number of small serial kernels per outer step**,
each doing little work. This reframes every solver-internal optimization:

| lever | measured effect | why |
|---|---|---|
| GPU broadcast-vs-einsum assembly | ~4x on assembly (landed) | real, but assembly is ~0.5 ms of 8.5 ms/iter |
| Hoist column-norm scaler out of loop | ~2-5%/iter (landed) | one fewer scatter/iter |
| float32 Cholesky + iterative refinement | **rejected** | real Schur S is ill-conditioned (cond ~1e6); f32 refinement stalls at ~4% error |
| gather vs vmap(dynamic_slice) in multiply | ~0% | XLA lowers both to the same kernel here |
| einsum->broadcast in reduced_rhs/back_sub | ~1% | matvec einsums are not GEMM-pathological |
| analytic vs autodiff reprojection Jacobian | ~0% (cost identical) | the Jacobian is one big vmapped fusion either way; op *count* is high but it is not the dispatch bottleneck |
| remove per-iter timestamp callback | ~0.5% | JAX pipelines the callback; it is not the bottleneck |

The linear solver — where the obvious optimizations live — is **1-3% of
per-iteration wall-clock** on these problems. A faster Cholesky or CG matvec
cannot move the total.

## What would actually move it

1. **Batch/vmap across problems.** Because the GPU is idle between
   dispatches, solving N problems at once amortizes the per-iteration kernel
   chain over N. This is how downstream users already get throughput
   (pyroki's batch-2000 vmapped IK); per-problem cost collapses. The single
   highest-leverage path, and it needs no solver changes.
2. **Fewer, larger kernels per outer step.** A real single-solve 2x needs the
   per-iteration graph to compile to a handful of big fused kernels instead
   of ~184 small ones — a structural rewrite of the cost/Jacobian/assemble/
   solve chain, research-grade and risky, with uncertain payoff.
3. **Analytic Jacobians help op *count* but not this bottleneck** on GPU;
   they matter more on CPU and for compile time. Worth offering for
   op-heavy costs, but not the single-solve GPU lever it first appears to be.

The honest summary: a 2x single-solve speedup on toy/Ladybug/Trafalgar is
**not reachable from solver-internal changes** — the bottleneck is the fixed
per-iteration kernel-dispatch chain. Throughput (batching) is the productive
direction.
