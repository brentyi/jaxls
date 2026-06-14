# jaxls benchmarks

Two things live here:

1. **`suite/`** — the standing benchmark + regression suite. One command,
   one report, a committed baseline to catch regressions. Use this for
   day-to-day optimizer work and CI.
2. **Study scripts** — the harnesses behind the deep-dive in
   [`results.md`](results.md) (Schur elimination, GPU optimization, the
   LM/scaler investigation) and the main-vs-PR tables in
   [`regression.md`](regression.md). Keep for reproducing those specific
   figures; the suite is what you run normally.

## The suite

```bash
# Default run: the gate set (bundle_adjustment + float32), diff vs baseline.
python -m benchmarks.suite

# Fast inner loop while hill-climbing (GPU only, no slow CPU baselines, ~30s).
python -m benchmarks.suite --quick --only bundle_adjustment

# CI / regression gate: exit 1 if any metric regressed past tolerance.
python -m benchmarks.suite --gate

# Bless the current numbers as the new baseline (do this deliberately).
python -m benchmarks.suite --update-baseline

# Opt-in workloads (kept out of the deterministic gate; run on their own).
python -m benchmarks.suite --only pyroki_ik          # downstream IK regression
python -m benchmarks.suite --only example_notebooks  # ~minutes, 19 notebooks
```

The default/gate set is `bundle_adjustment` + `float32_robustness` — fast and
in-process-reliable. `pyroki_ik` (needs pyroki installed) and
`example_notebooks` (19 jupyter subprocesses) are opt-in via `--only`: both
run fine standalone but are slow and/or unreliable as co-tenant subprocesses
of a CUDA-warmed parent, so they don't gate.

Run from the repo root. Needs a CUDA jaxlib for the GPU rows (falls back to
CPU otherwise) and `tyro` (dev dependency). Outputs:
`results/suite_results.json` (raw metrics) and `results/suite_report.md`
(human-readable diff vs baseline).

### How it works

- **`metrics.py`** — the contract. Every measurement is a `Metric` with a
  value, unit, `direction` (lower/higher better), and a regression
  tolerance. The report and the gate both read this list, so a metric added
  in a workload shows up in both automatically.
- **`workloads.py`** — each workload runs a piece of jaxls and emits
  metrics. Thin wrappers over the shared kernels in `../bal.py`,
  `../device_sweep.py`, `../trace_examples.py` — one implementation, not
  two. Current workloads: `bundle_adjustment` (Schur vs full-system,
  CPU+GPU, per-solve budgets), `example_notebooks` (every docs notebook's
  headline solve), `pyroki_ik` (downstream IK A/B — catches the kind of
  regression the scaler change caused), `float32_robustness` (Ladybug
  float32 NaN/accuracy).
- **`baseline.py` / `baseline.json`** — the committed baseline and the diff
  logic. Regression = a `lower_better` metric exceeding
  `baseline * (1 + rel_tol) + abs_tol` (or the reverse for `higher_better`).
  Time tolerances are loose (machine noise); correctness tolerances tight.

### Adding a workload

Write a `(SuiteConfig) -> WorkloadResult` in `workloads.py`, append metrics,
register it in `ALL`, then `--update-baseline`. Reuse `bal.run_k_iterations`
(warmup + min-of-repeats, returns accepted cost) for timing so the
methodology stays consistent.

### Per-iteration timestamps

`SolveSummary.time_history` records host `perf_counter` per LM step via a
callback inside the solve, so one solve yields a cost-vs-time trace (no
matched-k re-solving). This is what the example traces and BA plots use; it
is also available to any jaxls user for profiling their own solves.

## Study scripts (reproduce results.md / regression.md)

| script | what it produces |
|---|---|
| `device_sweep.py` | BA cost/time matrix + per-device and 3-up plots |
| `trace_examples.py` | example cost-vs-time traces (main vs PR) |
| `make_regression_report.py` | the `regression.md` tables |
| `profile_schur.py` | per-phase Schur timing (where GPU time goes) |
| `float32_check.py` | standalone float32 robustness check |
| `matched_iters.py` / `plot_frontier.py` | the original matched-k study |

BAL data auto-downloads to `/tmp`. The A/B scripts select the jaxls under
test via `PYTHONPATH` / a git worktree; see each script's docstring.
