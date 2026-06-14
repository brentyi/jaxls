"""jaxls benchmark + regression suite — one entry point.

    python -m benchmarks.suite                 # full run, report to stdout
    python -m benchmarks.suite --quick         # fast GPU-only inner loop
    python -m benchmarks.suite --gate          # diff vs baseline, exit 1 on regression
    python -m benchmarks.suite --update-baseline
    python -m benchmarks.suite --only bundle_adjustment float32_robustness

Run from the repo root (the suite uses PYTHONPATH=src for subprocess
workloads). Requires `tyro` (already a dev dependency).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import tyro  # noqa: E402

from . import baseline as bl  # noqa: E402
from . import workloads as wl  # noqa: E402
from .metrics import Metric, WorkloadResult  # noqa: E402


@dataclasses.dataclass
class Args:
    quick: bool = False
    """Fast inner-loop tier: GPU only, fewer problems/k, no slow CPU baselines."""
    gate: bool = False
    """Regression-gate mode: diff against the committed baseline and exit 1 if
    any metric regressed beyond tolerance."""
    update_baseline: bool = False
    """Overwrite the committed baseline with this run's metrics."""
    only: tuple[str, ...] = ()
    """Run only these workloads (default: all). Choices: """ + ", ".join(wl.ALL)
    report_path: Path = Path("benchmarks/results/suite_report.md")
    results_path: Path = Path("benchmarks/results/suite_results.json")
    repeats: int = 3


def main(args: Args) -> None:
    # --quick truncates the k-ladder, so its costs differ from the full tier
    # by design; gating/updating the shared baseline from a quick run would
    # compare apples to oranges. The baseline is always a full-tier artifact.
    if args.quick and (args.gate or args.update_baseline):
        sys.exit(
            "--quick is for eyeballing during development; it is not comparable "
            "to the (full-tier) baseline. Use a full run for --gate / "
            "--update-baseline."
        )
    cfg = wl.SuiteConfig(quick=args.quick, repeats=args.repeats)
    names = args.only or wl.DEFAULT
    bad = [n for n in names if n not in wl.ALL]
    if bad:
        sys.exit(f"unknown workload(s): {bad}; choices: {list(wl.ALL)}")

    results = []
    for name in names:
        print(f"\n########## {name} ##########", flush=True)
        try:
            results.append(wl.ALL[name](cfg))
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            results.append(WorkloadResult(name=name, skipped=f"crashed: {e}"))

    meta = {
        "jaxls": __import__("jaxls").__file__,
        "devices": wl._device_list(cfg),
        "quick": args.quick,
        "baseline": str(bl.BASELINE_PATH.name) if bl.BASELINE_PATH.exists() else "none",
    }
    run_json = bl.results_to_json(results, meta)
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(run_json, indent=2))
    print(f"\nwrote {args.results_path}")

    if args.update_baseline:
        bl.BASELINE_PATH.write_text(json.dumps(run_json, indent=2))
        print(f"updated baseline: {bl.BASELINE_PATH}")
        return

    current = {k: Metric.from_json(k, v) for k, v in run_json["metrics"].items()}
    baseline = bl.load_metrics(bl.BASELINE_PATH) if bl.BASELINE_PATH.exists() else {}
    diffs = bl.diff_against_baseline(current, baseline)
    report = bl.render_report(diffs, run_json["skipped"], meta)
    args.report_path.write_text(report)
    print(f"wrote {args.report_path}\n")
    print(report)

    if args.gate:
        n = sum(d.regressed for d in diffs)
        if not baseline:
            sys.exit("gate mode but no baseline committed; run --update-baseline first")
        sys.exit(1 if n else 0)


if __name__ == "__main__":
    main(tyro.cli(Args))
