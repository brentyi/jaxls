"""Load/save run JSON, diff against a committed baseline, render the report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .metrics import Metric, WorkloadResult

SUITE_DIR = Path(__file__).resolve().parent
BASELINE_PATH = SUITE_DIR / "baseline.json"


def results_to_json(results: list[WorkloadResult], meta: dict) -> dict:
    metrics: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for r in results:
        if r.skipped is not None:
            skipped[r.name] = r.skipped
        for m in r.metrics:
            metrics[m.key] = m.to_json()
    return {"meta": meta, "metrics": metrics, "skipped": skipped}


def load_metrics(path: Path) -> dict[str, Metric]:
    data = json.loads(path.read_text())
    return {k: Metric.from_json(k, v) for k, v in data.get("metrics", {}).items()}


@dataclass
class Diff:
    key: str
    current: Metric
    baseline: Metric | None
    regressed: bool

    @property
    def rel_change(self) -> float | None:
        if self.baseline is None or self.baseline.value == 0:
            return None
        return (self.current.value - self.baseline.value) / abs(self.baseline.value)


def diff_against_baseline(
    current: dict[str, Metric], baseline: dict[str, Metric]
) -> list[Diff]:
    diffs = []
    for key, cur in current.items():
        base = baseline.get(key)
        diffs.append(Diff(key, cur, base, cur.regressed(base) if base else False))
    return diffs


def render_report(diffs: list[Diff], skipped: dict[str, str], meta: dict) -> str:
    lines = [
        "# jaxls benchmark suite report",
        "",
        f"- jaxls: `{meta.get('jaxls', '?')}`",
        f"- devices: {meta.get('devices', '?')}  | quick: {meta.get('quick', False)}",
        f"- baseline: `{meta.get('baseline', 'none')}`",
        "",
    ]
    regressions = [d for d in diffs if d.regressed]
    if regressions:
        lines += [f"## ⚠️ {len(regressions)} regression(s)", ""]
        lines += ["| metric | baseline | current | change |", "|---|---:|---:|---:|"]
        for d in regressions:
            rc = f"{d.rel_change:+.1%}" if d.rel_change is not None else "—"
            lines.append(
                f"| {d.key} | {d.baseline.value:.6g} | {d.current.value:.6g} | {rc} |"
            )
        lines.append("")
    else:
        lines += ["## ✅ no regressions", ""]

    lines += [
        "## all metrics",
        "",
        "| metric | current | baseline | change | unit |",
        "|---|---:|---:|---:|---|",
    ]
    for d in sorted(diffs, key=lambda x: x.key):
        b = f"{d.baseline.value:.6g}" if d.baseline else "—"
        rc = f"{d.rel_change:+.1%}" if d.rel_change is not None else "—"
        flag = " ⚠️" if d.regressed else ""
        note = f" _{d.current.notes}_" if d.current.notes else ""
        lines.append(
            f"| {d.key}{flag} | {d.current.value:.6g} | {b} | {rc} | "
            f"{d.current.unit}{note} |"
        )
    if skipped:
        lines += ["", "## skipped workloads", ""]
        for name, why in skipped.items():
            lines.append(f"- **{name}**: {why}")
    return "\n".join(lines) + "\n"
