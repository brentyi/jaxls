"""The metric contract: a flat namespace of named scalar measurements.

Every workload emits `Metric`s. A metric has a value, a unit, and a
`direction` (is lower or higher better?) plus a default regression
tolerance. The regression gate (`baseline.py`) and the report
(`report.py`) both consume this same list, so adding a measurement in one
place makes it show up in the report and the gate automatically.

Metric `key`s are stable identifiers ("ladybug49.gpu.schur_dense.cost") —
they are the JSON keys and the baseline keys, so keep them deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Metric:
    key: str
    value: float
    unit: str = ""
    direction: Literal["lower_better", "higher_better", "neutral"] = "lower_better"
    rel_tol: float = 0.10
    """Fractional regression tolerance: a `lower_better` metric is flagged
    when it grows past value*(1+rel_tol). Time metrics are noisy, so they
    default looser than correctness metrics (set per-metric at creation)."""
    abs_tol: float = 0.0
    """Absolute slack added to the tolerance band (for near-zero values)."""
    notes: str = ""

    def regressed(self, baseline: "Metric") -> bool:
        """True if `self` is worse than `baseline` beyond tolerance."""
        if self.direction == "neutral":
            return False
        band = abs(baseline.value) * self.rel_tol + self.abs_tol
        if self.direction == "lower_better":
            return self.value > baseline.value + band
        return self.value < baseline.value - band

    def to_json(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "direction": self.direction,
            "rel_tol": self.rel_tol,
            "abs_tol": self.abs_tol,
            "notes": self.notes,
        }

    @staticmethod
    def from_json(key: str, d: dict) -> "Metric":
        return Metric(
            key=key,
            value=d["value"],
            unit=d.get("unit", ""),
            direction=d.get("direction", "lower_better"),
            rel_tol=d.get("rel_tol", 0.10),
            abs_tol=d.get("abs_tol", 0.0),
            notes=d.get("notes", ""),
        )


@dataclass
class WorkloadResult:
    """Output of one workload: its metrics plus optional artifact paths
    (plots, raw JSON) for the report."""

    name: str
    metrics: list[Metric] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    skipped: str | None = None  # reason, if the workload could not run
