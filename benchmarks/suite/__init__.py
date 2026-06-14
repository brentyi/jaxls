"""jaxls benchmark + regression suite.

A single entry point (`python -m benchmarks.suite`) that runs a set of
workloads, reduces each to named scalar metrics, and either:

  - writes a results JSON and a markdown report (reporting mode), or
  - diffs the run against a committed baseline JSON and exits nonzero if any
    metric regressed beyond its tolerance (gate mode, for CI / hill climbing).

Workloads (see `workloads.py`): bundle-adjustment matrix (Schur vs
full-system, CPU+GPU), example-notebook convergence, pyroki IK downstream,
and float32 robustness. The metric layer (`metrics.py`) is the contract:
add a metric there and both the report and the regression gate pick it up.
"""
