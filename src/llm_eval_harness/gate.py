"""CI gate with two kinds of block.

Hard floors (`pii_leak:max=0`) are checked on every candidate record and fail
on a single violation. They fail closed: a record that errored before the
floor metric was computed counts as unverified, which also blocks.

Metric regressions compare candidate against baseline, paired by item_id. For
a higher-is-better metric with diff = candidate - baseline:

- block: the upper bound of the 95% CI of diff is below 0 (the drop is real)
- warn:  diff < 0 but the CI reaches 0 (inconclusive, maybe a drop)
- pass:  otherwise

Lower-is-better metrics (`hallucination:lower`) use the mirrored rule. Only
blocks change the exit code: 0 pass, 1 block.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from llm_eval_harness.analysis import RunComparison, compare_runs
from llm_eval_harness.records import EvalRecord, OnError, metric_value

Status = Literal["pass", "warn", "block"]
_FLOOR = re.compile(r"^([A-Za-z0-9_.-]+):(max|min)=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)$")
_METRIC = re.compile(r"^([A-Za-z0-9_.-]+)(?::(higher|lower))?$")


@dataclass(frozen=True)
class Floor:
    metric: str
    bound: Literal["max", "min"]
    threshold: float

    @classmethod
    def parse(cls, spec: str) -> Floor:
        match = _FLOOR.match(spec.strip())
        if not match:
            raise ValueError(f"bad floor {spec!r}, expected NAME:max=VALUE or NAME:min=VALUE")
        name, bound, value = match.groups()
        return cls(metric=name, bound="max" if bound == "max" else "min", threshold=float(value))

    def violated_by(self, value: float) -> bool:
        return value > self.threshold if self.bound == "max" else value < self.threshold

    def __str__(self) -> str:
        return f"{self.metric} {self.bound}={self.threshold:g}"


@dataclass(frozen=True)
class GateMetric:
    name: str
    higher_is_better: bool = True

    @classmethod
    def parse(cls, spec: str) -> GateMetric:
        match = _METRIC.match(spec.strip())
        if not match:
            raise ValueError(f"bad metric {spec!r}, expected NAME, NAME:higher or NAME:lower")
        name, direction = match.groups()
        return cls(name=name, higher_is_better=direction != "lower")


@dataclass(frozen=True)
class FloorResult:
    floor: Floor
    n_checked: int
    violations: tuple[str, ...]
    unverified: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations and not self.unverified


@dataclass(frozen=True)
class RegressionResult:
    metric: GateMetric
    result: RunComparison
    status: Status


@dataclass(frozen=True)
class GateResult:
    floors: tuple[FloorResult, ...]
    regressions: tuple[RegressionResult, ...]

    @property
    def blocked(self) -> bool:
        return any(not f.passed for f in self.floors) or any(
            r.status == "block" for r in self.regressions
        )

    @property
    def exit_code(self) -> int:
        return 1 if self.blocked else 0


def check_floor(candidate: Sequence[EvalRecord], floor: Floor) -> FloorResult:
    """Check one floor on every record. Errored records without the metric are unverified."""
    violations: list[str] = []
    unverified: list[str] = []
    for record in candidate:
        value = metric_value(record, floor.metric)
        if value is None:
            unverified.append(record.item_id)
        elif floor.violated_by(float(value)):
            violations.append(record.item_id)
    return FloorResult(
        floor=floor,
        n_checked=len(candidate) - len(unverified),
        violations=tuple(sorted(violations)),
        unverified=tuple(sorted(unverified)),
    )


def regression_status(result: RunComparison, higher_is_better: bool) -> Status:
    c = result.comparison
    # Express everything as "improvement", so positive is always good.
    gain, gain_high = (c.diff, c.high) if higher_is_better else (-c.diff, -c.low)
    if gain_high < 0:
        return "block"
    if gain < 0:
        return "warn"
    return "pass"


def run_gate(
    baseline: Sequence[EvalRecord],
    candidate: Sequence[EvalRecord],
    floors: Sequence[Floor] = (),
    metrics: Sequence[GateMetric] = (),
    *,
    use_clusters: bool = False,
    on_error: OnError = "raise",
    n_boot: int = 10_000,
    seed: int = 0,
) -> GateResult:
    if not floors and not metrics:
        raise ValueError("the gate needs at least one floor or metric")
    floor_results = tuple(check_floor(candidate, floor) for floor in floors)
    regressions = []
    for metric in metrics:
        result = compare_runs(
            baseline,
            candidate,
            metric.name,
            use_clusters=use_clusters,
            on_error=on_error,
            n_boot=n_boot,
            confidence=0.95,
            seed=seed,
        )
        regressions.append(
            RegressionResult(metric, result, regression_status(result, metric.higher_is_better))
        )
    return GateResult(floors=floor_results, regressions=tuple(regressions))


def format_gate(result: GateResult) -> str:
    """Plain-text summary for CI logs."""
    lines = [f"llm-eval gate: {'BLOCK' if result.blocked else 'PASS'}"]
    if result.floors:
        lines.append("")
        lines.append("Hard floors (any single violation blocks)")
        for f in result.floors:
            tag = "PASS" if f.passed else "FAIL"
            detail = f"{len(f.violations)} violations in {f.n_checked} records"
            if f.violations:
                detail += f" {_preview(f.violations)}"
            if f.unverified:
                detail += f", {len(f.unverified)} unverified (errored) {_preview(f.unverified)}"
            lines.append(f"  [{tag}] {f.floor}: {detail}")
    if result.regressions:
        lines.append("")
        lines.append("Regressions (candidate - baseline, paired by item_id, 95% bootstrap CI)")
        for r in result.regressions:
            c = r.result.comparison
            direction = "higher is better" if r.metric.higher_is_better else "lower is better"
            line = (
                f"  [{r.status.upper()}] {r.metric.name} ({direction}): {c.diff:+.3f} "
                f"[{c.low:+.3f}, {c.high:+.3f}] n={c.n}"
            )
            if c.mcnemar is not None:
                line += f" McNemar p={c.mcnemar.pvalue:.3f}"
            if r.status == "warn":
                mde = "n/a" if r.result.mde is None else f"{r.result.mde:.3f}"
                line += f". Inconclusive, the MDE at this n is about {mde}"
            if r.result.excluded:
                line += f". Errored items excluded: {len(r.result.excluded)}"
            lines.append(line)
    return "\n".join(lines)


def _preview(ids: Sequence[str], limit: int = 5) -> str:
    shown = ", ".join(ids[:limit])
    return f"({shown}{', ...' if len(ids) > limit else ''})"
