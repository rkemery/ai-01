"""CI gate with two kinds of block.

Hard floors (`pii_leak:max=0`) are checked on every candidate record and fail
on a single violation. They fail closed: a record that errored before the
floor metric was computed counts as unverified, which blocks, and so does a
floor that checked no record at all (an empty candidate run).

Metric regressions compare candidate against baseline, paired by item_id,
with one rule: a drop blocks when it is significant at the 5% level, and the
line printed for it shows the statistic that rule used.

- pass/fail metric, no clustering: the exact McNemar test (p < 0.05)
- any metric with --cluster: the clustered t-test, whose 95% CI excludes 0
  exactly when p < 0.05
- numeric metric, no clustering: the 95% paired bootstrap CI excludes 0

A drop that is not significant warns (with the MDE) and does not block. For
lower-is-better metrics (`hallucination:lower`) a rise is the drop.

With on_error='exclude', a pass/fail item that errored in the candidate but not
in the baseline counts as a candidate failure, so errors can only make the
candidate look worse. Items that errored in the baseline, and errored items of
numeric metrics, are left out. As an extra guard the gate blocks if more than
`max_excluded` of the items (5% by default) errored in either run, since the
comparison then no longer covers the run. Only blocks change the exit code:
0 pass, 1 block.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from llm_eval_harness.analysis import RunComparison, compare_runs
from llm_eval_harness.records import EvalRecord, OnError, metric_value
from llm_eval_harness.stats import PairedComparison

Status = Literal["pass", "warn", "block"]
ALPHA = 0.05
DEFAULT_MAX_EXCLUDED = 0.05
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
        return self.n_checked > 0 and not self.violations and not self.unverified


@dataclass(frozen=True)
class RegressionResult:
    """`status` combines the significance rule and the limit on excluded items."""

    metric: GateMetric
    result: RunComparison
    status: Status
    excluded_limit: int

    @property
    def too_many_errors(self) -> bool:
        return self.result.n_errored > self.excluded_limit


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
    """Block a significant drop, warn on any other drop, pass otherwise.

    Significance comes from the comparison's p-value when it has one (exact
    McNemar, or the clustered t-test), else from the bootstrap CI excluding 0.
    """
    c = result.comparison
    # Express everything as "improvement", so positive is always good.
    gain = c.diff if higher_is_better else -c.diff
    if gain >= 0:
        return "pass"
    return "block" if _significant(c, higher_is_better) else "warn"


def run_gate(
    baseline: Sequence[EvalRecord],
    candidate: Sequence[EvalRecord],
    floors: Sequence[Floor] = (),
    metrics: Sequence[GateMetric] = (),
    *,
    use_clusters: bool = False,
    on_error: OnError = "raise",
    max_excluded: float = DEFAULT_MAX_EXCLUDED,
    n_boot: int = 10_000,
    seed: int = 0,
) -> GateResult:
    if not floors and not metrics:
        raise ValueError("the gate needs at least one floor or metric")
    if not 0.0 <= max_excluded < 1.0:
        raise ValueError(f"max_excluded must be in [0, 1), got {max_excluded}")
    floor_results = tuple(check_floor(candidate, floor) for floor in floors)
    regressions = []
    for metric in metrics:
        result = compare_runs(
            baseline,
            candidate,
            metric.name,
            use_clusters=use_clusters,
            on_error=on_error,
            candidate_errors_as=0.0 if metric.higher_is_better else 1.0,
            n_boot=n_boot,
            confidence=1.0 - ALPHA,
            seed=seed,
        )
        limit = math.floor(max_excluded * result.n_items + 1e-9)
        status = regression_status(result, metric.higher_is_better)
        if result.n_errored > limit:
            status = "block"
        regressions.append(RegressionResult(metric, result, status, limit))
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
            if f.n_checked == 0:
                detail += ". Blocked because no records were checked"
            lines.append(f"  [{tag}] {f.floor}: {detail}")
    if result.regressions:
        lines.append("")
        lines.append(
            "Regressions (candidate - baseline, paired by item_id). "
            f"A drop blocks when significant at the {ALPHA:.0%} level."
        )
        for r in result.regressions:
            lines.append(_regression_line(r))
    return "\n".join(lines)


def _regression_line(r: RegressionResult) -> str:
    c = r.result.comparison
    direction = "higher is better" if r.metric.higher_is_better else "lower is better"
    line = f"  [{r.status.upper()}] {r.metric.name} ({direction}): {c.diff:+.3f}, n={c.n}. "
    if c.test == "exact McNemar test" and c.mcnemar is not None:
        m = c.mcnemar
        line += (
            f"Discordant pairs {m.a_only} baseline-only, {m.b_only} candidate-only, "
            f"exact McNemar p={m.pvalue:.3f}"
        )
    elif c.test is not None and c.pvalue is not None:
        line += (
            f"95% CI [{c.low:+.3f}, {c.high:+.3f}], {c.test} p={c.pvalue:.3f} "
            f"({c.n_clusters} clusters)"
        )
    else:
        line += f"95% bootstrap CI [{c.low:+.3f}, {c.high:+.3f}]"
    gain = c.diff if r.metric.higher_is_better else -c.diff
    if gain < 0 and not _significant(c, r.metric.higher_is_better):
        if r.result.mde is not None:
            line += f". Inconclusive, the MDE at this n is about {r.result.mde:.3f}"
        else:  # only the exact McNemar MDE is None for a drop: too few discordant pairs
            line += ". Inconclusive, too few discordant pairs for 80% power at any effect size"
    if r.result.n_errored:
        res = r.result
        line += (
            f". Errored items: {res.n_errored} of {res.n_items} (candidate "
            f"{len(res.excluded_candidate)}, baseline {len(res.excluded_baseline)})"
        )
        if res.imputed:
            n_imputed = len(res.imputed)
            line += (
                f", {n_imputed} candidate-only counted as failure{'s' if n_imputed != 1 else ''}"
            )
        if res.excluded:
            line += f", {len(res.excluded)} excluded"
        if r.too_many_errors:
            line += (
                f", over the limit of {r.excluded_limit}, so the comparison no longer covers "
                "the run"
            )
        else:
            line += f", limit {r.excluded_limit}"
    return line + "."


def _significant(c: PairedComparison, higher_is_better: bool) -> bool:
    """Whether the drop direction is significant at ALPHA, by the comparison's own test."""
    if c.pvalue is not None:
        return c.pvalue < ALPHA
    return c.high < 0 if higher_is_better else c.low > 0


def _preview(ids: Sequence[str], limit: int = 5) -> str:
    shown = ", ".join(ids[:limit])
    return f"({shown}{', ...' if len(ids) > limit else ''})"
