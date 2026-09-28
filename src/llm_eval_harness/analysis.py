"""Record-level analysis: `stats.py` applied to lists of `EvalRecord`."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from llm_eval_harness.records import (
    EvalRecord,
    MetricColumn,
    OnError,
    RecordError,
    group_by_run,
    metric_column,
)
from llm_eval_harness.stats import (
    Interval,
    PairedComparison,
    PassKResult,
    clustered_se,
    mde_from_se,
    mde_paired_binary,
    mean_ci,
    paired_bootstrap,
    pass_k,
    wilson_interval,
    wilson_interval_clustered,
)


@dataclass(frozen=True)
class MetricSummary:
    """One metric from one run with its CI.

    `se` is the standard error behind the interval (the binomial SE for a
    Wilson interval), used for the MDE against another run of the same size.
    """

    metric: str
    run_id: str
    interval: Interval
    se: float
    binary: bool
    excluded: tuple[str, ...]

    @property
    def mde_vs_same_size_run(self) -> float | None:
        """Unpaired MDE against an independent run with the same n and variance.

        MDE = (z_{1 - alpha/2} + z_{0.8}) * sqrt(2) * SE. For an unclustered
        pass rate this equals `stats.mde_two_proportions(n, n, p)`. None when
        SE is 0 (for example a 100% pass rate), where the formula says nothing.
        """
        return None if self.se == 0 else mde_from_se(math.sqrt(2.0) * self.se)


@dataclass(frozen=True)
class RunComparison:
    """Candidate vs baseline on one metric, paired by item_id."""

    metric: str
    baseline_run: str
    candidate_run: str
    comparison: PairedComparison
    binary: bool
    mde: float | None
    excluded: tuple[str, ...]


def summarize_metric(
    records: Sequence[EvalRecord],
    metric: str,
    *,
    use_clusters: bool = False,
    on_error: OnError = "raise",
    confidence: float = 0.95,
) -> MetricSummary:
    """Point estimate and CI for one metric of one run.

    Pass rates get a Wilson interval, with the design-effect sample size when
    clustered. Continuous metrics get mean +/- z * SE with Miller's clustered SE.
    """
    column = metric_column(records, metric, on_error=on_error)
    values = np.fromiter(column.values.values(), dtype=np.float64)
    clusters = _cluster_list(column, list(column.values)) if use_clusters else None
    if column.binary and clusters is None:
        interval = wilson_interval(int(values.sum()), values.size, confidence)
        p = interval.estimate
        se = math.sqrt(p * (1.0 - p) / values.size)
    elif column.binary:
        interval = wilson_interval_clustered(values, clusters, confidence)
        se = clustered_se(values, clusters)
    else:
        interval = mean_ci(values, clusters, confidence)
        se = clustered_se(values, clusters)
    return MetricSummary(
        metric=metric,
        run_id=column.run_id,
        interval=interval,
        se=se,
        binary=column.binary,
        excluded=column.excluded,
    )


def compare_runs(
    baseline: Sequence[EvalRecord],
    candidate: Sequence[EvalRecord],
    metric: str,
    *,
    use_clusters: bool = False,
    on_error: OnError = "raise",
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> RunComparison:
    """Paired comparison of two runs on the same items.

    Both runs must cover the same item_ids. With on_error='exclude', an item
    that errored in either run is dropped from both and listed in `excluded`.

    The MDE uses `mde_paired_binary` at the observed discordant rate for an
    unclustered pass rate, and (z_{1 - alpha/2} + z_{0.8}) * SE(mean diff) with
    the clustered SE otherwise. It is None when no pair disagrees.
    """
    a = metric_column(baseline, metric, on_error=on_error)
    b = metric_column(candidate, metric, on_error=on_error)
    if a.binary != b.binary:
        raise RecordError(f"{metric!r} is binary in one run and numeric in the other")
    _check_same_items(a, b)
    excluded = tuple(sorted(set(a.excluded) | set(b.excluded)))
    ids = sorted(set(a.values) & set(b.values))
    if len(ids) < 2:
        raise RecordError(f"need at least 2 paired items for {metric!r}, got {len(ids)}")
    clusters = None
    if use_clusters:
        clusters = _cluster_list(a, ids)
        if clusters != _cluster_list(b, ids):
            raise RecordError("baseline and candidate disagree on item clusters")
    x = np.array([a.values[i] for i in ids])
    y = np.array([b.values[i] for i in ids])
    comparison = paired_bootstrap(x, y, clusters, n_boot=n_boot, confidence=confidence, seed=seed)
    return RunComparison(
        metric=metric,
        baseline_run=a.run_id,
        candidate_run=b.run_id,
        comparison=comparison,
        binary=a.binary,
        mde=_paired_mde(y - x, clusters, comparison),
        excluded=excluded,
    )


def pass_k_from_records(
    records: Sequence[EvalRecord],
    metric: str,
    k: int,
    *,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> PassKResult:
    """pass^k and pass@k where each run_id is one trial and item_id is the task.

    Every trial must have a boolean score for `metric`. An errored trial raises
    rather than being dropped, since dropping failures would inflate pass^k.
    """
    successes: dict[str, list[bool]] = {}
    for run_records in group_by_run(records).values():
        column = metric_column(run_records, metric, on_error="raise")
        if not column.binary:
            raise RecordError(f"pass^k needs a boolean metric, {metric!r} is numeric")
        for item_id, value in column.values.items():
            successes.setdefault(item_id, []).append(value == 1.0)
    counts = [(len(trials), sum(trials)) for _, trials in sorted(successes.items())]
    return pass_k(counts, k, confidence=confidence, n_boot=n_boot, seed=seed)


def _paired_mde(
    diffs: np.ndarray, clusters: list[str | None] | None, comparison: PairedComparison
) -> float | None:
    if comparison.mcnemar is not None and clusters is None:
        discordant_rate = comparison.mcnemar.discordant / comparison.n
        if discordant_rate == 0:
            return None
        return mde_paired_binary(comparison.n, discordant_rate)
    se = clustered_se(diffs, clusters)
    return None if se == 0 else mde_from_se(se)


def _check_same_items(a: MetricColumn, b: MetricColumn) -> None:
    ids_a = set(a.values) | set(a.excluded)
    ids_b = set(b.values) | set(b.excluded)
    if ids_a == ids_b:
        return
    only_a = sorted(ids_a - ids_b)
    only_b = sorted(ids_b - ids_a)
    raise RecordError(
        f"runs {a.run_id!r} and {b.run_id!r} cover different items: "
        f"{len(only_a)} only in baseline {only_a[:5]}, {len(only_b)} only in candidate {only_b[:5]}"
    )


def _cluster_list(column: MetricColumn, ids: Sequence[str]) -> list[str | None]:
    clusters = [column.clusters[i] for i in ids]
    if all(c is None for c in clusters):
        raise RecordError(
            f"clustering was requested but run {column.run_id!r} has no cluster labels"
        )
    return clusters
