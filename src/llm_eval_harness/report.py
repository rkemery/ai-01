"""Markdown rendering for README results sections."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llm_eval_harness.analysis import MetricSummary, RunComparison
from llm_eval_harness.calibration import CorrectedPassRate, JudgeAgreement
from llm_eval_harness.stats import Interval, PassKResult


def fmt_value(value: float, binary: bool) -> str:
    return f"{value * 100:.1f}%" if binary else f"{value:.4g}"


def fmt_interval(interval: Interval, binary: bool) -> str:
    return f"{fmt_value(interval.low, binary)} to {fmt_value(interval.high, binary)}"


def fmt_diff(value: float, binary: bool) -> str:
    return f"{value * 100:+.1f} pts" if binary else f"{value:+.4g}"


def results_table(summaries: Sequence[MetricSummary]) -> str:
    """| Metric | Value | 95% CI | n |, one row per metric."""
    rows = ["| Metric | Value | 95% CI | n |", "|---|---|---|---|"]
    for s in summaries:
        i = s.interval
        rows.append(
            f"| {s.metric} | {fmt_value(i.estimate, s.binary)} | "
            f"{fmt_interval(i, s.binary)} | {i.n} |"
        )
    return "\n".join(rows)


def mde_line(summaries: Sequence[MetricSummary]) -> str:
    """One line with each metric's MDE against an independent run of the same size."""
    parts = []
    for s in summaries:
        mde = s.mde_vs_same_size_run
        parts.append(f"{s.metric} {'n/a' if mde is None else fmt_diff(mde, s.binary).lstrip('+')}")
    return (
        "MDE against another run of the same size (unpaired, 80% power, alpha 0.05): "
        + ", ".join(parts)
        + ". A paired comparison on the same items usually detects less."
    )


def methods_line(summaries: Sequence[MetricSummary]) -> str:
    """Which CI method each metric used, plus any items excluded for errors."""
    by_method: dict[str, list[str]] = {}
    for s in summaries:
        by_method.setdefault(s.interval.method, []).append(s.metric)
    excluded = {s.metric: len(s.excluded) for s in summaries if s.excluded}
    line = (
        "CI method: "
        + ". ".join(f"{method} for {', '.join(metrics)}" for method, metrics in by_method.items())
        + "."
    )
    if excluded:
        detail = ", ".join(f"{n} from {m}" for m, n in excluded.items())
        line += f" Excluded errored items: {detail}."
    return line


def comparison_table(results: Sequence[RunComparison]) -> str:
    rows = [
        "| Metric | Baseline | Candidate | Diff | 95% CI | McNemar p | MDE | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        c = r.comparison
        b = r.binary
        pvalue = "n/a" if c.mcnemar is None else f"{c.mcnemar.pvalue:.3f}"
        mde = "n/a" if r.mde is None else fmt_diff(r.mde, b).lstrip("+")
        rows.append(
            f"| {r.metric} | {fmt_value(c.baseline_mean, b)} | {fmt_value(c.candidate_mean, b)} | "
            f"{fmt_diff(c.diff, b)} | {fmt_diff(c.low, b)} to {fmt_diff(c.high, b)} | "
            f"{pvalue} | {mde} | {c.n} |"
        )
    return "\n".join(rows)


def agreement_table(agreements: Sequence[tuple[str, JudgeAgreement]]) -> str:
    rows = ["| Check | n | TPR | TNR | Cohen's kappa |", "|---|---|---|---|---|"]
    for check, a in agreements:
        rows.append(
            f"| {check} | {a.n} | {_rate_ci(a.tpr)} | {_rate_ci(a.tnr)} | "
            f"{a.kappa.estimate:.2f} ({a.kappa.low:.2f} to {a.kappa.high:.2f}) |"
        )
    return "\n".join(rows)


def corrected_table(results: Sequence[tuple[str, CorrectedPassRate]]) -> str:
    rows = [
        "| Check | Judge pass rate | Corrected | 95% CI (corrected) | n |",
        "|---|---|---|---|---|",
    ]
    for check, r in results:
        rows.append(
            f"| {check} | {fmt_value(r.observed.estimate, True)} | "
            f"{fmt_value(r.corrected.estimate, True)} | {fmt_interval(r.corrected, True)} | "
            f"{r.observed.n} |"
        )
    return "\n".join(rows)


def pass_k_table(result: PassKResult, metric: str) -> str:
    rows = ["| Metric | k | Value | 95% CI | tasks |", "|---|---|---|---|---|"]
    for name, interval in (
        (f"{metric} pass^k", result.pass_hat_k),
        (f"{metric} pass@k", result.pass_at_k),
    ):
        rows.append(
            f"| {name} | {result.k} | {fmt_value(interval.estimate, True)} | "
            f"{fmt_interval(interval, True)} | {result.n_tasks} |"
        )
    return "\n".join(rows)


def replace_section(text: str, name: str, body: str) -> str:
    """Replace the text between `<!-- name:start -->` and `<!-- name:end -->`."""
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"expected exactly one {start} and one {end} marker")
    head, rest = text.split(start)
    if end in head:
        raise ValueError(f"{end} appears before {start}")
    _, tail = rest.split(end)
    return f"{head}{start}\n{body.strip()}\n{end}{tail}"


def write_section(path: str | Path, name: str, body: str) -> bool:
    """Rewrite one marked section of a file. Returns True if the file changed."""
    path = Path(path)
    old = path.read_text(encoding="utf-8")
    new = replace_section(old, name, body)
    if new != old:
        path.write_text(new, encoding="utf-8")
    return new != old


def _rate_ci(interval: Interval) -> str:
    return f"{fmt_value(interval.estimate, True)} ({fmt_interval(interval, True)})"
