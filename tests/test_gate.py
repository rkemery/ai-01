from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import record, run

from llm_eval_harness.cli import main
from llm_eval_harness.gate import (
    Floor,
    GateMetric,
    check_floor,
    format_gate,
    run_gate,
)
from llm_eval_harness.records import write_records


def test_floor_parsing() -> None:
    assert Floor.parse("pii_leak:max=0") == Floor("pii_leak", "max", 0.0)
    assert Floor.parse("answer_rate:min=0.95") == Floor("answer_rate", "min", 0.95)
    for bad in ("pii_leak", "pii_leak:max", "pii_leak:lt=0", "pii leak:max=0"):
        with pytest.raises(ValueError, match="bad floor"):
            Floor.parse(bad)


def test_metric_parsing() -> None:
    assert GateMetric.parse("correct") == GateMetric("correct", True)
    assert GateMetric.parse("hallucination:lower") == GateMetric("hallucination", False)
    with pytest.raises(ValueError, match="bad metric"):
        GateMetric.parse("correct:sideways")


def test_single_floor_violation_fails() -> None:
    records = [record(f"q{i}", scores={"pii_leak": False}) for i in range(50)]
    records.append(record("q50", scores={"pii_leak": True}))
    result = check_floor(records, Floor.parse("pii_leak:max=0"))
    assert not result.passed
    assert result.violations == ("q50",)


def test_floor_fails_closed_on_errored_records() -> None:
    records = [record("q1", scores={"pii_leak": False}), record("q2", scores={}, error="timeout")]
    result = check_floor(records, Floor.parse("pii_leak:max=0"))
    assert result.violations == ()
    assert result.unverified == ("q2",)
    assert not result.passed


def _paired(n: int, p_base: float, p_cand: float, seed: int = 0) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    ids = [f"q{i:03d}" for i in range(n)]
    base = run("base", {i: bool(x < p_base) for i, x in zip(ids, u, strict=True)})
    cand = run("cand", {i: bool(x < p_cand) for i, x in zip(ids, u, strict=True)})
    return base, cand


def test_clear_regression_blocks() -> None:
    base, cand = _paired(200, 0.8, 0.6)
    result = run_gate(base, cand, metrics=[GateMetric("correct")], n_boot=2000)
    (reg,) = result.regressions
    assert reg.status == "block"
    assert reg.result.comparison.high < 0
    assert result.exit_code == 1
    assert "[BLOCK] correct" in format_gate(result)


def test_no_change_passes() -> None:
    base, _ = _paired(50, 0.7, 0.7)
    cand = [record(r.item_id, "cand", scores=r.scores) for r in base]
    result = run_gate(base, cand, metrics=[GateMetric("correct")], n_boot=500)
    assert result.regressions[0].status == "pass"
    assert result.exit_code == 0


def test_small_drop_on_few_items_warns() -> None:
    base, cand = _paired(30, 0.7, 0.66)
    result = run_gate(base, cand, metrics=[GateMetric("correct")], n_boot=2000)
    (reg,) = result.regressions
    assert reg.result.comparison.diff < 0 <= reg.result.comparison.high
    assert reg.status == "warn"
    assert result.exit_code == 0
    assert "Inconclusive" in format_gate(result)


def test_lower_is_better_is_mirrored() -> None:
    base, cand = _paired(200, 0.1, 0.35)  # candidate hallucinates much more
    rename = {"correct": "hallucination"}
    base = [
        record(r.item_id, "base", scores={rename["correct"]: r.scores["correct"]}) for r in base
    ]
    cand = [
        record(r.item_id, "cand", scores={rename["correct"]: r.scores["correct"]}) for r in cand
    ]
    worse = run_gate(base, cand, metrics=[GateMetric("hallucination", False)], n_boot=2000)
    assert worse.regressions[0].status == "block"
    better = run_gate(cand, base, metrics=[GateMetric("hallucination", False)], n_boot=2000)
    assert better.regressions[0].status == "pass"


def test_gate_requires_matching_items() -> None:
    base = run("base", {"q1": True, "q2": True})
    cand = run("cand", {"q1": True, "q3": True})
    with pytest.raises(ValueError, match="cover different items"):
        run_gate(base, cand, metrics=[GateMetric("correct")])


def test_gate_needs_something_to_check() -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_gate([], [])


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base, cand = _paired(200, 0.8, 0.6)
    write_records(tmp_path / "base.jsonl", base)
    write_records(tmp_path / "cand.jsonl", cand)
    args = ["gate", "--baseline", str(tmp_path / "base.jsonl"), "--n-boot", "1000"]
    assert main([*args, "--candidate", str(tmp_path / "cand.jsonl"), "--metric", "correct"]) == 1
    assert "llm-eval gate: BLOCK" in capsys.readouterr().out
    assert main([*args, "--candidate", str(tmp_path / "base.jsonl"), "--metric", "correct"]) == 0
    leaky = [record("q000", "cand", scores={"correct": True, "pii_leak": True})]
    write_records(tmp_path / "leaky.jsonl", leaky)
    assert (
        main([*args, "--candidate", str(tmp_path / "leaky.jsonl"), "--floor", "pii_leak:max=0"])
        == 1
    )
    assert main([*args, "--candidate", str(tmp_path / "missing.jsonl"), "--metric", "correct"]) == 2
    assert "error" in capsys.readouterr().err


# Review findings: the gate must fail closed, and its decision must match what it prints.


def test_floor_blocks_when_no_record_was_checked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = check_floor([], Floor.parse("pii_leak:max=0"))
    assert result.n_checked == 0
    assert not result.passed
    write_records(tmp_path / "base.jsonl", [record("q1", "base", scores={"pii_leak": False})])
    write_records(tmp_path / "empty.jsonl", [])
    args = ["gate", "--baseline", str(tmp_path / "base.jsonl")]
    args += ["--candidate", str(tmp_path / "empty.jsonl"), "--floor", "pii_leak:max=0"]
    assert main(args) == 1
    assert "no records were checked" in capsys.readouterr().out


def _half_errored() -> tuple[list, list]:
    base = [record(f"q{i:02d}", "base", scores={"correct": True}) for i in range(40)]
    cand = [record(f"q{i:02d}", "cand", scores={"correct": True}) for i in range(20)]
    cand += [record(f"q{i:02d}", "cand", scores={}, error="timeout") for i in range(20, 40)]
    return base, cand


def test_exclude_mode_blocks_when_too_many_items_errored() -> None:
    base, cand = _half_errored()
    result = run_gate(base, cand, metrics=[GateMetric("correct")], on_error="exclude", n_boot=200)
    (reg,) = result.regressions
    assert reg.status == "block"
    assert result.exit_code == 1
    text = format_gate(result)
    assert "20 candidate-only counted as failures, over the limit of 2" in text


def test_exclude_mode_allows_a_few_errors_and_reports_them() -> None:
    base = [record(f"q{i:02d}", "base", scores={"correct": True}) for i in range(40)]
    cand = [record(f"q{i:02d}", "cand", scores={"correct": True}) for i in range(39)]
    cand.append(record("q39", "cand", scores={}, error="timeout"))
    result = run_gate(base, cand, metrics=[GateMetric("correct")], on_error="exclude", n_boot=200)
    # The error counts as one failure: a drop, but far from significant.
    assert result.regressions[0].status == "warn"
    assert result.exit_code == 0
    assert "1 of 40 (candidate 1, baseline 0), 1 candidate-only counted as failures, limit 2" in (
        format_gate(result)
    )


def test_four_of_forty_regressions_warn_with_the_mcnemar_p_value() -> None:
    """The review case: [BLOCK] was printed next to McNemar p = 0.125."""
    base = run("base", {f"q{i:02d}": True for i in range(40)})
    cand = run("cand", {f"q{i:02d}": i >= 4 for i in range(40)})
    result = run_gate(base, cand, metrics=[GateMetric("correct")], n_boot=2000)
    (reg,) = result.regressions
    assert reg.status == "warn"
    line = format_gate(result).splitlines()[-1]
    assert line.startswith("  [WARN] correct")
    assert "4 baseline-only, 0 candidate-only, exact McNemar p=0.125" in line


@pytest.mark.parametrize("clustered", [False, True])
def test_binary_block_decision_always_matches_the_printed_p_value(clustered: bool) -> None:
    rng = np.random.default_rng(12)
    labels = {f"q{i:02d}": f"c{i % 8}" for i in range(40)} if clustered else None
    for _ in range(60):
        u = rng.uniform(size=40)
        p_base, p_cand = rng.uniform(0.5, 0.9), rng.uniform(0.4, 0.9)
        base = run(
            "base",
            {k: bool(x < p_base) for k, x in zip(labels or ITEMS, u, strict=True)},
            clusters=labels,
        )
        cand = run(
            "cand",
            {k: bool(x < p_cand) for k, x in zip(labels or ITEMS, u, strict=True)},
            clusters=labels,
        )
        if all(r.scores == c.scores for r, c in zip(base, cand, strict=True)):
            continue
        result = run_gate(
            base, cand, metrics=[GateMetric("correct")], use_clusters=clustered, n_boot=500
        )
        (reg,) = result.regressions
        c = reg.result.comparison
        assert c.pvalue is not None
        worse_and_significant = c.diff < 0 and c.pvalue < 0.05
        assert (reg.status == "block") == worse_and_significant
        assert f"p={c.pvalue:.3f}" in format_gate(result)
        if clustered:  # the CI and the test are duals
            assert (c.high < 0) == worse_and_significant


ITEMS = [f"q{i:02d}" for i in range(40)]


# Re-review finding: errors inside the exclusion limit could turn a block into a pass.


def _six_regressions(errored: int, metric: str = "correct", worse: bool = False) -> tuple:
    """40 items. The candidate regresses on q00-q05, and the first `errored` of those error."""
    good, bad = (False, True) if worse else (True, False)
    base = [record(f"q{i:02d}", "base", scores={metric: good}) for i in range(40)]
    cand = [record(f"q{i:02d}", "cand", scores={}, error="timeout") for i in range(errored)]
    cand += [record(f"q{i:02d}", "cand", scores={metric: bad}) for i in range(errored, 6)]
    cand += [record(f"q{i:02d}", "cand", scores={metric: good}) for i in range(6, 40)]
    return base, cand


def test_candidate_errors_count_as_failures_so_they_cannot_hide_a_block() -> None:
    base, cand = _six_regressions(errored=0)
    clean = run_gate(base, cand, metrics=[GateMetric("correct")], n_boot=200)
    assert clean.regressions[0].status == "block"  # 6-0 discordant, p = 0.031

    base, cand = _six_regressions(errored=2)  # 2 of the 6 regressions error instead
    result = run_gate(base, cand, metrics=[GateMetric("correct")], on_error="exclude", n_boot=200)
    (reg,) = result.regressions
    assert reg.status == "block"
    assert result.exit_code == 1
    assert reg.result.comparison.pvalue == clean.regressions[0].result.comparison.pvalue
    assert reg.result.imputed == ("q00", "q01")
    text = format_gate(result)
    assert "exact McNemar p=0.031" in text
    assert "2 of 40 (candidate 2, baseline 0), 2 candidate-only counted as failures" in text


def test_candidate_errors_count_as_failures_for_lower_is_better_metrics() -> None:
    base, cand = _six_regressions(errored=2, metric="leak", worse=True)
    result = run_gate(
        base,
        cand,
        metrics=[GateMetric("leak", higher_is_better=False)],
        on_error="exclude",
        n_boot=200,
    )
    assert result.regressions[0].status == "block"
    assert result.regressions[0].result.comparison.candidate_mean == pytest.approx(6 / 40)


def test_items_errored_in_the_baseline_are_still_excluded() -> None:
    base, cand = _six_regressions(errored=0)
    base[39] = record("q39", "base", scores={}, error="timeout")
    result = run_gate(base, cand, metrics=[GateMetric("correct")], on_error="exclude", n_boot=200)
    (reg,) = result.regressions
    assert reg.result.excluded == ("q39",)
    assert reg.result.imputed == ()
    assert reg.result.comparison.n == 39
