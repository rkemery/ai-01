from __future__ import annotations

from pathlib import Path

import pytest
from helpers import record, run

from llm_eval_harness.analysis import compare_runs, pass_k_from_records, summarize_metric
from llm_eval_harness.report import (
    comparison_table,
    mde_line,
    replace_section,
    results_table,
    write_section,
)
from llm_eval_harness.stats import mde_two_proportions, wilson_interval


def test_results_table_and_mde_line() -> None:
    records = run("r", {f"q{i}": i < 28 for i in range(40)})
    summary = summarize_metric(records, "correct")
    assert summary.interval == wilson_interval(28, 40)
    table = results_table([summary])
    assert table.splitlines()[0] == "| Metric | Value | 95% CI | n |"
    assert "| correct | 70.0% | 54.6% to 81.9% | 40 |" in table
    assert summary.mde_vs_same_size_run == pytest.approx(mde_two_proportions(40, 40, 0.7))
    assert "correct 28.7 pts" in mde_line([summary])


def test_mde_is_na_for_zero_variance() -> None:
    summary = summarize_metric(run("r", {f"q{i}": False for i in range(10)}), "correct")
    assert summary.mde_vs_same_size_run is None
    assert "correct n/a" in mde_line([summary])


def test_comparison_table_has_mcnemar_and_mde() -> None:
    base = run("b", {f"q{i}": i % 2 == 0 for i in range(20)})
    cand = run("c", {f"q{i}": i % 4 != 3 for i in range(20)})
    result = compare_runs(base, cand, "correct", n_boot=500)
    table = comparison_table([result])
    assert "| correct | 50.0% | 75.0% | +25.0 pts |" in table
    assert result.comparison.mcnemar is not None
    assert f"{result.comparison.mcnemar.pvalue:.3f}" in table


def test_pass_k_from_records_uses_each_run_as_a_trial() -> None:
    trials = []
    for t, outcomes in enumerate([(True, True), (True, False), (True, True), (False, False)]):
        trials += run(f"trial-{t}", {"task-a": outcomes[0], "task-b": outcomes[1]})
    result = pass_k_from_records(trials, "correct", k=2, n_boot=200)
    # task-a: 3 of 4 pass, task-b: 2 of 4 pass.
    assert result.pass_hat_k.estimate == pytest.approx((3 / 6 + 1 / 6) / 2)


def test_pass_k_refuses_errored_trials() -> None:
    trials = [*run("t0", {"a": True}), record("a", "t1", scores={}, error="crash")]
    with pytest.raises(ValueError, match="crash"):
        pass_k_from_records(trials, "correct", k=1)


def test_replace_section(tmp_path: Path) -> None:
    text = "intro\n<!-- demo:start -->\nold\n<!-- demo:end -->\nrest\n"
    assert replace_section(text, "demo", "new") == (
        "intro\n<!-- demo:start -->\nnew\n<!-- demo:end -->\nrest\n"
    )
    with pytest.raises(ValueError, match="exactly one"):
        replace_section("no markers", "demo", "x")
    with pytest.raises(ValueError, match="exactly one"):
        replace_section(text + text, "demo", "x")
    with pytest.raises(ValueError, match="before"):
        replace_section("<!-- demo:end -->\n<!-- demo:start -->", "demo", "x")
    path = tmp_path / "README.md"
    path.write_text(text)
    assert write_section(path, "demo", "new")
    assert not write_section(path, "demo", "new")
