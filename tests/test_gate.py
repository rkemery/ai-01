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
