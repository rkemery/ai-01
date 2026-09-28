from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import record, run

from llm_eval_harness.records import (
    EvalRecord,
    MissingScoreError,
    RecordError,
    metric_column,
    read_records,
    write_records,
)


def test_roundtrip(tmp_path: Path) -> None:
    records = [
        record("q1", cluster="art-1", tokens_in=10, tokens_out=5, reasoning_tokens=2),
        record("q2", scores={"correct": False, "f1": 0.5}, meta={"note": "x"}),
    ]
    path = tmp_path / "out" / "results.jsonl"
    assert write_records(path, records) == 2
    assert read_records(path) == records


def test_append_mode_adds_lines(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    write_records(path, [record("q1")])
    write_records(path, [record("q2")], append=True)
    assert [r.item_id for r in read_records(path)] == ["q1", "q2"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"run_id": ""}, "run_id"),
        ({"model": None}, "model"),
        ({"scores": {"correct": float("nan")}}, "finite"),
        ({"scores": {"correct": "yes"}}, "bool or a finite number"),
        ({"scores": {"latency_ms": 1.0}}, "collides"),
        ({"scores": {}}, "must carry an error"),
        ({"tokens_in": -1}, "tokens_in"),
        ({"tokens_out": True}, "tokens_out"),
        ({"tokens_out": 3, "reasoning_tokens": 4}, "cannot exceed"),
        ({"cost_usd": -0.1}, "cost_usd"),
        ({"latency_ms": float("inf")}, "latency_ms"),
        ({"cluster": ""}, "cluster"),
        ({"error": ""}, "error"),
        ({"schema_version": 2}, "schema_version"),
        ({"meta": {1: "x"}}, "meta"),
    ],
)
def test_validation_rejects_bad_records(overrides: dict, message: str) -> None:
    with pytest.raises(RecordError, match=message):
        record(**overrides)


def test_errored_record_may_have_no_scores() -> None:
    r = record(scores={}, error="JudgeParseError: not JSON")
    assert r.error is not None


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _good_line(item_id: str = "q1", **overrides: object) -> str:
    data = record(item_id).to_dict()
    data.update(overrides)
    return json.dumps(data)


def test_read_reports_file_and_line_for_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    _write_lines(path, [_good_line(), "{not json"])
    with pytest.raises(RecordError, match=r"r\.jsonl:2: invalid JSON"):
        read_records(path)


def test_read_rejects_unknown_and_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    _write_lines(path, [_good_line(extra_field=1)])
    with pytest.raises(RecordError, match="unknown fields"):
        read_records(path)
    data = record().to_dict()
    del data["schema_version"]
    _write_lines(path, [json.dumps(data)])
    with pytest.raises(RecordError, match="missing fields"):
        read_records(path)


def test_read_rejects_nan_literal(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    _write_lines(path, [_good_line().replace('"cost_usd": 0.0', '"cost_usd": NaN')])
    with pytest.raises(RecordError, match="NaN is not allowed"):
        read_records(path)


def test_read_rejects_duplicates_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    _write_lines(path, [_good_line("q1"), "", "   ", _good_line("q2")])
    assert len(read_records(path)) == 2
    _write_lines(path, [_good_line("q1"), _good_line("q1")])
    with pytest.raises(RecordError, match="duplicate"):
        read_records(path)


def test_write_validates_everything_before_touching_the_file(tmp_path: Path) -> None:
    bad = record("q2")
    bad.tokens_in = -5  # mutated after construction
    path = tmp_path / "r.jsonl"
    with pytest.raises(RecordError):
        write_records(path, [record("q1"), bad])
    assert not path.exists()
    with pytest.raises(TypeError):
        write_records(path, [{"item_id": "q1"}])  # type: ignore[list-item]


def test_metric_column_basic_and_numeric_fields() -> None:
    records = [record("q1", latency_ms=100.0), record("q2", scores={"correct": False})]
    column = metric_column(records, "correct")
    assert column.values == {"q1": 1.0, "q2": 0.0}
    assert column.binary
    latency = metric_column(records, "latency_ms")
    assert latency.values == {"q1": 100.0, "q2": 0.0}
    assert not latency.binary


def test_metric_column_error_policy() -> None:
    records = [record("q1"), record("q2", scores={}, error="timeout")]
    with pytest.raises(MissingScoreError, match="timeout"):
        metric_column(records, "correct")
    column = metric_column(records, "correct", on_error="exclude")
    assert column.values == {"q1": 1.0}
    assert column.excluded == ("q2",)
    with pytest.raises(ValueError, match="on_error"):
        metric_column(records, "correct", on_error="ignore")  # type: ignore[arg-type]


def test_metric_column_missing_metric_without_error_is_a_contract_violation() -> None:
    with pytest.raises(RecordError, match="has no metric 'grounded'"):
        metric_column([record("q1")], "grounded")


def test_metric_column_rejects_mixed_types_and_multiple_runs() -> None:
    mixed = [record("q1"), record("q2", scores={"correct": 0.5})]
    with pytest.raises(RecordError, match="mixes bool and numeric"):
        metric_column(mixed, "correct")
    two_runs = run("a", {"q1": True}) + run("b", {"q1": True})
    with pytest.raises(RecordError, match="exactly one run"):
        metric_column(two_runs, "correct")


def test_record_is_a_plain_dataclass() -> None:
    r = EvalRecord.from_dict(record().to_dict())
    assert r == record()
