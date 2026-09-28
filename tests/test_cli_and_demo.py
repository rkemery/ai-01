from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest
from helpers import record, run

from llm_eval_harness.cli import main
from llm_eval_harness.demo import SECTION, run_demo
from llm_eval_harness.records import write_records

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "synthetic"


def test_stats_prints_table_mde_and_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = run("b", {f"q{i}": i % 3 != 0 for i in range(30)})
    cand = run("c", {f"q{i}": i % 5 != 0 for i in range(30)})
    write_records(tmp_path / "b.jsonl", base)
    write_records(tmp_path / "c.jsonl", cand)
    code = main(["stats", str(tmp_path / "c.jsonl"), "--compare", str(tmp_path / "b.jsonl")])
    out = capsys.readouterr().out
    assert code == 0
    assert "| correct | 80.0% |" in out
    assert "MDE against another run" in out
    assert "p from the exact McNemar test" in out


def test_stats_pass_k(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    trials = run("t0", {"a": True, "b": False}) + run("t1", {"a": True, "b": True})
    write_records(tmp_path / "trials.jsonl", trials)
    assert main(["stats", str(tmp_path / "trials.jsonl"), "--k", "2"]) == 0
    assert "| correct pass^k | 2 | 50.0% |" in capsys.readouterr().out


def test_report_writes_marked_section(tmp_path: Path) -> None:
    write_records(tmp_path / "r.jsonl", run("r", {"q1": True, "q2": False}))
    readme = tmp_path / "README.md"
    readme.write_text("# x\n<!-- results:start -->\n<!-- results:end -->\n")
    assert main(["report", str(tmp_path / "r.jsonl"), "--write", str(readme)]) == 0
    assert "| correct | 50.0% |" in readme.read_text()


def test_split_command(tmp_path: Path) -> None:
    out = tmp_path / "split.json"
    assert (
        main(["split", "--items", str(DATA / "items.jsonl"), "--n-dev", "10", "--out", str(out)])
        == 0
    )
    assert out.exists()


def test_calibrate_on_demo_data(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["calibrate", "--judge", str(DATA / "candidate.jsonl"), "--labels"]
    args += [str(DATA / "human_labels.jsonl"), "--split", str(DATA / "split.json")]
    args += ["--check", "correct", "--apply", str(DATA / "baseline.jsonl"), "--n-boot", "500"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "| Check | n | TPR | TNR | Cohen's kappa |" in out
    assert "Corrected pass rates for demo-baseline" in out


def test_label_command_reads_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = iter(["y", "n", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    out = tmp_path / "labels.jsonl"
    args = ["label", "--items", str(DATA / "items.jsonl"), "--checklist"]
    args += [str(DATA / "checklist.json"), "--out", str(out), "--labeler", "me", "--n", "3"]
    assert main(args) == 0
    assert len(out.read_text().splitlines()) == 1
    assert "Stopped. 1 of 3 saved" in capsys.readouterr().out


def test_cli_reports_bad_input_with_exit_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    assert main(["stats", str(bad)]) == 2
    assert "bad.jsonl:1: invalid JSON" in capsys.readouterr().err


def test_readme_demo_section_is_up_to_date() -> None:
    """The README numbers must be exactly what `llm-eval demo` prints today."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start, end = f"<!-- {SECTION}:start -->", f"<!-- {SECTION}:end -->"
    section = readme.split(start)[1].split(end)[0].strip()
    assert section == run_demo(DATA).strip(), "README is stale: run `make demo`"


def test_synthetic_data_regenerates_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = ROOT / "examples" / "make_synthetic_data.py"
    monkeypatch.setattr(sys, "argv", [str(script), str(tmp_path)])
    runpy.run_path(str(script), run_name="__main__")
    for committed in DATA.iterdir():
        assert (tmp_path / committed.name).read_bytes() == committed.read_bytes(), committed.name


def test_cli_hint_names_an_option_the_command_has(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [*run("r", {"q1": True, "q2": False}), record("q3", "r", scores={}, error="crash")]
    write_records(tmp_path / "r.jsonl", records)
    assert main(["stats", str(tmp_path / "r.jsonl")]) == 2
    err = capsys.readouterr().err
    assert "because of an error: crash" in err
    assert "Rerun with --on-error exclude" in err
    assert "on_error='exclude'" not in err
    assert main(["stats", str(tmp_path / "r.jsonl"), "--on-error", "exclude"]) == 0
    assert "Excluded errored items: 1 from correct" in capsys.readouterr().out


def test_stats_k_refuses_options_it_cannot_honor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trials = run("t0", {"a": True, "b": False}, clusters={"a": "x", "b": "y"})
    trials += run("t1", {"a": True, "b": True}, clusters={"a": "x", "b": "y"})
    write_records(tmp_path / "trials.jsonl", trials)
    path = str(tmp_path / "trials.jsonl")
    assert main(["stats", path, "--k", "2", "--cluster"]) == 2
    assert "--cluster is not supported with --k" in capsys.readouterr().err
    assert main(["stats", path, "--k", "2", "--on-error", "exclude"]) == 2
    assert "would inflate pass^k" in capsys.readouterr().err
    errored = [*trials, record("a", "t2", scores={}, error="crash")]
    write_records(tmp_path / "errored.jsonl", errored)
    assert main(["stats", str(tmp_path / "errored.jsonl"), "--k", "2"]) == 2
    err = capsys.readouterr().err
    assert "pass^k counts every trial" in err
    assert "--on-error" not in err


def _calibrate_args(*extra: str) -> list[str]:
    args = ["calibrate", "--judge", str(DATA / "candidate.jsonl"), "--labels"]
    return [*args, str(DATA / "human_labels.jsonl"), "--n-boot", "300", *extra]


def test_calibrate_works_on_demo_data_without_check(capsys: pytest.CaptureFixture[str]) -> None:
    split = ["--split", str(DATA / "split.json")]
    assert main(_calibrate_args(*split, "--apply", str(DATA / "baseline.jsonl"))) == 0
    out = capsys.readouterr().out
    assert "| correct |" in out
    assert "| grounded |" in out
    assert "pii_leak" not in out  # not a labeled check
    assert "Bootstrap replicates dropped" in out


def test_calibrate_requires_an_explicit_split(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(_calibrate_args())
    assert info.value.code == 2
    assert "--split" in capsys.readouterr().err


def test_calibrate_apply_refuses_another_judges_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [json.loads(line) for line in (DATA / "baseline.jsonl").read_text().splitlines()]
    for row in rows:
        row["meta"]["judge_fingerprint"] = "some-other-judge"
    other = tmp_path / "other.jsonl"
    other.write_text("".join(json.dumps(row) + "\n" for row in rows))
    split = ["--split", str(DATA / "split.json")]
    assert main(_calibrate_args(*split, "--apply", str(other))) == 2
    assert "judged by ['some-other-judge']" in capsys.readouterr().err


def test_calibrate_apply_handles_errored_records(capsys: pytest.CaptureFixture[str]) -> None:
    # The candidate run has one errored record, q-disputes-3.
    args = _calibrate_args("--split", str(DATA / "split.json"), "--apply")
    assert main([*args, str(DATA / "candidate.jsonl")]) == 2
    err = capsys.readouterr().err
    assert "q-disputes-3" in err
    assert "Rerun with --on-error exclude" in err
    assert main([*args, str(DATA / "candidate.jsonl"), "--on-error", "exclude"]) == 0
    assert "1 errored items left out of the corrected pass rate" in capsys.readouterr().out
