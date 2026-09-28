from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval_harness.client import BudgetExceeded, FakeClient, ModelRequest
from llm_eval_harness.judge import (
    Checklist,
    ChecklistItem,
    ChecklistJudge,
    JudgeParseError,
    PairwiseJudge,
    load_checklist,
    parse_checklist_reply,
    summarize_pairwise,
)

CHECKLIST = Checklist(
    name="t",
    items=(
        ChecklistItem("correct", "Does it match the reference?"),
        ChecklistItem("grounded", "Is every claim supported?"),
    ),
)
GOOD = json.dumps(
    {"correct": {"pass": True, "reason": "matches"}, "grounded": {"pass": False, "reason": "adds"}}
)


def make_judge(*replies: str, **kw: object) -> tuple[ChecklistJudge, FakeClient]:
    fake = FakeClient(list(replies))
    return ChecklistJudge(fake, "Llama-3.3-70B-Instruct", CHECKLIST, **kw), fake  # type: ignore[arg-type]


def test_judge_parses_verdicts_and_builds_reference_guided_prompt() -> None:
    judge, fake = make_judge(GOOD)
    verdict = judge.judge("What is the fee?", "It costs $15.", "There is no fee.")
    assert verdict.scores == {"correct": True, "grounded": False}
    assert verdict.results["grounded"].reason == "adds"
    prompt = fake.calls[0].input
    assert isinstance(prompt, str)
    for text in ("What is the fee?", "It costs $15.", "There is no fee.", "- correct:", "grounded"):
        assert text in prompt


def test_code_fence_is_tolerated() -> None:
    judge, _ = make_judge(f"```json\n{GOOD}\n```")
    assert judge.judge("q", "a", "r").scores["correct"] is True


@pytest.mark.parametrize(
    "reply",
    [
        "Both checks pass.",
        "[]",
        json.dumps({"correct": {"pass": True, "reason": "ok"}}),
        json.dumps(
            {
                "correct": {"pass": True, "reason": "ok"},
                "grounded": {"pass": True, "reason": "ok"},
                "extra": {"pass": True, "reason": "ok"},
            }
        ),
        json.dumps(
            {"correct": {"pass": "yes", "reason": "ok"}, "grounded": {"pass": True, "reason": "ok"}}
        ),
        json.dumps(
            {"correct": {"pass": 1, "reason": "ok"}, "grounded": {"pass": True, "reason": "ok"}}
        ),
        json.dumps({"correct": {"pass": True}, "grounded": {"pass": True, "reason": "ok"}}),
        f"Sure! {GOOD}",
    ],
)
def test_parse_failures_raise(reply: str) -> None:
    with pytest.raises(JudgeParseError):
        parse_checklist_reply(reply, CHECKLIST.ids)


def test_score_records_parse_failure_as_error_never_a_pass() -> None:
    judge, _ = make_judge("Looks good to me!")
    outcome = judge.score("q", "a", "r")
    assert outcome.scores == {}
    assert outcome.error is not None
    assert outcome.error.startswith("JudgeParseError")
    assert outcome.response is not None  # usage is kept so the call can still be billed


def test_score_does_not_swallow_other_errors() -> None:
    def broke(request: ModelRequest) -> str:
        raise BudgetExceeded("cap")

    judge = ChecklistJudge(FakeClient(broke), "gpt-5-mini", CHECKLIST)
    with pytest.raises(BudgetExceeded):
        judge.score("q", "a", "r")


def test_fingerprint_tracks_everything_that_defines_the_judge() -> None:
    base, _ = make_judge()
    same, _ = make_judge()
    assert base.fingerprint == same.fingerprint
    assert base.fingerprint != make_judge(template=base.template + " ")[0].fingerprint
    assert base.fingerprint != make_judge(temperature=0.0)[0].fingerprint
    other = ChecklistJudge(FakeClient([]), "gpt-5-mini", CHECKLIST)
    assert base.fingerprint != other.fingerprint


def test_load_checklist(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"name": "x", "items": [{"id": "a", "question": "Q?"}]}))
    assert load_checklist(path).ids == ("a",)
    path.write_text(json.dumps({"items": [{"id": "a", "question": "Q?"}] * 2}))
    with pytest.raises(ValueError, match="duplicate"):
        load_checklist(path)
    path.write_text(json.dumps({"items": [{"id": "has space", "question": "Q?"}]}))
    with pytest.raises(ValueError, match="must match"):
        load_checklist(path)


def pairwise(*winners: str) -> tuple[PairwiseJudge, FakeClient]:
    fake = FakeClient([json.dumps({"winner": w, "reason": "r"}) for w in winners])
    return PairwiseJudge(fake, "gpt-5-mini"), fake


def test_pairwise_consistent_preference() -> None:
    judge, fake = pairwise("1", "2")  # A first wins, then B first loses: A both times
    verdict = judge.compare("q", "answer A", "answer B", "ref")
    assert verdict.winner == "a"
    assert not verdict.flipped
    first, second = (str(call.input) for call in fake.calls)
    assert first.index("answer A") < first.index("answer B")
    assert second.index("answer B") < second.index("answer A")


@pytest.mark.parametrize(
    ("winners", "expected_first", "expected_second"),
    [(("1", "1"), "a", "b"), (("2", "2"), "b", "a"), (("tie", "1"), "tie", "b")],
)
def test_pairwise_flip_counts_as_tie(
    winners: tuple[str, str], expected_first: str, expected_second: str
) -> None:
    judge, _ = pairwise(*winners)
    verdict = judge.compare("q", "A", "B", "ref")
    assert (verdict.first_order, verdict.second_order) == (expected_first, expected_second)
    assert verdict.flipped
    assert verdict.winner == "tie"


def test_pairwise_summary_reports_flip_rate() -> None:
    judge, _ = pairwise("1", "2", "1", "1", "tie", "tie", "2", "1")
    verdicts = [judge.compare("q", "A", "B", "ref") for _ in range(4)]
    summary = summarize_pairwise(verdicts)
    assert (summary.a_wins, summary.b_wins, summary.ties) == (1, 1, 2)
    assert summary.flips == 1
    assert summary.flip_rate.estimate == pytest.approx(0.25)
    assert summary.flip_rate.method == "Wilson"


def test_pairwise_parse_error_raises() -> None:
    fake = FakeClient([json.dumps({"winner": "A", "reason": "r"})])
    with pytest.raises(JudgeParseError, match="winner"):
        PairwiseJudge(fake, "gpt-5-mini").compare("q", "a", "b", "r")
