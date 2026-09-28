from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from helpers import record

from llm_eval_harness.judge import Checklist, ChecklistItem
from llm_eval_harness.labeling import (
    LabelError,
    LabelItem,
    Sampling,
    SessionResult,
    disagreeing_items,
    labeling_order,
    read_items,
    read_labels,
    run_session,
    uniform_order,
)

CHECKLIST = Checklist(
    name="t",
    items=(ChecklistItem("correct", "Correct?"), ChecklistItem("grounded", "Grounded?")),
)
ITEMS = [
    LabelItem(item_id=f"q{i}", question=f"question {i}", answer=f"answer {i}", reference=f"ref {i}")
    for i in range(5)
]


def scripted(answers: list[str]) -> tuple[Callable[[str], str], list[str]]:
    """An input() stand-in that replays answers, then raises EOFError like a closed stdin."""
    it = iter(answers)
    prompts: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return input_fn, prompts


def session(
    out: Path,
    answers: list[str],
    *,
    seed: int = 3,
    sampling: Sampling = "uniform",
    limit: int | None = None,
) -> tuple[SessionResult, list[str]]:
    input_fn, prompts = scripted(answers)
    printed: list[str] = []
    result = run_session(
        uniform_order(ITEMS, 3),
        CHECKLIST,
        out,
        labeler="rk",
        sampling=sampling,
        seed=seed,
        limit=limit,
        input_fn=input_fn,
        print_fn=printed.append,
    )
    return result, printed + prompts


def test_full_session_writes_one_label_per_item(tmp_path: Path) -> None:
    out = tmp_path / "labels.jsonl"
    result, _ = session(out, ["y", "n"] * 5)
    labels = read_labels(out)
    assert [r.item_id for r in labels] == [i.item_id for i in uniform_order(ITEMS, 3)]
    assert all(r.labels == {"correct": True, "grounded": False} for r in labels)
    assert all(r.sampling == "uniform" and r.seed == 3 and r.labeler == "rk" for r in labels)
    assert result.labeled_now == 5


def test_resume_after_quit_mid_item(tmp_path: Path) -> None:
    out = tmp_path / "labels.jsonl"
    # Finish two items, then quit halfway through the third.
    result, _ = session(out, ["y", "y", "n", "n", "y", "q"])
    assert result.stopped_early
    assert len(read_labels(out)) == 2
    order = [i.item_id for i in uniform_order(ITEMS, 3)]
    result, output = session(out, ["n", "y"] * 3)
    assert result.labeled_now == 3
    assert [r.item_id for r in read_labels(out)] == order
    assert f"question {order[0][1:]}" not in "\n".join(output)  # finished items are not re-shown


def test_end_of_input_stops_cleanly(tmp_path: Path) -> None:
    out = tmp_path / "labels.jsonl"
    result, _ = session(out, ["y", "y"])
    assert result.stopped_early
    assert len(read_labels(out)) == 1


def test_invalid_answers_are_asked_again(tmp_path: Path) -> None:
    out = tmp_path / "labels.jsonl"
    _, output = session(out, ["maybe", "Y", "no"], limit=1)
    assert any("Please answer y or n" in line for line in output)
    assert read_labels(out)[0].labels == {"correct": True, "grounded": False}


def test_limit_caps_the_sample(tmp_path: Path) -> None:
    out = tmp_path / "labels.jsonl"
    result, _ = session(out, ["y", "y"] * 5, limit=2)
    assert result.total_labeled == 2
    assert [r.item_id for r in read_labels(out)] == [i.item_id for i in uniform_order(ITEMS, 3)[:2]]


def test_resume_refuses_a_different_seed_or_mode(tmp_path: Path) -> None:
    out = tmp_path / "labels.jsonl"
    session(out, ["y", "y"], limit=1)
    with pytest.raises(LabelError, match="seed"):
        session(out, [], seed=4)
    with pytest.raises(LabelError, match="sampling"):
        session(out, [], sampling="disagreement")


def test_items_are_shown_blind(tmp_path: Path) -> None:
    items_path = tmp_path / "items.jsonl"
    rows = [
        {
            "item_id": "q1",
            "question": "What is the fee?",
            "answer": "None.",
            "reference": "No fee.",
            "meta": {"model": "gpt-6-luna", "config": "candidate", "judge": "pass"},
        }
    ]
    items_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    items = read_items(items_path)
    input_fn, prompts = scripted(["y", "y"])
    printed: list[str] = []
    run_session(
        items,
        CHECKLIST,
        tmp_path / "labels.jsonl",
        labeler="rk",
        sampling="uniform",
        seed=0,
        input_fn=input_fn,
        print_fn=printed.append,
    )
    everything = "\n".join(printed + prompts)
    assert "What is the fee?" in everything
    for hidden in ("gpt-6-luna", "candidate", "judge"):
        assert hidden not in everything


def test_uniform_order_is_seeded_and_ignores_input_order() -> None:
    assert uniform_order(ITEMS, 1) == uniform_order(list(reversed(ITEMS)), 1)
    assert uniform_order(ITEMS, 1) != uniform_order(ITEMS, 2)


def test_disagreement_sampling_selects_only_disagreements() -> None:
    a = [record("q0"), record("q1"), record("q2", scores={}, error="parse"), record("q3")]
    b = [
        record("q0", run_id="b"),
        record("q1", run_id="b", scores={"correct": False}),
        record("q2", run_id="b"),
        record("q3", run_id="b"),
    ]
    found = disagreeing_items(a, b, ["correct"])
    assert found == {"q1", "q2"}
    ordered = labeling_order(ITEMS, 0, "disagreement", found)
    assert {i.item_id for i in ordered} == {"q1", "q2"}
    with pytest.raises(ValueError, match="disagreeing item ids"):
        labeling_order(ITEMS, 0, "disagreement")


def test_read_labels_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    good = {
        "schema_version": 1,
        "item_id": "q1",
        "labels": {"correct": True},
        "labeler": "rk",
        "sampling": "uniform",
        "seed": 0,
        "created_at": "2026-09-28T00:00:00+00:00",
    }
    path.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n")
    with pytest.raises(LabelError, match="second label"):
        read_labels(path)
    path.write_text(json.dumps({**good, "labels": {"correct": "yes"}}) + "\n")
    with pytest.raises(LabelError, match="true or false"):
        read_labels(path)
    path.write_text(json.dumps({**good, "sampling": "adaptive"}) + "\n")
    with pytest.raises(LabelError, match="sampling"):
        read_labels(path)


def test_read_items_requires_text_fields(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    path.write_text(json.dumps({"item_id": "q1", "question": "Q"}) + "\n")
    with pytest.raises(LabelError, match="answer"):
        read_items(path)
    path.write_text(json.dumps({"item_id": 1, "question": "Q", "answer": "A"}) + "\n")
    with pytest.raises(LabelError, match="item_id must be a string"):
        read_items(path)
