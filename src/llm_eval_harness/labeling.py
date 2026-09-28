"""Blind, randomized, resumable human labeling in the terminal.

The labeler sees the question, the reference and the answer. Nothing else from
the items file is shown (no model, config, run or judge verdict), which keeps
the labels blind. The order is a seeded shuffle of the sorted item ids, so
re-running with the same seed resumes the same sequence. Each label is appended
and flushed as soon as an item is finished, so quitting or crashing loses at
most the item in progress.

Every label records what sample it belongs to: the labeler, the sampling mode,
the seed, a hash of the item ids the order was drawn from, and the target size
(--n). A resumed session must match all five, so one labels file never mixes
two samples or two people.

Sampling modes:

- uniform (default): a seeded random order over all items. Use these labels to
  measure judge agreement (TPR, TNR, kappa).
- disagreement: only items where two judges disagree. Good for finding judge
  bugs, but the sample is biased toward hard cases, so the labels are tagged
  and `calibration` refuses to use them for agreement metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np

from llm_eval_harness.judge import Checklist
from llm_eval_harness.records import EvalRecord, metric_column

LABEL_SCHEMA_VERSION = 2
Sampling = Literal["uniform", "disagreement"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LabelError(ValueError):
    """A labels or items file breaks its format."""


@dataclass(kw_only=True)
class LabelRecord:
    """One human's yes/no answers to every checklist question for one item.

    `items_sha256` (see `items_fingerprint`) and `n_target` identify the sample
    the item was drawn for: the pool of item ids that was shuffled, and how
    many of the shuffled order were to be labeled.
    """

    schema_version: int = LABEL_SCHEMA_VERSION
    item_id: str
    labels: dict[str, bool]
    labeler: str
    sampling: Sampling
    seed: int
    items_sha256: str
    n_target: int
    created_at: str

    def __post_init__(self) -> None:
        if self.schema_version != LABEL_SCHEMA_VERSION:
            raise LabelError(f"unsupported label schema_version {self.schema_version!r}")
        for name in ("item_id", "labeler", "created_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise LabelError(f"{name} must be a non-empty string, got {value!r}")
        if not isinstance(self.labels, dict) or not self.labels:
            raise LabelError("labels must be a non-empty dict")
        for key, value in self.labels.items():
            if not isinstance(key, str) or not isinstance(value, bool):
                raise LabelError(f"label {key!r} must map to true or false, got {value!r}")
        if self.sampling not in get_args(Sampling):
            raise LabelError(f"sampling must be one of {get_args(Sampling)}, got {self.sampling!r}")
        if type(self.seed) is not int:
            raise LabelError(f"seed must be an int, got {self.seed!r}")
        if not isinstance(self.items_sha256, str) or not _SHA256.match(self.items_sha256):
            raise LabelError(f"items_sha256 must be a sha256 hex digest, got {self.items_sha256!r}")
        if type(self.n_target) is not int or self.n_target < 1:
            raise LabelError(f"n_target must be a positive int, got {self.n_target!r}")


@dataclass(frozen=True)
class LabelItem:
    """The parts of an item a labeler is allowed to see."""

    item_id: str
    question: str
    answer: str
    reference: str | None = None


@dataclass(frozen=True)
class SessionResult:
    labeled_now: int
    total_labeled: int
    target: int
    stopped_early: bool


def read_labels(path: str | Path) -> list[LabelRecord]:
    """Read a labels JSONL file strictly. One label record per item."""
    records: list[LabelRecord] = []
    seen: set[str] = set()
    known = {f.name for f in fields(LabelRecord)}
    for lineno, data in _iter_json_lines(path):
        where = f"{path}:{lineno}"
        if not isinstance(data, dict) or set(data) - known:
            raise LabelError(f"{where}: not a label record")
        try:
            record = LabelRecord(**data)
        except TypeError as exc:
            raise LabelError(f"{where}: {exc}") from exc
        except LabelError as exc:
            raise LabelError(f"{where}: {exc}") from exc
        if record.item_id in seen:
            raise LabelError(f"{where}: second label for item {record.item_id!r}")
        seen.add(record.item_id)
        records.append(record)
    return records


def append_label(path: str | Path, record: LabelRecord) -> None:
    """Append one label and flush it to disk before returning."""
    line = json.dumps(asdict(record), ensure_ascii=False) + "\n"
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def read_items(path: str | Path) -> list[LabelItem]:
    """Read items to label: JSONL with item_id, question, answer and optional reference.

    Other keys are allowed and ignored, which is what keeps model names and
    judge outputs out of the labeler's view.
    """
    items: list[LabelItem] = []
    seen: set[str] = set()
    for lineno, data in _iter_json_lines(path):
        where = f"{path}:{lineno}"
        if not isinstance(data, dict):
            raise LabelError(f"{where}: expected a JSON object")
        missing = {"item_id", "question", "answer"} - set(data)
        if missing:
            raise LabelError(f"{where}: missing {sorted(missing)}")
        values = {k: data.get(k) for k in ("item_id", "question", "answer", "reference")}
        for key, value in values.items():
            if (value is not None or key != "reference") and not isinstance(value, str):
                raise LabelError(f"{where}: {key} must be a string")
        if values["item_id"] in seen:
            raise LabelError(f"{where}: duplicate item_id {values['item_id']!r}")
        seen.add(values["item_id"])
        items.append(LabelItem(**values))
    return items


def items_fingerprint(item_ids: Iterable[str]) -> str:
    """sha256 of the sorted item ids, one per line. Order-independent."""
    return hashlib.sha256("\n".join(sorted(item_ids)).encode("utf-8")).hexdigest()


def uniform_order(items: Sequence[LabelItem], seed: int) -> list[LabelItem]:
    """Seeded shuffle of the items sorted by id, so input order does not matter."""
    ordered = sorted(items, key=lambda item: item.item_id)
    perm = np.random.default_rng(seed).permutation(len(ordered))
    return [ordered[i] for i in perm]


def disagreeing_items(
    judge_a: Sequence[EvalRecord], judge_b: Sequence[EvalRecord], checks: Sequence[str]
) -> set[str]:
    """Item ids where two judge runs differ on any check, or where only one of them errored."""
    found: set[str] = set()
    for check in checks:
        a = metric_column(judge_a, check, on_error="exclude")
        b = metric_column(judge_b, check, on_error="exclude")
        both = set(a.values) & set(b.values)
        found |= {i for i in both if a.values[i] != b.values[i]}
        found |= set(a.excluded) ^ set(b.excluded)
    return found


def labeling_order(
    items: Sequence[LabelItem],
    seed: int,
    sampling: Sampling = "uniform",
    disagreements: set[str] | None = None,
) -> list[LabelItem]:
    """The order items are shown in. Disagreement mode keeps only the flagged items."""
    if sampling == "uniform":
        return uniform_order(items, seed)
    if sampling == "disagreement":
        if disagreements is None:
            raise ValueError("disagreement sampling needs the set of disagreeing item ids")
        return uniform_order([i for i in items if i.item_id in disagreements], seed)
    raise ValueError(f"unknown sampling mode {sampling!r}")


def run_session(
    ordered: Sequence[LabelItem],
    checklist: Checklist,
    out_path: str | Path,
    *,
    labeler: str,
    sampling: Sampling,
    seed: int,
    limit: int | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> SessionResult:
    """Label items in order, skipping ones already in `out_path`.

    `limit` caps the sample at the first `limit` items of the order, so a
    resumed session with the same items, seed and limit finishes the same
    sample. Resuming with a different item set, limit, labeler, sampling mode
    or seed raises `LabelError` before anything is shown. Typing q (or end of
    input) stops after saving everything finished so far.
    """
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError(f"limit must be a positive int, got {limit!r}")
    out_path = Path(out_path)
    target = list(ordered if limit is None else ordered[:limit])
    if not target:
        raise LabelError("no items to label")
    sample = _Sample(
        labeler=labeler,
        sampling=sampling,
        seed=seed,
        items_sha256=items_fingerprint(item.item_id for item in ordered),
        n_target=len(target),
    )
    existing = read_labels(out_path) if out_path.exists() else []
    _check_compatible(existing, checklist, sample, target, out_path)
    done = {record.item_id for record in existing}
    todo = [item for item in target if item.item_id not in done]
    total = len(done)
    print_fn(f"{total} of {len(target)} labeled. {len(todo)} to go. Type q to stop.")
    labeled_now = 0
    for item in todo:
        _show_item(item, total + 1, len(target), print_fn)
        labels = _ask_checklist(checklist, input_fn, print_fn)
        if labels is None:
            print_fn(f"Stopped. {total} of {len(target)} saved to {out_path}.")
            return SessionResult(labeled_now, total, len(target), stopped_early=True)
        append_label(
            out_path,
            LabelRecord(
                item_id=item.item_id,
                labels=labels,
                labeler=sample.labeler,
                sampling=sample.sampling,
                seed=sample.seed,
                items_sha256=sample.items_sha256,
                n_target=sample.n_target,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        labeled_now += 1
        total += 1
    print_fn(f"Done. {total} of {len(target)} labeled in {out_path}.")
    return SessionResult(labeled_now, total, len(target), stopped_early=False)


@dataclass(frozen=True)
class _Sample:
    labeler: str
    sampling: Sampling
    seed: int
    items_sha256: str
    n_target: int


def _check_compatible(
    existing: Sequence[LabelRecord],
    checklist: Checklist,
    sample: _Sample,
    target: Sequence[LabelItem],
    path: Path,
) -> None:
    fix = "Resume with the same settings, or use a new labels file."
    for record in existing:
        if record.labeler != sample.labeler:
            raise LabelError(
                f"{path} holds labels by labeler {record.labeler!r}, not {sample.labeler!r}. "
                "One labels file holds one labeler, so agreement between labelers can be "
                "measured. Use a new labels file."
            )
        if record.sampling != sample.sampling or record.seed != sample.seed:
            raise LabelError(
                f"{path} was started with sampling={record.sampling!r} seed={record.seed}, "
                f"not sampling={sample.sampling!r} seed={sample.seed}. {fix}"
            )
        if record.items_sha256 != sample.items_sha256:
            raise LabelError(
                f"{path} was started on a different item set (items_sha256 "
                f"{record.items_sha256[:12]}..., now {sample.items_sha256[:12]}...). Adding or "
                f"removing items reshuffles the order, so the labels would not be one random "
                f"sample. {fix}"
            )
        if record.n_target != sample.n_target:
            raise LabelError(
                f"{path} was started with --n {record.n_target}, not --n {sample.n_target}. {fix}"
            )
        if set(record.labels) != set(checklist.ids):
            raise LabelError(
                f"{path} has labels for {sorted(record.labels)}, "
                f"but the checklist asks {sorted(checklist.ids)}"
            )
    outside = sorted({r.item_id for r in existing} - {item.item_id for item in target})
    if outside:
        raise LabelError(f"{path} has labels for items outside this sample: {outside[:5]}. {fix}")


def _show_item(item: LabelItem, position: int, total: int, print_fn: Callable[[str], None]) -> None:
    print_fn("")
    print_fn(f"=== Item {position} of {total} ===")
    print_fn(f"QUESTION:\n{item.question}\n")
    if item.reference is not None:
        print_fn(f"REFERENCE:\n{item.reference}\n")
    print_fn(f"ANSWER:\n{item.answer}\n")


def _ask_checklist(
    checklist: Checklist, input_fn: Callable[[str], str], print_fn: Callable[[str], None]
) -> dict[str, bool] | None:
    labels: dict[str, bool] = {}
    for check in checklist.items:
        answer = _ask_yes_no(f"{check.id}: {check.question} [y/n] ", input_fn, print_fn)
        if answer is None:
            return None
        labels[check.id] = answer
    return labels


def _ask_yes_no(
    prompt: str, input_fn: Callable[[str], str], print_fn: Callable[[str], None]
) -> bool | None:
    while True:
        try:
            raw = input_fn(prompt).strip().lower()
        except EOFError:
            return None
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        if raw in ("q", "quit"):
            return None
        print_fn("Please answer y or n (q stops and saves).")


def _iter_json_lines(path: str | Path) -> Iterator[tuple[int, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                raise LabelError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
