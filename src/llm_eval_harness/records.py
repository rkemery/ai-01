"""The JSONL results contract: one `EvalRecord` per item per run.

Every repo in the portfolio writes this shape and every tool here reads it.
Reading is strict. A malformed line raises `RecordError` with the file and line
number, it is never skipped. Repeated trials of the same item (for pass^k) are
separate runs with their own `run_id`.

Token convention: `tokens_out` is the billed output count and already includes
`reasoning_tokens`, matching what the OpenAI Responses API reports in `usage`.

Two kinds of failure: `error` means the model call itself failed, so there is
no answer and its latency, cost and token counts are not measurements.
`score_error` means the answer exists but a scorer failed (for example the
judge reply did not parse), so only the missing scores are affected.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, get_args

SCHEMA_VERSION = 1

# Record fields that can be read as metrics next to the entries in `scores`.
NUMERIC_FIELDS = ("tokens_in", "tokens_out", "reasoning_tokens", "cost_usd", "latency_ms")
_REQUIRED_KEYS = frozenset({"schema_version", "run_id", "item_id", "config", "model", "scores"})

OnError = Literal["raise", "exclude"]


class RecordError(ValueError):
    """A record breaks the results contract."""


class MissingScoreError(RecordError):
    """A metric is absent because its record carries an error, and on_error is 'raise'.

    `detail` names the item, run, metric and error without the hint, so the CLI
    can point at its own flag instead.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(
            f"{detail}. Pass on_error='exclude' to leave errored items out and report them."
        )
        self.detail = detail


@dataclass(kw_only=True)
class EvalRecord:
    """One scored item from one run. Validated on construction."""

    schema_version: int = SCHEMA_VERSION
    run_id: str
    item_id: str
    config: str
    model: str
    scores: dict[str, float | bool]
    cluster: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None
    score_error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_record(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalRecord:
        if not isinstance(data, Mapping):
            raise RecordError(f"a record must be a JSON object, got {type(data).__name__}")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise RecordError(f"unknown fields {sorted(unknown)}")
        missing = _REQUIRED_KEYS - set(data)
        if missing:
            raise RecordError(f"missing fields {sorted(missing)}")
        return cls(**data)


def validate_record(record: EvalRecord) -> None:
    """Raise `RecordError` if the record breaks the contract."""
    if type(record.schema_version) is not int or record.schema_version != SCHEMA_VERSION:
        raise RecordError(f"schema_version must be {SCHEMA_VERSION}, got {record.schema_version!r}")
    for name in ("run_id", "item_id", "config", "model"):
        _check_text(name, getattr(record, name))
    if record.cluster is not None:
        _check_text("cluster", record.cluster)
    _check_scores(record.scores)
    for name in ("tokens_in", "tokens_out", "reasoning_tokens"):
        value = getattr(record, name)
        if type(value) is not int or value < 0:
            raise RecordError(f"{name} must be a non-negative int, got {value!r}")
    if record.reasoning_tokens > record.tokens_out:
        raise RecordError("reasoning_tokens are part of tokens_out and cannot exceed it")
    for name in ("cost_usd", "latency_ms"):
        value = getattr(record, name)
        if not _is_number(value) or value < 0:
            raise RecordError(f"{name} must be a finite number >= 0, got {value!r}")
    for name in ("error", "score_error"):
        if getattr(record, name) is not None:
            _check_text(name, getattr(record, name))
    if not isinstance(record.meta, dict) or not all(isinstance(k, str) for k in record.meta):
        raise RecordError("meta must be a dict with string keys")
    if not record.scores and record.error is None and record.score_error is None:
        raise RecordError("a record with no scores must carry an error or a score_error")


def write_records(path: str | Path, records: Iterable[EvalRecord], *, append: bool = False) -> int:
    """Write records as JSONL and return how many were written.

    Everything is validated before the file is opened, so a bad record never
    leaves a half-written file behind.
    """
    batch = list(records)
    for record in batch:
        if not isinstance(record, EvalRecord):
            raise TypeError(f"expected EvalRecord, got {type(record).__name__}")
        validate_record(record)
    check_unique(batch, source=str(path))
    lines = [json.dumps(r.to_dict(), ensure_ascii=False, allow_nan=False) + "\n" for r in batch]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    return len(batch)


def iter_records(path: str | Path) -> Iterator[EvalRecord]:
    """Yield records from a JSONL file. Whitespace-only lines are allowed and hold no data."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                data = json.loads(line, parse_constant=_reject_constant)
            except json.JSONDecodeError as exc:
                raise RecordError(f"{where}: invalid JSON ({exc.msg})") from exc
            except ValueError as exc:
                raise RecordError(f"{where}: {exc}") from exc
            try:
                yield EvalRecord.from_dict(data)
            except RecordError as exc:
                raise RecordError(f"{where}: {exc}") from exc


def read_records(path: str | Path) -> list[EvalRecord]:
    """Read and validate every record, rejecting duplicate (run_id, item_id) pairs."""
    records = list(iter_records(path))
    check_unique(records, source=str(path))
    return records


def check_unique(records: Iterable[EvalRecord], source: str = "records") -> None:
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record.run_id, record.item_id)
        if key in seen:
            raise RecordError(
                f"{source}: duplicate record for run_id={record.run_id!r} "
                f"item_id={record.item_id!r}"
            )
        seen.add(key)


def group_by_run(records: Iterable[EvalRecord]) -> dict[str, list[EvalRecord]]:
    runs: dict[str, list[EvalRecord]] = {}
    for record in records:
        runs.setdefault(record.run_id, []).append(record)
    return runs


def single_run_id(records: Sequence[EvalRecord]) -> str:
    """Return the one run_id shared by all records, or raise."""
    run_ids = {r.run_id for r in records}
    if len(run_ids) != 1:
        raise RecordError(f"expected records from exactly one run, got {sorted(run_ids)}")
    return run_ids.pop()


@dataclass(frozen=True)
class MetricColumn:
    """One metric from one run, keyed by item_id. Booleans are stored as 0.0 and 1.0.

    `clusters` covers every item, excluded ones included.
    """

    metric: str
    run_id: str
    values: dict[str, float]
    clusters: dict[str, str | None]
    binary: bool
    excluded: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return len(self.values)


def metric_column(
    records: Sequence[EvalRecord], metric: str, *, on_error: OnError = "raise"
) -> MetricColumn:
    """Extract one metric from a single run.

    A metric comes from `scores`, or from a numeric record field such as
    `latency_ms`. When a record has an `error` or a `score_error` and lacks the
    metric, `on_error` decides: 'raise' (the default) raises
    `MissingScoreError`, 'exclude' leaves the item out and lists it in
    `excluded` so callers can report it. A missing metric on a record without
    either is always a contract violation.

    The record fields (`latency_ms`, `cost_usd`, token counts) of a record
    whose model call failed (`error`) count as missing too: a crash reports
    0 ms and a timeout reports the timeout, so neither belongs in a mean. A
    record with only a `score_error` keeps them, since its model call worked.
    Sum `cost_usd` over every record for total spend.
    """
    if on_error not in get_args(OnError):
        raise ValueError(f"on_error must be one of {get_args(OnError)}, got {on_error!r}")
    if not records:
        raise RecordError("no records")
    run_id = single_run_id(records)
    check_unique(records, source=f"run {run_id!r}")
    values: dict[str, float] = {}
    clusters: dict[str, str | None] = {}
    excluded: list[str] = []
    kinds: set[bool] = set()
    for record in records:
        raw = metric_value(record, metric)
        if raw is None:
            if on_error == "raise":
                raise MissingScoreError(
                    f"item {record.item_id!r} in run {run_id!r} has no {metric!r} "
                    f"because of an error: {record.error or record.score_error}"
                )
            excluded.append(record.item_id)
            clusters[record.item_id] = record.cluster
            continue
        kinds.add(isinstance(raw, bool))
        values[record.item_id] = float(raw)
        clusters[record.item_id] = record.cluster
    if len(kinds) > 1:
        raise RecordError(f"metric {metric!r} mixes bool and numeric values in run {run_id!r}")
    if not values:
        raise RecordError(f"no usable values for {metric!r} in run {run_id!r}")
    return MetricColumn(
        metric=metric,
        run_id=run_id,
        values=values,
        clusters=clusters,
        binary=kinds == {True},
        excluded=tuple(excluded),
    )


def metric_value(record: EvalRecord, metric: str) -> float | bool | None:
    """The metric's value, or None when it is missing because the record has an error.

    A score an errored record still carries (for example a PII check that ran
    before the judge failed) is returned. Record fields are None when the model
    call failed (`error`), and returned when only scoring failed (`score_error`).
    """
    if metric in record.scores:
        return record.scores[metric]
    if metric in NUMERIC_FIELDS:
        return None if record.error is not None else getattr(record, metric)
    if record.error is not None or record.score_error is not None:
        return None
    raise RecordError(
        f"item {record.item_id!r} in run {record.run_id!r} has no metric {metric!r} "
        f"(scores: {sorted(record.scores)})"
    )


def _check_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RecordError(f"{name} must be a non-empty string, got {value!r}")


def _check_scores(scores: object) -> None:
    if not isinstance(scores, dict):
        raise RecordError(f"scores must be a dict, got {type(scores).__name__}")
    for key, value in scores.items():
        if not isinstance(key, str) or not key:
            raise RecordError(f"score names must be non-empty strings, got {key!r}")
        if key in NUMERIC_FIELDS:
            raise RecordError(f"score name {key!r} collides with a record field")
        if not isinstance(value, bool) and not _is_number(value):
            raise RecordError(f"score {key!r} must be a bool or a finite number, got {value!r}")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _reject_constant(name: str) -> float:
    raise ValueError(f"{name} is not allowed in results files")
