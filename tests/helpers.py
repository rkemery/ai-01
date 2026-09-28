from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_eval_harness.records import EvalRecord


def record(item_id: str = "q1", run_id: str = "run-a", **overrides: Any) -> EvalRecord:
    fields: dict[str, Any] = {
        "run_id": run_id,
        "item_id": item_id,
        "config": "cfg",
        "model": "gpt-6-luna",
        "scores": {"correct": True},
    }
    fields.update(overrides)
    return EvalRecord(**fields)


def run(
    run_id: str,
    values: Mapping[str, bool | float],
    metric: str = "correct",
    clusters: Mapping[str, str] | None = None,
) -> list[EvalRecord]:
    return [
        record(
            item_id,
            run_id,
            scores={metric: value},
            cluster=None if clusters is None else clusters[item_id],
        )
        for item_id, value in values.items()
    ]
