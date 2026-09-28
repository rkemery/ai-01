"""Convert Inspect AI eval logs into `EvalRecord`s.

Needs the `inspect` extra (inspect-ai==0.3.271). Other repos only read the
JSONL this produces, so they never need Inspect installed.

Mapping:

- run_id: the log's eval_id, with "-e<epoch>" appended when the log has more
  than one epoch (each epoch is one trial, which is what pass^k expects)
- item_id: the sample id
- scores: one entry per scorer. A scorer whose values are all booleans or
  Inspect's "C" / "I" becomes boolean. Otherwise values become floats with
  C = 1, I = 0, P = 0.5, N = 0. Dict-valued scores are flattened to
  "scorer.key". Anything else raises.
- tokens: usage of the evaluated model only. Usage by other models (for
  example a model-graded scorer) goes into meta["other_model_usage"].
- cost_usd: Inspect's total_cost when it is known, otherwise 0.0 with
  meta["cost_known"] = False.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llm_eval_harness.records import EvalRecord, RecordError

_STRING_VALUES = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}
_BINARY_STRINGS = {"C", "I"}


def _inspect_log_module() -> Any:
    try:
        from inspect_ai import log
    except ImportError as exc:
        raise ImportError(
            "llm_eval_harness.inspect_adapter needs inspect-ai, which ships with the "
            "'inspect' extra. Install it with: pip install 'llm-eval-harness[inspect]'"
        ) from exc
    return log


def records_from_inspect_log(
    log_or_path: Any,
    *,
    config: str | None = None,
    cluster_key: str | None = None,
) -> list[EvalRecord]:
    """Convert one Inspect log (an `EvalLog` or a path to one) into records.

    `config` defaults to the task name. `cluster_key` names a sample metadata
    field to use as the record's cluster.
    """
    log = log_or_path
    if isinstance(log_or_path, str | Path):
        log = _inspect_log_module().read_eval_log(str(log_or_path))
    if log.samples is None:
        raise RecordError("the Inspect log has no samples (was it read with header_only?)")
    model = log.eval.model
    epochs = {sample.epoch for sample in log.samples}
    raw_scores = [_flatten_scores(sample.scores or {}) for sample in log.samples]
    binary_keys = _binary_keys(raw_scores)
    records = []
    for sample, raw in zip(log.samples, raw_scores, strict=True):
        run_id = log.eval.eval_id if len(epochs) == 1 else f"{log.eval.eval_id}-e{sample.epoch}"
        usage = sample.model_usage or {}
        own = usage.get(model)
        others = {name: u.total_tokens for name, u in usage.items() if name != model}
        meta: dict[str, Any] = {
            "source": "inspect_ai",
            "task": log.eval.task,
            "epoch": sample.epoch,
            "cost_known": own is not None and own.total_cost is not None,
        }
        if others:
            meta["other_model_usage"] = others
        records.append(
            EvalRecord(
                run_id=run_id,
                item_id=str(sample.id),
                config=config or log.eval.task,
                model=model,
                scores={k: _convert(v, k in binary_keys) for k, v in raw.items()},
                cluster=_cluster(sample, cluster_key),
                tokens_in=own.input_tokens if own else 0,
                tokens_out=own.output_tokens if own else 0,
                reasoning_tokens=(own.reasoning_tokens or 0) if own else 0,
                cost_usd=(own.total_cost or 0.0) if own else 0.0,
                latency_ms=(sample.total_time or 0.0) * 1000.0,
                error=sample.error.message if sample.error else None,
                meta=meta,
            )
        )
    return records


def _flatten_scores(scores: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for name, score in scores.items():
        value = score.value
        if isinstance(value, Mapping):
            for key, sub in value.items():
                flat[f"{name}.{key}"] = sub
        else:
            flat[name] = value
    return flat


def _binary_keys(all_scores: list[dict[str, Any]]) -> set[str]:
    keys = {key for scores in all_scores for key in scores}
    return {
        key
        for key in keys
        if all(
            isinstance(s[key], bool) or (isinstance(s[key], str) and s[key] in _BINARY_STRINGS)
            for s in all_scores
            if key in s
        )
    }


def _convert(value: Any, binary: bool) -> float | bool:
    if isinstance(value, bool):
        return value if binary else float(value)
    if isinstance(value, str):
        if value not in _STRING_VALUES:
            raise RecordError(f"unsupported Inspect score value {value!r}")
        return value == "C" if binary else _STRING_VALUES[value]
    if isinstance(value, int | float):
        return float(value)
    raise RecordError(f"unsupported Inspect score value {value!r} ({type(value).__name__})")


def _cluster(sample: Any, cluster_key: str | None) -> str | None:
    if cluster_key is None:
        return None
    value = (sample.metadata or {}).get(cluster_key)
    if not isinstance(value, str) or not value:
        raise RecordError(f"sample {sample.id!r} has no string metadata {cluster_key!r}")
    return value
