from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_eval_harness.analysis import pass_k_from_records
from llm_eval_harness.inspect_adapter import records_from_inspect_log
from llm_eval_harness.records import RecordError


def fake_log(scores_per_sample: list[dict[str, object]], epochs: int = 1) -> SimpleNamespace:
    samples = []
    for epoch in range(1, epochs + 1):
        for i, scores in enumerate(scores_per_sample):
            samples.append(
                SimpleNamespace(
                    id=f"s{i}",
                    epoch=epoch,
                    scores={k: SimpleNamespace(value=v) for k, v in scores.items()},
                    model_usage={
                        "openai/gpt-6-luna": SimpleNamespace(
                            input_tokens=100,
                            output_tokens=20,
                            reasoning_tokens=None,
                            total_cost=None,
                            total_tokens=120,
                        ),
                        "grader/model": SimpleNamespace(total_tokens=50),
                    },
                    total_time=1.5,
                    error=None,
                    metadata={"article": f"a{i % 2}"},
                )
            )
    return SimpleNamespace(
        eval=SimpleNamespace(eval_id="ev1", model="openai/gpt-6-luna", task="tiny"),
        samples=samples,
    )


def test_score_conversion_rules() -> None:
    log = fake_log(
        [
            {"match": "C", "partial": "P", "f1": 0.5, "multi": {"a": True, "b": 1}},
            {"match": "I", "partial": "C", "f1": 1, "multi": {"a": False, "b": 0}},
        ]
    )
    first, second = records_from_inspect_log(log, cluster_key="article")
    assert first.scores == {
        "match": True,
        "partial": 0.5,
        "f1": 0.5,
        "multi.a": True,
        "multi.b": 1.0,
    }
    assert second.scores["match"] is False
    assert second.scores["partial"] == 1.0
    assert (first.run_id, first.item_id, first.config, first.cluster) == ("ev1", "s0", "tiny", "a0")
    assert (first.tokens_in, first.tokens_out, first.latency_ms) == (100, 20, 1500.0)
    assert first.meta["cost_known"] is False
    assert first.meta["other_model_usage"] == {"grader/model": 50}


def test_epochs_become_separate_runs() -> None:
    records = records_from_inspect_log(fake_log([{"match": "C"}, {"match": "I"}], epochs=3))
    assert sorted({r.run_id for r in records}) == ["ev1-e1", "ev1-e2", "ev1-e3"]


def test_unknown_score_values_raise() -> None:
    with pytest.raises(RecordError, match="unsupported"):
        records_from_inspect_log(fake_log([{"match": "maybe"}]))
    with pytest.raises(RecordError, match="unsupported"):
        records_from_inspect_log(fake_log([{"match": [1, 2]}]))


def test_real_inspect_task_on_mockllm(tmp_path: Path) -> None:
    """Runs a two-sample Inspect task on the built-in mockllm/model. No network, no keys."""
    pytest.importorskip("inspect_ai")
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import includes
    from inspect_ai.solver import generate

    task = Task(
        dataset=[
            Sample(
                id="hit", input="Say anything", target="Default output", metadata={"article": "a1"}
            ),
            Sample(id="miss", input="Say anything", target="refund", metadata={"article": "a2"}),
        ],
        solver=generate(),
        scorer=includes(),
        name="tiny",
    )
    (log,) = inspect_eval(
        task, model="mockllm/model", log_dir=str(tmp_path), display="none", epochs=2
    )
    assert log.status == "success"
    log_file = next(tmp_path.glob("*.eval"))
    records = records_from_inspect_log(log_file, cluster_key="article")
    assert len(records) == 4
    assert {r.model for r in records} == {"mockllm/model"}
    by_item = {(r.item_id, r.meta["epoch"]): r for r in records}
    assert by_item[("hit", 1)].scores == {"includes": True}
    assert by_item[("miss", 2)].scores == {"includes": False}
    assert by_item[("hit", 1)].cluster == "a1"
    assert by_item[("hit", 1)].tokens_out > 0
    result = pass_k_from_records(records, "includes", k=2, n_boot=100)
    assert result.pass_hat_k.estimate == pytest.approx(0.5)
