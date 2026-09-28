from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval_harness.client import (
    DEFAULT_PRICES,
    BudgetExceeded,
    CacheCorrupt,
    CachedClient,
    CacheMiss,
    DollarCap,
    FakeClient,
    ModelClient,
    ModelRequest,
    ModelResponse,
    Price,
    RetryingClient,
    UnknownModelPrice,
    cost_usd,
)


def req(text: str = "hello", model: str = "gpt-6-luna", **kw: object) -> ModelRequest:
    return ModelRequest(model=model, input=text, **kw)  # type: ignore[arg-type]


def resp(**kw: object) -> ModelResponse:
    fields: dict[str, object] = {
        "text": "hi",
        "model": "gpt-6-luna",
        "input_tokens": 100,
        "output_tokens": 10,
    }
    fields.update(kw)
    return ModelResponse(**fields)  # type: ignore[arg-type]


def test_cache_key_is_canonical() -> None:
    a = req(extra={"b": 1, "a": {"y": 2, "x": 1}})
    b = req(extra={"a": {"x": 1, "y": 2}, "b": 1})
    assert a.cache_key() == b.cache_key()
    assert len(a.cache_key()) == 64
    assert a.cache_key() != req(extra={"b": 2, "a": {"y": 2, "x": 1}}).cache_key()
    assert req().cache_key() != req(reasoning_effort="low").cache_key()
    assert req().cache_key() != req(model="gpt-5-mini").cache_key()


def test_request_must_be_json_safe() -> None:
    with pytest.raises(TypeError):
        req(extra={"obj": object()}).cache_key()


def test_response_validation() -> None:
    with pytest.raises(ValueError, match="reasoning_tokens"):
        resp(output_tokens=5, reasoning_tokens=6)
    with pytest.raises(ValueError, match="cached_input_tokens"):
        resp(input_tokens=5, cached_input_tokens=6)
    with pytest.raises(ValueError, match="non-negative int"):
        resp(input_tokens=-1)


def test_fake_client_script_and_calls() -> None:
    fake = FakeClient(["one two", resp(text="custom")])
    assert isinstance(fake, ModelClient)
    first = fake.complete(req("a b c"))
    assert (first.text, first.input_tokens, first.output_tokens) == ("one two", 3, 2)
    assert fake.complete(req()).text == "custom"
    with pytest.raises(RuntimeError, match="exhausted after 2 calls"):
        fake.complete(req())
    assert len(fake.calls) == 3


def test_fake_client_callable() -> None:
    fake = FakeClient(lambda r: str(r.input).upper())
    assert fake.complete(req("abc")).text == "ABC"


def test_cache_miss_then_hit(tmp_path: Path) -> None:
    inner = FakeClient(["answer"])
    cached = CachedClient(inner, tmp_path)
    first = cached.complete(req())
    second = cached.complete(req())
    assert len(inner.calls) == 1
    assert (cached.misses, cached.hits) == (1, 1)
    assert not first.from_cache
    assert second.from_cache
    assert second.text == first.text
    path = cached.path_for(req())
    assert path.parent.name == path.stem[:2]
    stored = json.loads(path.read_text())
    assert stored["request"]["input"] == "hello"


def test_replay_only_raises_on_miss_and_never_calls_inner(tmp_path: Path) -> None:
    CachedClient(FakeClient(["recorded"]), tmp_path).complete(req("seen"))
    inner = FakeClient(["should not be used"])
    replay = CachedClient(inner, tmp_path, replay_only=True)
    assert replay.complete(req("seen")).text == "recorded"
    with pytest.raises(CacheMiss, match="gpt-6-luna"):
        replay.complete(req("unseen"))
    assert inner.calls == []
    assert CachedClient(None, tmp_path, replay_only=True).complete(req("seen")).from_cache


def test_cache_needs_inner_unless_replay_only(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inner client"):
        CachedClient(None, tmp_path)


def test_corrupt_cache_file_raises(tmp_path: Path) -> None:
    cached = CachedClient(FakeClient(["x"]), tmp_path)
    cached.complete(req())
    path = cached.path_for(req())
    payload = json.loads(path.read_text())
    payload["request"]["input"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(CacheCorrupt, match="different request"):
        cached.complete(req())
    path.write_text("{broken")
    with pytest.raises(CacheCorrupt, match="unreadable"):
        cached.complete(req())


def test_cost_from_usage_and_reasoning_bills_as_output() -> None:
    price = DEFAULT_PRICES["gpt-5-mini"]  # 0.25 in, 2.00 out, 0.025 cached
    plain = resp(input_tokens=1_000_000, output_tokens=0)
    assert cost_usd(price, plain) == pytest.approx(0.25)
    reasoning = resp(input_tokens=0, output_tokens=1_000, reasoning_tokens=800)
    # 1,000 billed output tokens (800 of them reasoning) at $2.00 per million.
    assert cost_usd(price, reasoning) == pytest.approx(0.002)
    cached = resp(input_tokens=1_000_000, cached_input_tokens=400_000, output_tokens=0)
    assert cost_usd(price, cached) == pytest.approx(0.6 * 0.25 + 0.4 * 0.025)
    llama = DEFAULT_PRICES["Llama-3.3-70B-Instruct"]
    no_cached_rate = resp(input_tokens=1_000_000, cached_input_tokens=500_000, output_tokens=0)
    assert cost_usd(llama, no_cached_rate) == pytest.approx(0.71)


def test_plan_prices() -> None:
    assert DEFAULT_PRICES["gpt-6-luna"] == Price(0.10, 0.50, 0.01)
    assert DEFAULT_PRICES["gpt-6-sol"] == Price(2.00, 10.00, 0.20)
    assert (
        DEFAULT_PRICES["gpt-5-mini"].input_per_m,
        DEFAULT_PRICES["gpt-5-mini"].output_per_m,
    ) == (
        0.25,
        2.00,
    )
    assert DEFAULT_PRICES["Llama-3.3-70B-Instruct"] == Price(0.71, 0.71)


def test_dollar_cap_blocks_once_reached() -> None:
    # Each call: 1M input tokens of gpt-6-sol = $2.00.
    inner = FakeClient(lambda r: resp(model=r.model, input_tokens=1_000_000, output_tokens=0))
    cap = DollarCap(inner, cap_usd=5.0)
    for _ in range(3):
        cap.complete(req(model="gpt-6-sol"))
    assert cap.spent_usd == pytest.approx(6.0)  # the third call pushed past the cap
    with pytest.raises(BudgetExceeded, match=r"\$6.0000 of the \$5.00 cap"):
        cap.complete(req(model="gpt-6-sol"))
    assert len(inner.calls) == 3


def test_dollar_cap_refuses_unknown_models_before_calling() -> None:
    inner = FakeClient(["x"])
    with pytest.raises(UnknownModelPrice, match="mystery-model"):
        DollarCap(inner, cap_usd=1.0).complete(req(model="mystery-model"))
    assert inner.calls == []


def test_dollar_cap_does_not_charge_cache_hits(tmp_path: Path) -> None:
    inner = FakeClient(lambda r: resp(input_tokens=1_000_000, output_tokens=0))
    cap = DollarCap(CachedClient(inner, tmp_path), cap_usd=1.0)
    cap.complete(req())
    cap.complete(req())
    assert cap.spent_usd == pytest.approx(0.10)
    assert cap.calls == 2


class Flaky(Exception):
    pass


def test_retrying_client_retries_listed_errors_with_jitter() -> None:
    attempts = {"n": 0}

    def script(request: ModelRequest) -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise Flaky("429")
        return "ok"

    sleeps: list[float] = []
    client = RetryingClient(FakeClient(script), (Flaky,), base_delay_s=1.0, sleep=sleeps.append)
    assert client.complete(req()).text == "ok"
    assert client.retries == 2
    assert 0 <= sleeps[0] <= 1.0
    assert 0 <= sleeps[1] <= 2.0


def test_retrying_client_gives_up_and_passes_other_errors() -> None:
    def always(request: ModelRequest) -> str:
        raise Flaky("still down")

    sleeps: list[float] = []
    client = RetryingClient(FakeClient(always), (Flaky,), max_attempts=3, sleep=sleeps.append)
    with pytest.raises(Flaky):
        client.complete(req())
    assert len(sleeps) == 2

    def broken(request: ModelRequest) -> str:
        raise KeyError("bug")

    client = RetryingClient(FakeClient(broken), (Flaky,), sleep=sleeps.append)
    with pytest.raises(KeyError):
        client.complete(req())
    assert len(sleeps) == 2
