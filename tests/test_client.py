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
    input_token_bound,
    max_cost_usd,
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


def test_dollar_cap_refuses_the_call_that_could_pass_the_cap() -> None:
    # Each call bills 100k output tokens of gpt-6-sol ($1.00) and 10 input tokens.
    inner = FakeClient(lambda r: resp(model=r.model, input_tokens=10, output_tokens=100_000))
    cap = DollarCap(inner, cap_usd=5.0)
    request = req(model="gpt-6-sol", max_output_tokens=100_000)
    for _ in range(4):
        cap.complete(request)
    assert cap.spent_usd == pytest.approx(4 * (10 * 2.0 + 100_000 * 10.0) / 1e6)
    # A fifth call could cost a little over $1.00, which would pass $5.00.
    with pytest.raises(BudgetExceeded, match=r"\$4\.0001 of the \$5\.00 cap"):
        cap.complete(request)
    assert len(inner.calls) == 4


def test_dollar_cap_refuses_unknown_models_before_calling() -> None:
    inner = FakeClient(["x"])
    with pytest.raises(UnknownModelPrice, match="mystery-model"):
        DollarCap(inner, cap_usd=1.0).complete(req(model="mystery-model"))
    assert inner.calls == []


def test_dollar_cap_does_not_charge_cache_hits(tmp_path: Path) -> None:
    inner = FakeClient(lambda r: resp(input_tokens=1_000_000, output_tokens=0))
    cap = DollarCap(CachedClient(inner, tmp_path), cap_usd=1.0)
    cap.complete(req(max_output_tokens=10))
    cap.complete(req(max_output_tokens=10))
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


# Review findings: trials, cache key normalization, incomplete replies, the cap's overshoot.


def test_trials_of_one_request_get_their_own_cache_entries(tmp_path: Path) -> None:
    """Without a trial index, 3 trials returned one cached reply 3 times, so pass^k = pass^1."""
    replies = iter(["answer A", "answer B", "answer C"])
    cached = CachedClient(FakeClient(lambda r: next(replies)), tmp_path)
    first = [cached.complete(req("same task", trial=t)).text for t in range(3)]
    assert first == ["answer A", "answer B", "answer C"]
    assert (cached.hits, cached.misses) == (0, 3)
    replay = CachedClient(None, tmp_path, replay_only=True)
    assert [replay.complete(req("same task", trial=t)).text for t in range(3)] == first


def test_trial_must_be_a_non_negative_int() -> None:
    for bad in (-1, 1.0, True):
        with pytest.raises(ValueError, match="trial"):
            req(trial=bad)


def test_cache_key_treats_equal_numbers_alike() -> None:
    assert req(temperature=0).cache_key() == req(temperature=0.0).cache_key()
    assert req(extra={"top_p": 1}).cache_key() == req(extra={"top_p": 1.0}).cache_key()
    assert req(temperature=0.5).cache_key() != req(temperature=0).cache_key()


def test_incomplete_replies_are_not_cached(tmp_path: Path) -> None:
    inner = FakeClient([resp(finish_reason="max_output_tokens"), resp(text="full")])
    cached = CachedClient(inner, tmp_path)
    assert cached.complete(req()).finish_reason == "max_output_tokens"
    assert not cached.path_for(req()).exists()
    assert cached.not_stored == 1
    assert cached.complete(req()).text == "full"
    assert cached.complete(req()).from_cache
    assert len(inner.calls) == 2


def test_dollar_cap_refuses_a_call_that_could_pass_the_cap() -> None:
    """Review case: a $1 cap let one gpt-6-sol call spend $20."""
    inner = FakeClient([resp(model="gpt-6-sol", input_tokens=10, output_tokens=2_000_000)])
    cap = DollarCap(inner, cap_usd=1.0)
    with pytest.raises(BudgetExceeded, match="could cost up to"):
        cap.complete(req(model="gpt-6-sol", max_output_tokens=2_000_000))
    assert inner.calls == []
    assert cap.spent_usd == 0.0


def test_dollar_cap_needs_max_output_tokens() -> None:
    inner = FakeClient(["x"])
    with pytest.raises(ValueError, match="max_output_tokens"):
        DollarCap(inner, cap_usd=1.0).complete(req())
    assert inner.calls == []


def test_dollar_cap_never_passes_the_cap_even_in_the_worst_case() -> None:
    """Every call bills its worst case: one input token per byte plus the allowance, and
    max_output_tokens of output. Spend still stays under the cap."""
    request = req("hello world " * 50, model="gpt-6-sol", max_output_tokens=4000)
    worst = max_cost_usd(DEFAULT_PRICES["gpt-6-sol"], request)
    inner = FakeClient(
        lambda r: resp(
            model=r.model, input_tokens=input_token_bound(r), output_tokens=r.max_output_tokens
        )
    )
    cap = DollarCap(inner, cap_usd=1.0)
    calls = int(1.0 // worst)  # as many worst-case calls as fit
    for _ in range(calls):
        cap.complete(request)
    with pytest.raises(BudgetExceeded):
        cap.complete(request)
    assert len(inner.calls) == calls
    assert cap.spent_usd <= 1.0
    assert cap.spent_usd > 1.0 - worst  # it stops only when one more call might not fit
    assert cap.spent_usd == pytest.approx(len(inner.calls) * worst)


def test_dollar_cap_charges_the_worst_case_for_calls_that_raise() -> None:
    """Re-review finding: 20 retried timeouts under a $0.01 cap left spend at $0.

    A timed-out request may still be billed in full, so each attempt that raises
    is charged its reservation, and retries stop when the next one might not fit.
    """
    attempts = {"n": 0}

    def timeout(request: ModelRequest) -> str:
        attempts["n"] += 1
        raise TimeoutError("the server may still bill this")

    request = req(model="gpt-6-sol", max_output_tokens=400)
    worst = max_cost_usd(DEFAULT_PRICES["gpt-6-sol"], request)
    cap = DollarCap(FakeClient(timeout), cap_usd=0.01)
    client = RetryingClient(cap, (TimeoutError,), max_attempts=4, sleep=lambda s: None)
    fits = int(0.01 // worst)
    assert fits == 2
    # Attempts 1 and 2 time out and are retried. Attempt 3 might not fit, so it is refused.
    with pytest.raises(BudgetExceeded):
        client.complete(request)
    assert attempts["n"] == fits
    assert cap.spent_usd == pytest.approx(fits * worst)
    assert cap.charged_for_errors_usd == pytest.approx(fits * worst)
    assert cap.spent_usd <= 0.01
