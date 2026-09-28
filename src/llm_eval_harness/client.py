"""Model client protocol and the wrappers every live run goes through.

Typical stack, outermost first:

    DollarCap(CachedClient(RetryingClient(FoundryClient(), retry_on=...), "cache/"), cap_usd=5)

The cap sees every call, the cache answers repeats for free, and only cache
misses reach the network. In CI the cache runs with `replay_only=True` and no
inner client, so a missing entry fails loudly instead of calling a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelRequest:
    """Everything that determines a model's reply. Its canonical JSON is the cache key.

    `input` is a string or a list of chat messages ({"role": ..., "content": ...}).
    `extra` holds any other request parameters and is passed through to the SDK.
    """

    model: str
    input: str | list[dict[str, str]]
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.input, str | list):
            raise TypeError("input must be a string or a list of messages")
        if self.max_output_tokens is not None and (
            type(self.max_output_tokens) is not int or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive int")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        """Sorted keys, no whitespace, UTF-8 kept as is. Raises TypeError if not JSON-safe."""
        return _canonical(self.to_dict())

    def cache_key(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelResponse:
    """A model reply with the usage needed for cost and the results contract.

    `output_tokens` is the billed output count and includes `reasoning_tokens`,
    as in the OpenAI `usage` object. `cached_input_tokens` is the part of
    `input_tokens` served from the provider's prompt cache. `from_cache` is True
    when the reply came from our disk cache and cost nothing this time.
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    from_cache: bool = False

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning_tokens are part of output_tokens and cannot exceed it")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")


@runtime_checkable
class ModelClient(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse: ...


Reply = str | ModelResponse


class FakeClient:
    """Deterministic client for tests and offline demos.

    `script` is either a sequence of replies used in order (running out raises
    `RuntimeError`) or a function from request to reply. A string reply becomes a
    `ModelResponse` with whitespace-split word counts standing in for tokens.
    Every request is kept in `calls`.
    """

    def __init__(
        self,
        script: Sequence[Reply] | Callable[[ModelRequest], Reply],
        *,
        latency_ms: float = 0.0,
    ) -> None:
        self._script = script
        self._next = 0
        self._latency_ms = latency_ms
        self.calls: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        if callable(self._script):
            reply = self._script(request)
        else:
            if self._next >= len(self._script):
                raise RuntimeError(f"FakeClient script exhausted after {self._next} calls")
            reply = self._script[self._next]
            self._next += 1
        if isinstance(reply, ModelResponse):
            return reply
        if isinstance(reply, str):
            return ModelResponse(
                text=reply,
                model=request.model,
                input_tokens=_word_count(request),
                output_tokens=len(reply.split()),
                latency_ms=self._latency_ms,
            )
        raise TypeError(f"FakeClient script must yield str or ModelResponse, got {reply!r}")


class CacheMiss(LookupError):
    """No cached response exists and the cache is in replay-only mode."""


class CacheCorrupt(ValueError):
    """A cache file does not match the request that maps to it, or is unreadable."""


class CachedClient:
    """Disk cache in front of another client, keyed by sha256 of the canonical request JSON.

    Files live at `<cache_dir>/<key[:2]>/<key>.json` and hold both the request and
    the response, so a cache directory can be read and diffed by hand. Writes are
    atomic (temp file then rename). A hit returns the stored response with
    `from_cache=True` and the latency measured on the original call.

    With `replay_only=True` a miss raises `CacheMiss` and the inner client is
    never called. Use that in CI.
    """

    def __init__(
        self, inner: ModelClient | None, cache_dir: str | Path, *, replay_only: bool = False
    ) -> None:
        if inner is None and not replay_only:
            raise ValueError("an inner client is required unless replay_only=True")
        self._inner = inner
        self.cache_dir = Path(cache_dir)
        self.replay_only = replay_only
        self.hits = 0
        self.misses = 0

    def path_for(self, request: ModelRequest) -> Path:
        key = request.cache_key()
        return self.cache_dir / key[:2] / f"{key}.json"

    def complete(self, request: ModelRequest) -> ModelResponse:
        path = self.path_for(request)
        if path.exists():
            self.hits += 1
            return _load_cached(path, request)
        self.misses += 1
        if self.replay_only or self._inner is None:
            raise CacheMiss(
                f"no cached response for model {request.model!r} (key {path.stem}) in "
                f"{self.cache_dir}. Record it with a live client and replay_only=False."
            )
        response = self._inner.complete(request)
        _store_cached(path, request, response)
        return response


@dataclass(frozen=True)
class Price:
    """USD per million tokens. Cached input falls back to the full input rate when unset."""

    input_per_m: float
    output_per_m: float
    cached_input_per_m: float | None = None


# List prices from PLAN.md (Azure Foundry, Global Standard, 2026-09-28). Keys are
# deployment names, which is what requests carry.
DEFAULT_PRICES: dict[str, Price] = {
    "gpt-6-luna": Price(0.10, 0.50, 0.01),
    "gpt-6-sol": Price(2.00, 10.00, 0.20),
    "gpt-5-mini": Price(0.25, 2.00, 0.025),
    "Llama-3.3-70B-Instruct": Price(0.71, 0.71),
}


class BudgetExceeded(RuntimeError):
    """The dollar cap has been reached, so the next call was refused."""


class UnknownModelPrice(ValueError):
    """A request named a model that has no entry in the price table."""


def cost_usd(price: Price, response: ModelResponse) -> float:
    """Cost of one call from its usage. Reasoning tokens bill as output (they are in output_tokens).

    cost = ((input - cached) * p_in + cached * p_cached + output * p_out) / 1e6
    """
    cached_rate = (
        price.input_per_m if price.cached_input_per_m is None else price.cached_input_per_m
    )
    uncached = response.input_tokens - response.cached_input_tokens
    total = (
        uncached * price.input_per_m
        + response.cached_input_tokens * cached_rate
        + response.output_tokens * price.output_per_m
    )
    return total / 1_000_000


class DollarCap:
    """Client wrapper that tracks spend from `usage` and stops at a hard cap.

    The check happens before each call: once `spent_usd >= cap_usd` every further
    call raises `BudgetExceeded`. Spend can therefore pass the cap by at most the
    cost of the last call made. A model missing from the price table is refused
    before any call. Replies served from the disk cache cost nothing. Not
    thread-safe: share one instance per thread, or add a lock if you need to.
    """

    def __init__(
        self,
        inner: ModelClient,
        cap_usd: float,
        prices: Mapping[str, Price] | None = None,
    ) -> None:
        if cap_usd <= 0:
            raise ValueError(f"cap_usd must be positive, got {cap_usd}")
        self._inner = inner
        self.cap_usd = cap_usd
        self.prices = dict(DEFAULT_PRICES if prices is None else prices)
        self.spent_usd = 0.0
        self.calls = 0

    def price_for(self, model: str) -> Price:
        try:
            return self.prices[model]
        except KeyError:
            raise UnknownModelPrice(
                f"no price for model {model!r}. Known: {sorted(self.prices)}"
            ) from None

    def complete(self, request: ModelRequest) -> ModelResponse:
        price = self.price_for(request.model)
        if self.spent_usd >= self.cap_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} of the ${self.cap_usd:.2f} cap, "
                f"refusing a call to {request.model!r}"
            )
        response = self._inner.complete(request)
        self.calls += 1
        if not response.from_cache:
            self.spent_usd += cost_usd(price, response)
        return response


class RetryingClient:
    """Retry a client on the given exception types with exponential backoff and full jitter.

    The delay before retry number r (1-based) is uniform on
    [0, min(max_delay_s, base_delay_s * 2**(r - 1))]. The jitter RNG is seeded, and
    `sleep` can be swapped for a fake in tests. Only exception types listed in
    `retry_on` are retried. Anything else, and the last failure, propagates.
    """

    def __init__(
        self,
        inner: ModelClient,
        retry_on: tuple[type[BaseException], ...],
        *,
        max_attempts: int = 4,
        base_delay_s: float = 1.0,
        max_delay_s: float = 30.0,
        seed: int = 0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not retry_on:
            raise ValueError("retry_on must name at least one exception type")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._inner = inner
        self.retry_on = retry_on
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self._rng = random.Random(seed)
        self._sleep = sleep
        self.retries = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._inner.complete(request)
            except self.retry_on:
                if attempt == self.max_attempts:
                    raise
                ceiling = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
                self.retries += 1
                self._sleep(self._rng.uniform(0.0, ceiling))
        raise AssertionError("unreachable")  # pragma: no cover


def _canonical(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _word_count(request: ModelRequest) -> int:
    parts = [request.instructions or ""]
    if isinstance(request.input, str):
        parts.append(request.input)
    else:
        parts.extend(str(message.get("content", "")) for message in request.input)
    return sum(len(part.split()) for part in parts)


_RESPONSE_FIELDS = {f.name for f in fields(ModelResponse)} - {"from_cache"}


def _store_cached(path: Path, request: ModelRequest, response: ModelResponse) -> None:
    payload = {
        "request": request.to_dict(),
        "response": {k: v for k, v in asdict(response).items() if k in _RESPONSE_FIELDS},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _load_cached(path: Path, request: ModelRequest) -> ModelResponse:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stored_request = payload["request"]
        stored_response = payload["response"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CacheCorrupt(f"unreadable cache file {path}: {exc}") from exc
    if _canonical(stored_request) != request.canonical_json():
        raise CacheCorrupt(f"cache file {path} holds a different request than its key")
    if set(stored_response) != _RESPONSE_FIELDS:
        raise CacheCorrupt(f"cache file {path} has unexpected response fields")
    try:
        return ModelResponse(**stored_response, from_cache=True)
    except (TypeError, ValueError) as exc:
        raise CacheCorrupt(f"invalid cached response in {path}: {exc}") from exc
