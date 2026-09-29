"""Azure AI Foundry client over the OpenAI v1 endpoint and the Responses API.

Needs the `azure` extra: pip install "llm-eval-harness[azure]".

Configuration comes from the environment:

- AZURE_OPENAI_BASE_URL     (required, e.g. https://<resource>.openai.azure.com/openai/v1/)
- AZURE_OPENAI_API_KEY      (if set, key auth is used)
- AZURE_OPENAI_TOKEN_SCOPE  (Entra ID scope, default cognitiveservices)

Without a key, auth goes through Entra ID: `DefaultAzureCredential` wrapped by
`azure.identity.get_bearer_token_provider`, which the OpenAI SDK accepts as a
callable `api_key` and calls for a fresh token on each request.

SDK retries are off (`max_retries=0`). Wrap the client in `RetryingClient`
with `retryable_errors()` to get a retry policy you can see and test.

Tested only with the SDK mocked. The live Entra ID and key paths were not
exercised by the tests in this repo.
"""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

from llm_eval_harness.client import ModelRequest, ModelResponse

DEFAULT_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"

# gpt-6 deployments reject any temperature other than the default, 1.0.
_DEFAULT_TEMPERATURE_ONLY = ("gpt-6",)


def _require(module: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"llm_eval_harness.azure needs the {module!r} package, which ships with the "
            "'azure' extra. Install it with: pip install 'llm-eval-harness[azure]' "
            "(or: uv add 'llm-eval-harness[azure]')"
        ) from exc


def build_sdk_client(env: Mapping[str, str] | None = None) -> Any:
    """Create an `openai.OpenAI` client for the Foundry v1 endpoint with max_retries=0."""
    env = os.environ if env is None else env
    openai = _require("openai")
    base_url = env.get("AZURE_OPENAI_BASE_URL")
    if not base_url:
        raise ValueError(
            "AZURE_OPENAI_BASE_URL is not set. Point it at your Foundry resource's v1 "
            "endpoint, e.g. https://<resource>.openai.azure.com/openai/v1/"
        )
    api_key: str | Callable[[], str]
    key = env.get("AZURE_OPENAI_API_KEY")
    if key:
        api_key = key
    else:
        identity = _require("azure.identity")
        scope = env.get("AZURE_OPENAI_TOKEN_SCOPE") or DEFAULT_TOKEN_SCOPE
        api_key = identity.get_bearer_token_provider(identity.DefaultAzureCredential(), scope)
    return openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0)


def retryable_errors() -> tuple[type[BaseException], ...]:
    """OpenAI SDK errors worth retrying: rate limits, timeouts, connection and 5xx errors."""
    openai = _require("openai")
    return (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )


def request_kwargs(request: ModelRequest) -> dict[str, Any]:
    """Translate a `ModelRequest` into `client.responses.create(**kwargs)` arguments."""
    kwargs: dict[str, Any] = {"model": request.model, "input": request.input}
    if request.instructions is not None:
        kwargs["instructions"] = request.instructions
    if request.max_output_tokens is not None:
        kwargs["max_output_tokens"] = request.max_output_tokens
    if request.reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": request.reasoning_effort}
    if request.temperature is not None:
        if request.model.startswith(_DEFAULT_TEMPERATURE_ONLY) and request.temperature != 1.0:
            raise ValueError(
                f"{request.model!r} only accepts the default temperature, 1.0, got "
                f"{request.temperature}. Leave temperature unset or pass 1.0."
            )
        kwargs["temperature"] = request.temperature
    clash = set(request.extra) & set(kwargs)
    if clash:
        raise ValueError(f"extra repeats first-class request fields: {sorted(clash)}")
    kwargs.update(request.extra)
    return kwargs


def parse_response(response: Any, latency_ms: float) -> ModelResponse:
    """Build a `ModelResponse` from an OpenAI `Response` object.

    Raises `ValueError` when usage is missing, because a call with no usage cannot
    be billed or checked against the cap.
    """
    usage = response.usage
    if usage is None:
        raise ValueError(f"response {response.id!r} has no usage")
    output_details = usage.output_tokens_details
    input_details = usage.input_tokens_details
    finish_reason = "stop"
    if response.status == "incomplete":
        details = response.incomplete_details
        finish_reason = details.reason if details is not None and details.reason else "incomplete"
    elif response.status != "completed":
        finish_reason = response.status or "unknown"
    return ModelResponse(
        text=response.output_text,
        model=response.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=(output_details.reasoning_tokens or 0) if output_details else 0,
        cached_input_tokens=(input_details.cached_tokens or 0) if input_details else 0,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
    )


class FoundryClient:
    """`ModelClient` for Azure Foundry deployments through the Responses API.

    Pass `sdk_client` to inject a preconfigured (or fake) OpenAI client.
    Otherwise one is built from the environment by `build_sdk_client`.
    """

    def __init__(
        self,
        sdk_client: Any | None = None,
        *,
        env: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._sdk = sdk_client if sdk_client is not None else build_sdk_client(env)
        self._clock = clock

    def complete(self, request: ModelRequest) -> ModelResponse:
        kwargs = request_kwargs(request)
        start = self._clock()
        response = self._sdk.responses.create(**kwargs)
        latency_ms = (self._clock() - start) * 1000.0
        return parse_response(response, latency_ms)
