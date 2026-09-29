"""FoundryClient tests with the SDK mocked. No network, and no Azure credentials are read.

Live key and Entra ID auth are not exercised here.
"""

from __future__ import annotations

import inspect
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from llm_eval_harness import azure
from llm_eval_harness.client import ModelRequest


class FakeOpenAIModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("openai")
        self.created: list[dict[str, Any]] = []
        module = self

        class OpenAI:
            def __init__(self, **kwargs: Any) -> None:
                module.created.append(kwargs)

        self.OpenAI = OpenAI


class FakeIdentityModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("azure.identity")
        self.scopes: list[str] = []
        module = self

        class DefaultAzureCredential:
            pass

        def get_bearer_token_provider(credential: object, *scopes: str) -> Any:
            assert isinstance(credential, DefaultAzureCredential)
            module.scopes.extend(scopes)
            return lambda: "token-from-entra"

        self.DefaultAzureCredential = DefaultAzureCredential
        self.get_bearer_token_provider = get_bearer_token_provider


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> FakeOpenAIModule:
    module = FakeOpenAIModule()
    monkeypatch.setitem(sys.modules, "openai", module)
    return module


@pytest.fixture
def fake_identity(monkeypatch: pytest.MonkeyPatch) -> FakeIdentityModule:
    module = FakeIdentityModule()
    monkeypatch.setitem(sys.modules, "azure.identity", module)
    return module


BASE_URL = "https://example.test/openai/v1/"


def test_key_auth_uses_key_env_url_and_no_sdk_retries(fake_openai: FakeOpenAIModule) -> None:
    azure.build_sdk_client({"AZURE_OPENAI_API_KEY": "k-123", "AZURE_OPENAI_BASE_URL": BASE_URL})
    (kwargs,) = fake_openai.created
    assert kwargs == {"base_url": BASE_URL, "api_key": "k-123", "max_retries": 0}


def test_missing_base_url_raises(fake_openai: FakeOpenAIModule) -> None:
    with pytest.raises(ValueError, match="AZURE_OPENAI_BASE_URL"):
        azure.build_sdk_client({"AZURE_OPENAI_API_KEY": "k-123"})
    assert fake_openai.created == []


def test_entra_auth_when_no_key(
    fake_openai: FakeOpenAIModule, fake_identity: FakeIdentityModule
) -> None:
    azure.build_sdk_client({"AZURE_OPENAI_BASE_URL": "https://example.test/openai/v1/"})
    (kwargs,) = fake_openai.created
    assert kwargs["base_url"] == "https://example.test/openai/v1/"
    assert kwargs["max_retries"] == 0
    assert callable(kwargs["api_key"])
    assert kwargs["api_key"]() == "token-from-entra"
    assert fake_identity.scopes == [azure.DEFAULT_TOKEN_SCOPE]


def test_entra_scope_from_env(
    fake_openai: FakeOpenAIModule, fake_identity: FakeIdentityModule
) -> None:
    azure.build_sdk_client(
        {"AZURE_OPENAI_TOKEN_SCOPE": "api://custom/.default", "AZURE_OPENAI_BASE_URL": BASE_URL}
    )
    assert fake_identity.scopes == ["api://custom/.default"]


def test_missing_sdk_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)  # makes `import openai` raise ImportError
    with pytest.raises(ImportError, match=r"llm-eval-harness\[azure\]"):
        azure.build_sdk_client({"AZURE_OPENAI_API_KEY": "k"})


def test_request_kwargs_passes_reasoning_effort_and_extra() -> None:
    request = ModelRequest(
        model="gpt-6-luna",
        input="hi",
        instructions="be brief",
        max_output_tokens=64,
        reasoning_effort="none",
        extra={"prompt_cache_key": "rag-v1"},
    )
    assert azure.request_kwargs(request) == {
        "model": "gpt-6-luna",
        "input": "hi",
        "instructions": "be brief",
        "max_output_tokens": 64,
        "reasoning": {"effort": "none"},
        "prompt_cache_key": "rag-v1",
    }


def test_temperature_is_refused_for_gpt6_and_kept_for_llama() -> None:
    with pytest.raises(ValueError, match="default temperature"):
        azure.request_kwargs(ModelRequest(model="gpt-6-sol", input="x", temperature=0.0))
    kwargs = azure.request_kwargs(
        ModelRequest(model="Llama-3.3-70B-Instruct", input="x", temperature=0.0)
    )
    assert kwargs["temperature"] == 0.0
    assert "temperature" not in azure.request_kwargs(ModelRequest(model="gpt-6-luna", input="x"))


def test_extra_cannot_override_first_class_fields() -> None:
    with pytest.raises(ValueError, match="repeats"):
        azure.request_kwargs(ModelRequest(model="gpt-6-luna", input="x", extra={"model": "y"}))


def _sdk_response(status: str = "completed", usage: Any = "default") -> SimpleNamespace:
    if usage == "default":
        usage = SimpleNamespace(
            input_tokens=1200,
            output_tokens=90,
            input_tokens_details=SimpleNamespace(cached_tokens=1024),
            output_tokens_details=SimpleNamespace(reasoning_tokens=40),
        )
    return SimpleNamespace(
        id="resp_1",
        model="gpt-5-mini-2025-08-07",
        status=status,
        incomplete_details=SimpleNamespace(reason="max_output_tokens")
        if status == "incomplete"
        else None,
        output_text="the answer",
        usage=usage,
    )


class FakeSDK:
    def __init__(self, response: SimpleNamespace) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses = SimpleNamespace(create=self._create)
        self._response = response

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self._response


def test_foundry_client_complete_maps_usage_and_latency() -> None:
    sdk = FakeSDK(_sdk_response())
    ticks = iter([10.0, 10.25])
    client = azure.FoundryClient(sdk, clock=lambda: next(ticks))
    out = client.complete(ModelRequest(model="gpt-5-mini", input="q", reasoning_effort="minimal"))
    assert sdk.calls == [{"model": "gpt-5-mini", "input": "q", "reasoning": {"effort": "minimal"}}]
    assert out.text == "the answer"
    assert (out.input_tokens, out.output_tokens) == (1200, 90)
    assert (out.reasoning_tokens, out.cached_input_tokens) == (40, 1024)
    assert out.latency_ms == pytest.approx(250.0)
    assert out.finish_reason == "stop"


def test_foundry_client_reports_truncation_and_refuses_missing_usage() -> None:
    out = azure.FoundryClient(FakeSDK(_sdk_response("incomplete"))).complete(
        ModelRequest(model="gpt-6-luna", input="q")
    )
    assert out.finish_reason == "max_output_tokens"
    with pytest.raises(ValueError, match="no usage"):
        azure.FoundryClient(FakeSDK(_sdk_response(usage=None))).complete(
            ModelRequest(model="gpt-6-luna", input="q")
        )


def test_names_used_exist_in_the_installed_sdk() -> None:
    """Guards against SDK renames: every name the client touches must exist in openai."""
    openai = pytest.importorskip("openai")
    from openai.resources.responses import Responses
    from openai.types.responses import Response, ResponseUsage
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

    init = inspect.signature(openai.OpenAI.__init__).parameters
    assert {"api_key", "base_url", "max_retries"} <= set(init)
    create = inspect.signature(Responses.create).parameters
    assert {
        "model",
        "input",
        "instructions",
        "max_output_tokens",
        "reasoning",
        "temperature",
    } <= set(create)
    assert {"id", "model", "status", "incomplete_details", "usage"} <= set(Response.model_fields)
    assert isinstance(inspect.getattr_static(Response, "output_text"), property)
    assert {
        "input_tokens",
        "output_tokens",
        "input_tokens_details",
        "output_tokens_details",
    } <= set(ResponseUsage.model_fields)
    assert "cached_tokens" in InputTokensDetails.model_fields
    assert "reasoning_tokens" in OutputTokensDetails.model_fields
    assert all(issubclass(e, openai.OpenAIError) for e in azure.retryable_errors())


def test_real_sdk_client_is_configured_without_network() -> None:
    """Build a real openai.OpenAI with a dummy key. Construction makes no request."""
    pytest.importorskip("openai")
    client = azure.build_sdk_client(
        {"AZURE_OPENAI_API_KEY": "dummy-not-a-real-key", "AZURE_OPENAI_BASE_URL": BASE_URL}
    )
    assert client.max_retries == 0
    assert str(client.base_url) == BASE_URL


def test_gpt6_accepts_its_default_temperature() -> None:
    """Review finding: temperature=1.0, the default, was refused too."""
    kwargs = azure.request_kwargs(ModelRequest(model="gpt-6-luna", input="x", temperature=1.0))
    assert kwargs["temperature"] == 1.0
    with pytest.raises(ValueError, match="default temperature"):
        azure.request_kwargs(ModelRequest(model="gpt-6-luna", input="x", temperature=0.7))


def test_trial_is_not_sent_to_the_model() -> None:
    kwargs = azure.request_kwargs(ModelRequest(model="gpt-6-luna", input="x", trial=3))
    assert "trial" not in kwargs
