"""OllamaClient unit tests (HTTP intercepted; no real server needed)."""

from __future__ import annotations

import logging

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import (
    LLMModelNotFoundError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.base import LLMClient, LLMHealth, SupportsHealthCheck
from app.llm.ollama_client import OllamaClient
from tests.conftest import FakeOllama


def test_implements_llm_client_protocol():
    client = FakeOllama().client()
    assert isinstance(client, LLMClient)
    assert isinstance(client, SupportsHealthCheck)


def test_generate_sends_chat_request_and_returns_content():
    fake = FakeOllama(reply=lambda user: f"echo:{user}")
    out = fake.client().generate("Hello <PERSON_1>", system="Be brief.")
    assert out == "echo:Hello <PERSON_1>"
    assert fake.paths == ["/api/tags", "/api/chat"]  # reachability checked first
    payload = fake.chat_payloads[0]
    assert payload == {
        "model": "qwen2.5:7b",
        "stream": False,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello <PERSON_1>"},
        ],
    }


def test_generate_without_system_prompt():
    fake = FakeOllama()
    fake.client().generate("Hi")
    assert [m["role"] for m in fake.chat_payloads[0]["messages"]] == ["user"]


def test_untagged_model_matches_latest():
    fake = FakeOllama(models=("nomic-embed-text:latest",))
    assert fake.client(model_name="nomic-embed-text").check_health() is LLMHealth.AVAILABLE


def test_missing_model_raises_without_chat_or_pull():
    fake = FakeOllama(models=("llama3.1:8b",))
    with pytest.raises(LLMModelNotFoundError):
        fake.client().generate("Hi")
    assert fake.paths == ["/api/tags"]


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout])
def test_unreachable_server_raises_unavailable(error):
    fake = FakeOllama(tags_error=error)
    with pytest.raises(LLMUnavailableError):
        fake.client().generate("Hi")
    assert fake.chat_payloads == []


def test_generation_timeout_raises_timeout():
    with pytest.raises(LLMTimeoutError):
        FakeOllama(chat_error=httpx.ReadTimeout).client().generate("Hi")


def test_connect_timeout_during_chat_raises_unavailable():
    with pytest.raises(LLMUnavailableError):
        FakeOllama(chat_error=httpx.ConnectTimeout).client().generate("Hi")


def test_http_404_on_chat_is_model_not_found():
    with pytest.raises(LLMModelNotFoundError):
        FakeOllama(chat_status=404, chat_body={"error": "model not found"}).client().generate("Hi")


def test_http_error_does_not_include_response_body():
    fake = FakeOllama(chat_status=500, chat_body={"error": "failed on <PERSON_1>"})
    with pytest.raises(LLMResponseError) as exc_info:
        fake.client().generate("Hi")
    assert "PERSON" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("body", [{"message": {}}, {"message": {"content": 42}}, ["unexpected"]])
def test_malformed_response_raises(body):
    with pytest.raises(LLMResponseError):
        FakeOllama(chat_body=body).client().generate("Hi")


@pytest.mark.parametrize("prompt", ["", "   "])
def test_empty_prompt_rejected_before_network(prompt):
    fake = FakeOllama()
    with pytest.raises(LLMResponseError):
        fake.client().generate(prompt)
    assert fake.paths == []


@pytest.mark.parametrize(
    ("fake", "expected"),
    [
        (FakeOllama(), LLMHealth.AVAILABLE),
        (FakeOllama(models=("qwen2:7b",)), LLMHealth.MODEL_MISSING),
        (FakeOllama(tags_error=httpx.ConnectError), LLMHealth.UNAVAILABLE),
    ],
)
def test_check_health(fake, expected):
    assert fake.client().check_health() is expected


def test_does_not_log_prompt_or_response(caplog):
    fake = FakeOllama(reply=lambda user: "Reply about <PERSON_1> and secret-response-text")
    with caplog.at_level(logging.DEBUG):
        fake.client().generate("secret-prompt-text <PERSON_1>")
    assert "secret-prompt-text" not in caplog.text
    assert "secret-response-text" not in caplog.text
    assert "<PERSON_1>" not in caplog.text


def test_from_settings_applies_configuration():
    settings = Settings(
        ollama_base_url="http://127.0.0.1:11500", ollama_model="llama3.1:8b",
        ollama_connect_timeout=2.0, ollama_request_timeout=30.0,
    )
    client = OllamaClient.from_settings(settings)
    assert client.model_name == "llama3.1:8b"
    assert str(client._client.base_url) == "http://127.0.0.1:11500"
    assert client._client.timeout.connect == 2.0
    assert client._client.timeout.read == 30.0
    assert client._client.trust_env is False  # proxy env vars ignored
    client.close()


def test_real_unreachable_port_fails_safely():
    """No mock: a closed local port must surface as LLMUnavailableError."""
    client = OllamaClient(base_url="http://127.0.0.1:9", connect_timeout=0.5, request_timeout=1.0)
    assert client.check_health() is LLMHealth.UNAVAILABLE
    with pytest.raises(LLMUnavailableError):
        client.generate("Hi")
    client.close()
