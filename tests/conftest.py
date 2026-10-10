"""Shared fixtures. All PII in this test suite is synthetic.

Phone numbers use the reserved fictional 555-01xx range; email domains use
example.com/example.org (RFC 2606); SSNs are made-up or published samples.
"""

from __future__ import annotations

import pytest

from app.security.gateway import PIISecurityGateway
from app.security.pii_detector import PIIDetector


class RecordingMockLLM:
    """Test double for the LLM boundary that records every payload it receives.

    It replies by echoing any tokens it saw, simulating a model that refers
    back to the (pseudonymized) entities in its answer.
    """

    model_name = "mock-llm"

    def __init__(self) -> None:
        self.received: list[str] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.received.append(prompt)
        if system is not None:
            self.received.append(system)
        return f"Acknowledged. I have recorded the details you provided: {prompt}"


@pytest.fixture(scope="session")
def detector() -> PIIDetector:
    # Loading en_core_web_lg takes a few seconds; share one instance.
    return PIIDetector()


@pytest.fixture(scope="session")
def gateway(detector: PIIDetector) -> PIISecurityGateway:
    return PIISecurityGateway(detector)


@pytest.fixture
def mock_llm() -> RecordingMockLLM:
    return RecordingMockLLM()


# --------------------------------------------------------------------------
# Phase 2 fixtures: intercept the Ollama HTTP boundary and build API clients.
# --------------------------------------------------------------------------

import json  # noqa: E402
from collections.abc import Callable, Iterator  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.audit.logger import AuditLogger, InMemoryAuditSink  # noqa: E402
from app.core.config import DEFAULT_REIDENTIFICATION_CEILING, Settings  # noqa: E402
from app.llm.ollama_client import OllamaClient  # noqa: E402
from app.main import create_app  # noqa: E402
from app.security.secure_llm import SecureLLMService  # noqa: E402


class FakeOllama:
    """In-process stand-in for the Ollama HTTP API, recording every request.

    Used through `httpx.MockTransport`, so the real `OllamaClient` code runs
    and the assertions are made on the exact bytes that would go on the wire.
    """

    def __init__(
        self,
        models: tuple[str, ...] = ("qwen2.5:7b", "nomic-embed-text:latest"),
        reply: Callable[[str], str] | None = None,
        chat_status: int = 200,
        chat_body: object | None = None,
        tags_error: type[Exception] | None = None,
        chat_error: type[Exception] | None = None,
    ) -> None:
        self.models = models
        self.reply = reply or (lambda user: f"Summary of the request: {user}")
        self.chat_status = chat_status
        self.chat_body = chat_body
        self.tags_error = tags_error
        self.chat_error = chat_error
        self.paths: list[str] = []
        self.chat_payloads: list[dict] = []
        self.raw_bodies: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        self.raw_bodies.append(request.content.decode("utf-8", errors="replace"))
        if request.url.path == "/api/tags":
            if self.tags_error:
                raise self.tags_error("simulated", request=request)
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        if request.url.path == "/api/chat":
            payload = json.loads(request.content)
            self.chat_payloads.append(payload)
            if self.chat_error:
                raise self.chat_error("simulated", request=request)
            if self.chat_body is not None:
                return httpx.Response(self.chat_status, json=self.chat_body)
            user = next(m["content"] for m in payload["messages"] if m["role"] == "user")
            return httpx.Response(
                self.chat_status,
                json={"model": payload["model"], "done": True,
                      "message": {"role": "assistant", "content": self.reply(user)}},
            )
        return httpx.Response(404, json={"error": "not found"})

    def client(self, model_name: str = "qwen2.5:7b") -> OllamaClient:
        return OllamaClient(model_name=model_name, transport=httpx.MockTransport(self.handler))

    @property
    def wire(self) -> str:
        """Everything sent to Ollama, both raw and JSON-decoded."""
        decoded = [m["content"] for p in self.chat_payloads for m in p["messages"]]
        return "\n".join(self.raw_bodies + decoded)


@dataclass
class Harness:
    client: TestClient
    audit: InMemoryAuditSink
    fake: FakeOllama | None
    service: SecureLLMService


@pytest.fixture
def make_harness(gateway: PIISecurityGateway) -> Iterator[Callable[..., Harness]]:
    """Build a TestClient around a real SecureLLMService with injected parts."""
    opened: list[TestClient] = []

    def _make(
        fake: FakeOllama | None = None,
        llm: object | None = None,
        gateway_override: PIISecurityGateway | None = None,
        ceiling: tuple[str, ...] = DEFAULT_REIDENTIFICATION_CEILING,
        sinks: list | None = None,
    ) -> Harness:
        if llm is None:
            fake = fake or FakeOllama()
            llm = fake.client()
        audit = InMemoryAuditSink()
        service = SecureLLMService(
            gateway=gateway_override or gateway,
            llm=llm,  # type: ignore[arg-type]
            audit=AuditLogger([audit, *(sinks or [])]),
            reidentification_ceiling=ceiling,
        )
        client = TestClient(create_app(settings=Settings(), service=service),
                            raise_server_exceptions=False)
        client.__enter__()
        opened.append(client)
        return Harness(client, audit, fake, service)

    yield _make
    for client in opened:
        client.__exit__(None, None, None)
