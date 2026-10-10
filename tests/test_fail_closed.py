"""Fail-closed behaviour: security failures never reach the LLM, and no error
response or audit record exposes PII, tokens or internal state."""

from __future__ import annotations

import httpx
import pytest

from app.audit.events import AuditEventType
from app.core.exceptions import ErrorCode
from app.security.gateway import PIISecurityGateway, ProtectedPrompt
from app.security.pii_detector import PIIDetector
from app.security.reidentifier import Reidentifier
from app.security.tokenizer import PIITokenizer, TokenizationResult, TokenVault
from tests.conftest import FakeOllama

MESSAGE = "My name is David Chen, email david.chen@example.com, SSN 123-45-6789."
RAW = ["David Chen", "david.chen@example.com", "123-45-6789"]


def assert_safe_error(resp, status: int, code: str) -> None:
    assert resp.status_code == status
    body = resp.json()
    assert body["error"]["code"] == code
    assert set(body) == {"request_id", "error"}
    text = resp.text
    for raw in RAW:
        assert raw not in text
    for marker in ("<PERSON_", "<EMAIL_", "<SSN_", "Traceback", "TokenVault", "simulated"):
        assert marker not in text


def assert_failure_audited(h, code: ErrorCode) -> None:
    failed = [e for e in h.audit.events if e.event is AuditEventType.REQUEST_FAILED]
    assert len(failed) == 1 and failed[0].error_category is code


# ---------------------------------------------------------- before the LLM


def test_presidio_failure_does_not_call_ollama(make_harness):
    class BrokenAnalyzer:
        def analyze(self, **kwargs):
            raise RuntimeError(f"crash while reading {kwargs['text']}")

    gateway = PIISecurityGateway(PIIDetector(analyzer=BrokenAnalyzer()))  # type: ignore[arg-type]
    h = make_harness(gateway_override=gateway)
    resp = h.client.post("/api/v1/chat", json={"message": MESSAGE})
    assert_safe_error(resp, 500, "PII_DETECTION_FAILED")
    assert h.fake.chat_payloads == [] and h.fake.paths == []
    assert_failure_audited(h, ErrorCode.PII_DETECTION_FAILED)


def test_tokenization_failure_does_not_call_ollama(make_harness, detector):
    class BrokenTokenizer(PIITokenizer):
        def tokenize(self, text, entities, vault=None):
            raise RuntimeError(f"cannot tokenize {text}")

    h = make_harness(gateway_override=PIISecurityGateway(detector, tokenizer=BrokenTokenizer()))
    resp = h.client.post("/api/v1/chat", json={"message": MESSAGE})
    assert_safe_error(resp, 500, "TOKENIZATION_FAILED")
    assert h.fake.paths == []
    assert_failure_audited(h, ErrorCode.TOKENIZATION_FAILED)


def test_security_validation_failure_does_not_call_ollama(make_harness, detector):
    class LeakyTokenizer(PIITokenizer):
        """Records PII in the vault but forgets to replace it in the text."""

        def tokenize(self, text, entities, vault=None):
            vault = vault or TokenVault()
            for e in entities:
                vault.get_or_create(e.entity_type, text[e.start : e.end])
            return TokenizationResult(text, vault, 0)

    h = make_harness(gateway_override=PIISecurityGateway(detector, tokenizer=LeakyTokenizer()))
    resp = h.client.post("/api/v1/chat", json={"message": MESSAGE})
    assert_safe_error(resp, 500, "SECURITY_VALIDATION_FAILED")
    assert h.fake.paths == []
    assert_failure_audited(h, ErrorCode.SECURITY_VALIDATION_FAILED)


# ------------------------------------------------------------- LLM failures


@pytest.mark.parametrize(
    ("fake", "status", "code"),
    [
        (FakeOllama(tags_error=httpx.ConnectError), 503, "LLM_UNAVAILABLE"),
        (FakeOllama(models=("llama3.1:8b",)), 503, "LLM_MODEL_UNAVAILABLE"),
        (FakeOllama(chat_error=httpx.ReadTimeout), 504, "LLM_TIMEOUT"),
        (FakeOllama(chat_status=500, chat_body={"error": "boom <PERSON_1>"}), 502, "LLM_BAD_RESPONSE"),
        (FakeOllama(chat_body={"unexpected": True}), 502, "LLM_BAD_RESPONSE"),
    ],
    ids=["ollama-unavailable", "model-missing", "timeout", "http-500", "malformed"],
)
def test_llm_failures_return_safe_service_errors(make_harness, fake, status, code):
    h = make_harness(fake=fake)
    resp = h.client.post("/api/v1/chat", json={"message": MESSAGE})
    assert_safe_error(resp, status, code)
    assert_failure_audited(h, ErrorCode(code))
    # Even on failure paths, Ollama never saw raw PII.
    for raw in RAW:
        assert raw not in fake.wire


def test_model_missing_never_falls_back_or_pulls(make_harness):
    fake = FakeOllama(models=("llama3.1:8b", "qwen2:7b"))
    h = make_harness(fake=fake)
    h.client.post("/api/v1/chat", json={"message": "hello"})
    assert fake.paths == ["/api/tags"]  # no /api/chat with another model, no /api/pull


def test_unexpected_llm_exception_is_contained(make_harness):
    class ExplodingLLM:
        model_name = "qwen2.5:7b"

        def generate(self, prompt, *, system=None):
            raise RuntimeError(f"internal failure echoing {prompt}")

    h = make_harness(llm=ExplodingLLM())
    resp = h.client.post("/api/v1/chat", json={"message": MESSAGE})
    assert_safe_error(resp, 502, "LLM_BAD_RESPONSE")


# ------------------------------------------------------ re-identification


def test_reidentification_failure_returns_no_mapping_or_state(make_harness, detector):
    class BrokenReidentifier(Reidentifier):
        def reidentify(self, text, vault, policy):
            raise RuntimeError(f"failed restoring {vault.items()} in {text}")

    gateway = PIISecurityGateway(detector, reidentifier=BrokenReidentifier())
    h = make_harness(gateway_override=gateway)
    resp = h.client.post(
        "/api/v1/chat", json={"message": MESSAGE, "allowed_reidentification_types": ["PERSON"]}
    )
    assert_safe_error(resp, 500, "REIDENTIFICATION_FAILED")
    assert "response" not in resp.json()
    assert_failure_audited(h, ErrorCode.REIDENTIFICATION_FAILED)


# ------------------------------------------------------- mapping lifecycle


class RecordingGateway(PIISecurityGateway):
    def __init__(self, detector, **kwargs):
        super().__init__(detector, **kwargs)
        self.protected: list[ProtectedPrompt] = []

    def protect(self, text, request_id=None, vault=None):
        result = super().protect(text, request_id, vault)
        self.protected.append(result)
        return result


@pytest.mark.parametrize("fake", [FakeOllama(), FakeOllama(chat_error=httpx.ReadTimeout)],
                         ids=["success", "llm-timeout"])
def test_vault_is_destroyed_after_request(make_harness, detector, fake):
    gateway = RecordingGateway(detector)
    h = make_harness(fake=fake, gateway_override=gateway)
    h.client.post("/api/v1/chat", json={"message": MESSAGE})
    assert len(gateway.protected) == 1
    vault = gateway.protected[0].vault
    assert len(vault) == 0
    assert vault.reveal("<PERSON_1>") is None
