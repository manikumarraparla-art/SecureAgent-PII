"""FastAPI endpoint tests: health, chat contract, policy, validation."""

from __future__ import annotations

import httpx
import pytest

from tests.conftest import FakeOllama

SPEC_REQUEST = {
    "message": "My name is David Chen and my email is david.chen@example.com. Summarize my request.",
    "allowed_reidentification_types": ["PERSON"],
}


# ------------------------------------------------------------------ health


def test_health_reports_available(make_harness):
    resp = make_harness().client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"version": "1.0.0", "status": "healthy", "ollama": "available", "provider": "ollama", "llm": "available", "tool_calling": True, "model": "qwen2.5:7b"}


@pytest.mark.parametrize(
    ("fake", "ollama"),
    [(FakeOllama(tags_error=httpx.ConnectError), "unavailable"),
     (FakeOllama(models=("llama3.1:8b",)), "model_missing")],
)
def test_health_reports_degraded(make_harness, fake, ollama):
    resp = make_harness(fake=fake).client.get("/health")
    assert resp.status_code == 503
    assert resp.json() == {"version": "1.0.0", "status": "degraded", "ollama": ollama, "provider": "ollama", "llm": ollama, "tool_calling": True, "model": "qwen2.5:7b"}


def test_health_exposes_no_configuration(make_harness):
    body = make_harness().client.get("/health").text
    for secret_ish in ("localhost:11434", "spacy", "threshold", "audit", "OLLAMA_"):
        assert secret_ish not in body


# -------------------------------------------------------------------- chat


def test_chat_spec_example_partial_reidentification(make_harness):
    h = make_harness()
    resp = h.client.post("/api/v1/chat", json=SPEC_REQUEST)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"request_id", "response", "security", "model"}
    assert body["model"] == "qwen2.5:7b"
    assert body["security"] == {
        "pii_detected": True,
        "entity_types": ["EMAIL_ADDRESS", "PERSON"],
        "reidentification_decision": "PARTIAL",
    }
    assert "David Chen" in body["response"]  # PERSON allowed
    assert "david.chen@example.com" not in body["response"]  # EMAIL withheld
    assert "<EMAIL_1>" in body["response"]
    assert resp.headers["X-Request-ID"] == body["request_id"]


def test_default_policy_denies_reidentification(make_harness):
    body = make_harness().client.post(
        "/api/v1/chat", json={"message": SPEC_REQUEST["message"]}
    ).json()
    assert body["security"]["reidentification_decision"] == "DENIED"
    assert "David Chen" not in body["response"]
    assert "<PERSON_1>" in body["response"]


def test_server_ceiling_caps_requested_types(make_harness):
    """US_SSN is outside the default ceiling even when the caller asks for it."""
    h = make_harness()
    body = h.client.post("/api/v1/chat", json={
        "message": "David Chen has SSN 123-45-6789.",
        "allowed_reidentification_types": ["PERSON", "US_SSN"],
    }).json()
    assert "David Chen" in body["response"]
    assert "123-45-6789" not in body["response"]
    assert body["security"]["reidentification_decision"] == "PARTIAL"


def test_ceiling_can_be_widened_by_server_config(make_harness):
    h = make_harness(ceiling=("PERSON", "US_SSN"))
    body = h.client.post("/api/v1/chat", json={
        "message": "David Chen has SSN 123-45-6789.",
        "allowed_reidentification_types": ["PERSON", "US_SSN"],
    }).json()
    assert "123-45-6789" in body["response"]
    assert body["security"]["reidentification_decision"] == "ALLOWED"


def test_message_without_pii(make_harness):
    body = make_harness().client.post(
        "/api/v1/chat", json={"message": "Summarize the travel policy."}
    ).json()
    assert body["security"] == {
        "pii_detected": False, "entity_types": [], "reidentification_decision": "NOT_REQUIRED",
    }


def test_client_supplied_request_id_is_ignored(make_harness):
    resp = make_harness().client.post(
        "/api/v1/chat", json={"message": "hello"}, headers={"X-Request-ID": "David Chen"}
    )
    assert resp.json()["request_id"] != "David Chen"
    assert resp.headers["X-Request-ID"] == resp.json()["request_id"]


def test_response_contains_no_internal_state(make_harness):
    text = make_harness().client.post("/api/v1/chat", json=SPEC_REQUEST).text
    for marker in ("TokenVault", "vault", "sanitized", "score", "mapping", "system"):
        assert marker not in text


# -------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "   "},
        {"message": "x" * 8001},
        {"message": "SSN 123-45-6789", "allowed_reidentification_types": ["CREDIT_CARD"]},
        {"message": "SSN 123-45-6789", "extra_field": "david.chen@example.com"},
        {"allowed_reidentification_types": ["PERSON"]},
    ],
    ids=["empty", "blank", "too-long", "bad-type", "extra-field", "missing-message"],
)
def test_validation_errors_do_not_echo_input(make_harness, payload):
    h = make_harness()
    resp = h.client.post("/api/v1/chat", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "INVALID_REQUEST"
    for leaked in ("123-45-6789", "david.chen@example.com", "xxxxxxxx", "CREDIT_CARD"):
        assert leaked not in resp.text
    assert h.fake.paths == []


def test_malformed_json_does_not_echo_input(make_harness):
    resp = make_harness().client.post(
        "/api/v1/chat", content=b'{"message": "SSN 123-45-6789"', headers={"content-type": "application/json"}
    )
    assert resp.status_code == 422
    assert "123-45-6789" not in resp.text
