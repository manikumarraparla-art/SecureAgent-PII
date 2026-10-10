"""Privacy-safe audit logging tests."""

from __future__ import annotations

import json
import logging
import stat

import httpx
import pytest

from app.audit.events import AuditEvent, AuditEventType, AuditStatus
from app.audit.logger import AuditLogger, JsonlAuditSink
from app.core.exceptions import ErrorCode
from app.security.reidentifier import ReidentificationDecision
from tests.conftest import FakeOllama

ALLOWED_KEYS = {
    "timestamp", "request_id", "event", "status", "pii_types", "pii_count",
    "provider", "model", "reidentification", "latency_ms", "error_category",
}

SYNTHETIC_MESSAGES = [
    "My name is David Chen.\nMy email is david.chen@example.com.\nMy SSN is 123-45-6789.",
    "Call Maria Garcia at (415) 555-0132 or maria.garcia@example.org.",
]
RAW_VALUES = [
    "David Chen", "david.chen@example.com", "123-45-6789", "123456789",
    "Maria Garcia", "(415) 555-0132", "4155550132", "maria.garcia@example.org",
]


# ------------------------------------------------------------ event schema


def test_event_serializes_only_schema_fields():
    event = AuditEvent(
        request_id="abc-123", event=AuditEventType.REQUEST_COMPLETED, status=AuditStatus.SUCCESS,
        model="qwen2.5:7b", pii_types=("PERSON", "EMAIL_ADDRESS", "PERSON"), pii_count=2,
        reidentification=ReidentificationDecision.PARTIAL, latency_ms=12.345,
    )
    record = event.to_dict()
    assert set(record) <= ALLOWED_KEYS
    assert record["pii_types"] == ["EMAIL_ADDRESS", "PERSON"]
    assert record["latency_ms"] == 12.3
    assert "error_category" not in record


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_id": "David Chen"},
        {"request_id": "x" * 65},
        {"model": "david.chen@example.com is the model"},
        {"pii_types": ("david.chen@example.com",)},
        {"pii_count": -1},
        {"error_category": "123-45-6789"},
        {"reidentification": "ALLOWED for David Chen"},
        {"event": "free text"},
    ],
)
def test_event_rejects_anything_outside_schema(kwargs):
    base = {"request_id": "abc", "event": AuditEventType.PII_SANITIZED, "status": AuditStatus.SUCCESS}
    with pytest.raises(ValueError):
        AuditEvent(**{**base, **kwargs})


def test_jsonl_sink_writes_owner_only_file(tmp_path):
    path = tmp_path / "audit" / "audit.jsonl"
    sink = JsonlAuditSink(path)
    for _ in range(2):
        sink.write(AuditEvent("abc", AuditEventType.PII_SANITIZED, AuditStatus.SUCCESS))
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["event"] == "PII_SANITIZED"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_sink_failure_does_not_raise_or_log_content(caplog):
    class BrokenSink:
        def write(self, event):
            raise OSError("disk full while writing David Chen")

    with caplog.at_level(logging.ERROR):
        AuditLogger([BrokenSink()]).record(
            AuditEvent("abc", AuditEventType.PII_SANITIZED, AuditStatus.SUCCESS)
        )
    assert "David Chen" not in caplog.text
    assert "OSError" in caplog.text


# --------------------------------------------- end-to-end audit privacy


def _audit_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _assert_audit_is_clean(path) -> list[dict]:
    raw_text = path.read_text()
    for raw in RAW_VALUES:
        assert raw not in raw_text, "raw PII found in audit log"
    for marker in ("<PERSON_", "<EMAIL_", "<SSN_", "<PHONE_", "Summary", "My name"):
        assert marker not in raw_text, "prompt/response/token content found in audit log"
    records = _audit_lines(path)
    for record in records:
        assert set(record) <= ALLOWED_KEYS
    return records


@pytest.mark.parametrize("message", SYNTHETIC_MESSAGES)
def test_successful_request_audit_contains_no_pii(make_harness, tmp_path, message):
    path = tmp_path / "audit.jsonl"
    h = make_harness(sinks=[JsonlAuditSink(path)])
    resp = h.client.post("/api/v1/chat", json={
        "message": message, "allowed_reidentification_types": ["PERSON", "EMAIL_ADDRESS"],
    })
    assert resp.status_code == 200

    records = _assert_audit_is_clean(path)
    assert [r["event"] for r in records] == [
        "PII_SANITIZED", "LLM_REQUEST_COMPLETED", "REQUEST_COMPLETED",
    ]
    final = records[-1]
    assert final["status"] == "SUCCESS"
    assert final["model"] == "qwen2.5:7b"
    assert final["pii_count"] >= 2
    assert final["reidentification"] in {"ALLOWED", "PARTIAL"}
    assert final["request_id"] == resp.json()["request_id"]
    assert final["latency_ms"] >= 0


@pytest.mark.parametrize(
    ("fake", "category"),
    [(FakeOllama(tags_error=httpx.ConnectError), "LLM_UNAVAILABLE"),
     (FakeOllama(chat_error=httpx.ReadTimeout), "LLM_TIMEOUT")],
)
def test_failed_request_audit_contains_no_pii(make_harness, tmp_path, fake, category):
    path = tmp_path / "audit.jsonl"
    h = make_harness(fake=fake, sinks=[JsonlAuditSink(path)])
    h.client.post("/api/v1/chat", json={"message": SYNTHETIC_MESSAGES[0]})
    records = _assert_audit_is_clean(path)
    assert [r["event"] for r in records] == ["PII_SANITIZED", "LLM_REQUEST_FAILED", "REQUEST_FAILED"]
    assert records[-1]["error_category"] == category
    assert records[-1]["pii_types"] == ["EMAIL_ADDRESS", "PERSON", "US_SSN"]


def test_application_logs_contain_no_pii(make_harness, caplog):
    h = make_harness()
    with caplog.at_level(logging.DEBUG):
        for message in SYNTHETIC_MESSAGES:
            h.client.post("/api/v1/chat", json={
                "message": message, "allowed_reidentification_types": ["PERSON"],
            })
    logged = "\n".join(r.getMessage() for r in caplog.records)
    for raw in RAW_VALUES:
        assert raw not in logged
    assert "<PERSON_1>" not in logged


def test_audit_records_error_category_enum_only(make_harness):
    h = make_harness(fake=FakeOllama(chat_error=httpx.ReadTimeout))
    h.client.post("/api/v1/chat", json={"message": "hello"})
    failed = [e for e in h.audit.events if e.status is AuditStatus.FAILURE]
    assert {e.error_category for e in failed} == {ErrorCode.LLM_TIMEOUT}
