"""Phase 1 acceptance: no raw detected PII reaches the (mocked) LLM boundary.

Flow under test, exactly as later phases will run it:
    raw input -> gateway.protect() -> llm.generate(sanitized) -> gateway.restore()
"""

from __future__ import annotations

import logging

import pytest

from app.core.exceptions import PIIDetectionError, PIILeakError
from app.llm.base import LLMClient
from app.security.gateway import PIISecurityGateway
from app.security.pii_detector import PIIDetector
from app.security.reidentifier import ReidentificationDecision, ReidentificationPolicy
from app.security.tokenizer import TokenVault

# (input, raw values that must never cross the boundary)
SYNTHETIC_CASES = [
    (
        "My name is John Smith. My email is john.smith@example.com.\nMy SSN is 123-45-6789.",
        ["John Smith", "john.smith@example.com", "123-45-6789"],
    ),
    (
        "Please call Maria Garcia at (415) 555-0132 or email maria.garcia@example.org.",
        ["Maria Garcia", "(415) 555-0132", "maria.garcia@example.org"],
    ),
    (
        "Employee David Chen, SSN 536-22-1846, phone +1 212-555-0187, "
        "asked whether david chen can update his record.",
        ["David Chen", "david chen", "536-22-1846", "+1 212-555-0187"],
    ),
    (
        "Escalate to Priya Patel (priya.patel@example.com, 312-555-0199). "
        "Backup contact: Liam O'Connor, liam.oconnor@example.org.",
        ["Priya Patel", "priya.patel@example.com", "312-555-0199",
         "liam.oconnor@example.org"],
    ),
]


def _digits(value: str) -> str:
    return "".join(c for c in value if c.isdigit())


def _assert_not_leaked(payloads: list[str], raw_values: list[str]) -> None:
    for payload in payloads:
        folded = payload.casefold()
        payload_digits = _digits(payload)
        for raw in raw_values:
            assert raw.casefold() not in folded, "raw PII value reached the LLM boundary"
            if sum(c.isdigit() for c in raw) >= 7:
                # Catch reformatted numbers too (e.g. separators removed).
                assert _digits(raw) not in payload_digits, "raw PII digits reached the LLM"


def test_mock_llm_satisfies_client_protocol(mock_llm):
    assert isinstance(mock_llm, LLMClient)


def test_specification_example_sanitized_exactly(gateway, mock_llm):
    text = "My name is John Smith. My email is john.smith@example.com.\nMy SSN is 123-45-6789."
    protected = gateway.protect(text)
    mock_llm.generate(protected.sanitized_text)
    assert mock_llm.received == [
        "My name is <PERSON_1>. My email is <EMAIL_1>.\nMy SSN is <SSN_1>."
    ]


@pytest.mark.parametrize(("text", "raw_values"), SYNTHETIC_CASES)
def test_no_raw_pii_reaches_llm_boundary(gateway, mock_llm, text, raw_values):
    protected = gateway.protect(text)
    mock_llm.generate(protected.sanitized_text, system="You are a helpful assistant.")

    assert mock_llm.received, "LLM was not called"
    _assert_not_leaked(mock_llm.received, raw_values)
    assert protected.total_entities > 0


@pytest.mark.parametrize(("text", "raw_values"), SYNTHETIC_CASES)
def test_round_trip_restores_pii_only_after_llm(gateway, mock_llm, text, raw_values):
    protected = gateway.protect(text)
    llm_output = mock_llm.generate(protected.sanitized_text)
    _assert_not_leaked([llm_output], raw_values)  # model output is token-only too

    restored = gateway.restore(llm_output, protected, ReidentificationPolicy.allow_all())
    assert restored.decision is ReidentificationDecision.ALLOWED
    for raw in raw_values:
        # Repeated mentions restore to the first surface form, hence casefold.
        assert raw.casefold() in restored.text.casefold()


def test_llm_never_receives_token_mapping(gateway, mock_llm):
    protected = gateway.protect("My name is John Smith and my SSN is 123-45-6789.")
    mock_llm.generate(protected.sanitized_text)
    payload = "".join(mock_llm.received)
    for token in protected.vault:
        value = protected.vault.reveal(token)
        assert f"{token}={value}" not in payload
        assert f"{token}: {value}" not in payload
        assert value not in payload


def test_policy_controls_reidentification(gateway, mock_llm):
    protected = gateway.protect("Contact John Smith, SSN 123-45-6789.")
    llm_output = mock_llm.generate(protected.sanitized_text)
    restored = gateway.restore(llm_output, protected, ReidentificationPolicy.allow("PERSON"))
    assert "John Smith" in restored.text
    assert "123-45-6789" not in restored.text
    assert restored.decision is ReidentificationDecision.PARTIAL


def test_protected_prompt_is_safe_to_log(gateway):
    protected = gateway.protect("My name is John Smith. SSN 123-45-6789.")
    assert "John Smith" not in repr(protected)
    assert "123-45-6789" not in repr(protected)
    assert protected.entity_counts == {"PERSON": 1, "US_SSN": 1}


def test_no_raw_pii_in_logs_even_at_debug(gateway, caplog):
    raw_values = ["John Smith", "john.smith@example.com", "123-45-6789", "415-555-0132"]
    text = "I am John Smith, john.smith@example.com, 123-45-6789, 415-555-0132."
    with caplog.at_level(logging.DEBUG):
        protected = gateway.protect(text)
        gateway.restore(protected.sanitized_text, protected, ReidentificationPolicy.allow_all())
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert caplog.records, "expected audit-safe log records"
    for raw in raw_values:
        assert raw not in logged
    assert "<PERSON_1>" not in logged  # token mappings/sanitized prompts are not logged either


def test_detection_failure_fails_closed(mock_llm):
    class BrokenAnalyzer:
        def analyze(self, **kwargs):
            raise RuntimeError("model crashed")

    gateway = PIISecurityGateway(PIIDetector(analyzer=BrokenAnalyzer()))  # type: ignore[arg-type]
    with pytest.raises(PIIDetectionError):
        protected = gateway.protect("My SSN is 123-45-6789.")
        mock_llm.generate(protected.sanitized_text)
    assert mock_llm.received == []


def test_leak_verification_fails_closed():
    vault = TokenVault()
    vault.get_or_create("US_SSN", "123-45-6789")
    with pytest.raises(PIILeakError) as exc_info:
        PIISecurityGateway.verify_no_leak("SSN 123-45-6789", vault)
    assert "123-45-6789" not in str(exc_info.value)


def test_clean_prompt_passes_through_unchanged(gateway, mock_llm):
    text = "Summarize our travel reimbursement policy."
    protected = gateway.protect(text)
    mock_llm.generate(protected.sanitized_text)
    assert mock_llm.received == [text]
    assert protected.total_entities == 0


def test_leak_check_ignores_values_matching_token_labels():
    """A value like 'Email' (spaCy false positive) must not match inside '<EMAIL_1>'."""
    vault = TokenVault()
    vault.get_or_create("PERSON", "Email")
    vault.get_or_create("EMAIL_ADDRESS", "david.chen@example.com")
    PIISecurityGateway.verify_no_leak("<PERSON_1> <EMAIL_1> today.", vault)
    with pytest.raises(PIILeakError):
        PIISecurityGateway.verify_no_leak("<PERSON_1> Email <EMAIL_1>", vault)
