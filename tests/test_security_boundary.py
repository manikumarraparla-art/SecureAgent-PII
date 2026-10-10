"""Phase 2 acceptance: raw detected PII never reaches the Ollama HTTP boundary.

Requests go through the real FastAPI app, SecureLLMService, gateway and
OllamaClient; only the network transport is replaced by FakeOllama, which
records the exact bytes that would be sent to Ollama.
"""

from __future__ import annotations

import re

import pytest

from tests.conftest import FakeOllama


def _digits(text: str) -> str:
    return "".join(c for c in text if c.isdigit())


def assert_not_on_wire(fake: FakeOllama, raw_values: list[str]) -> None:
    wire = fake.wire
    folded = wire.casefold()
    wire_digits = _digits(wire)
    assert fake.chat_payloads, "Ollama /api/chat was never called"
    for raw in raw_values:
        visible = "".join(c for c in raw if c not in "​‌‍⁠﻿")
        assert raw.casefold() not in folded, f"raw PII reached Ollama ({len(raw)} chars)"
        assert visible.casefold() not in folded, "raw PII (invisible chars removed) reached Ollama"
        if sum(c.isdigit() for c in raw) >= 7:
            assert _digits(raw) not in wire_digits, "raw PII digits reached Ollama"


def _user_content(fake: FakeOllama) -> str:
    return next(m["content"] for m in fake.chat_payloads[-1]["messages"] if m["role"] == "user")


SPEC_MESSAGE = "My name is David Chen.\nMy email is david.chen@example.com.\nMy SSN is 123-45-6789."


def test_spec_message_reaches_ollama_only_as_tokens(make_harness):
    h = make_harness()
    resp = h.client.post("/api/v1/chat", json={"message": SPEC_MESSAGE})
    assert resp.status_code == 200

    sent = _user_content(h.fake)
    # Phase 1 token format: <PERSON_n>, <EMAIL_n>, <SSN_n>.
    assert re.search(r"<PERSON_\d+>", sent)
    assert re.search(r"<EMAIL_\d+>", sent)
    assert re.search(r"<SSN_\d+>", sent)
    assert sent == "My name is <PERSON_1>.\nMy email is <EMAIL_1>.\nMy SSN is <SSN_1>."
    assert_not_on_wire(h.fake, ["David Chen", "david.chen@example.com", "123-45-6789"])


def test_only_expected_ollama_endpoints_are_called(make_harness):
    h = make_harness()
    h.client.post("/api/v1/chat", json={"message": SPEC_MESSAGE})
    assert set(h.fake.paths) == {"/api/tags", "/api/chat"}  # never /api/pull
    payload = h.fake.chat_payloads[0]
    assert payload["model"] == "qwen2.5:7b"
    assert payload["stream"] is False
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]


BYPASS_CASES = {
    "lowercase repeat of name": (
        "My name is David Chen. Please ask david chen to confirm.",
        ["David Chen", "david chen"],
    ),
    "upper-case email": ("Email DAVID.CHEN@EXAMPLE.COM today.", ["DAVID.CHEN@EXAMPLE.COM"]),
    "space-separated SSN": ("My SSN is 123 45 6789.", ["123 45 6789"]),
    "dot-separated SSN": ("My SSN is 123.45.6789.", ["123.45.6789"]),
    "full-width SSN": ("My SSN is １２３-４５-６７８９.", ["１２３-４５-６７８９", "123-45-6789"]),
    "zero-width space in SSN": ("My SSN is 123-45-​6789.", ["123-45-​6789"]),
    "zero-width joiner in email": (
        "Email david.chen@exam‍ple.com now.", ["david.chen@exam‍ple.com"],
    ),
    "unformatted SSN with context": ("My SSN is 536221846.", ["536221846"]),
    "phone without separators": ("Call me on 4155550132.", ["4155550132"]),
    "phone with country code": ("Call me on +1 (415) 555-0132.", ["(415) 555-0132"]),
}


@pytest.mark.parametrize("case", list(BYPASS_CASES), ids=list(BYPASS_CASES))
def test_normalized_and_bypass_forms_do_not_reach_ollama(make_harness, case):
    message, raw_values = BYPASS_CASES[case]
    h = make_harness()
    resp = h.client.post("/api/v1/chat", json={"message": message})
    assert resp.status_code == 200
    assert_not_on_wire(h.fake, raw_values)


def test_boundary_assertion_would_catch_a_leak():
    """Guard against a vacuous test: the checker must fail on raw PII."""
    fake = FakeOllama()
    fake.chat_payloads.append({"messages": [{"role": "user", "content": "SSN 123-45-6789"}]})
    with pytest.raises(AssertionError):
        assert_not_on_wire(fake, ["123-45-6789"])
