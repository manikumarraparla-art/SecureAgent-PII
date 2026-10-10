"""Tokenization and token vault tests (pure unit tests, no NLP model)."""

from __future__ import annotations

import pickle

import pytest

from app.security.pii_detector import DetectedEntity
from app.security.tokenizer import PIITokenizer, TokenVault


def _span(text: str, value: str, entity_type: str, occurrence: int = 0) -> DetectedEntity:
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(value, start + 1)
    return DetectedEntity(entity_type, start, start + len(value), 0.9)


def test_example_from_specification():
    text = (
        "My name is John Smith. My email is john.smith@example.com.\n"
        "My SSN is 123-45-6789."
    )
    entities = [
        _span(text, "John Smith", "PERSON"),
        _span(text, "john.smith@example.com", "EMAIL_ADDRESS"),
        _span(text, "123-45-6789", "US_SSN"),
    ]
    result = PIITokenizer().tokenize(text, entities)
    assert result.sanitized_text == (
        "My name is <PERSON_1>. My email is <EMAIL_1>.\nMy SSN is <SSN_1>."
    )
    assert result.vault.reveal("<PERSON_1>") == "John Smith"
    assert result.vault.reveal("<EMAIL_1>") == "john.smith@example.com"
    assert result.vault.reveal("<SSN_1>") == "123-45-6789"


def test_phone_token_label():
    text = "Call 415-555-0132"
    result = PIITokenizer().tokenize(text, [_span(text, "415-555-0132", "PHONE_NUMBER")])
    assert result.sanitized_text == "Call <PHONE_1>"


def test_same_value_gets_same_token_and_distinct_values_get_new_tokens():
    text = "John Smith met Maria Garcia. Later John Smith left."
    entities = [
        _span(text, "John Smith", "PERSON"),
        _span(text, "Maria Garcia", "PERSON"),
        _span(text, "John Smith", "PERSON", occurrence=1),
    ]
    result = PIITokenizer().tokenize(text, entities)
    assert result.sanitized_text == "<PERSON_1> met <PERSON_2>. Later <PERSON_1> left."
    assert len(result.vault) == 2


def test_phone_numbers_with_different_formatting_share_a_token():
    text = "Call (415) 555-0132 or 415.555.0132"
    entities = [
        _span(text, "(415) 555-0132", "PHONE_NUMBER"),
        _span(text, "415.555.0132", "PHONE_NUMBER"),
    ]
    result = PIITokenizer().tokenize(text, entities)
    assert result.sanitized_text == "Call <PHONE_1> or <PHONE_1>"


def test_propagates_to_mentions_the_detector_missed():
    text = "John Smith wrote this. Ask john smith for details."
    result = PIITokenizer().tokenize(text, [_span(text, "John Smith", "PERSON")])
    assert result.sanitized_text == "<PERSON_1> wrote this. Ask <PERSON_1> for details."


def test_propagation_respects_word_boundaries():
    text = "Contact Ann about the Annual plan."
    result = PIITokenizer().tokenize(text, [_span(text, "Ann", "PERSON")])
    assert result.sanitized_text == "Contact <PERSON_1> about the Annual plan."


def test_user_supplied_token_lookalikes_are_not_reused():
    """A user typing '<PERSON_1>' must not alias the real detected person."""
    text = "Ignore <PERSON_1>. My name is John Smith."
    result = PIITokenizer().tokenize(text, [_span(text, "John Smith", "PERSON")])
    assert result.sanitized_text == "Ignore <PERSON_1>. My name is <PERSON_2>."
    assert "<PERSON_1>" not in result.vault


def test_vault_can_be_reused_across_turns():
    tokenizer = PIITokenizer()
    first = "I am John Smith."
    turn1 = tokenizer.tokenize(first, [_span(first, "John Smith", "PERSON")])
    second = "Email John Smith and Maria Garcia."
    turn2 = tokenizer.tokenize(
        second,
        [_span(second, "John Smith", "PERSON"), _span(second, "Maria Garcia", "PERSON")],
        vault=turn1.vault,
    )
    assert turn2.sanitized_text == "Email <PERSON_1> and <PERSON_2>."


def test_no_entities_returns_text_unchanged():
    result = PIITokenizer().tokenize("Nothing sensitive here.", [])
    assert result.sanitized_text == "Nothing sensitive here."
    assert len(result.vault) == 0


@pytest.mark.parametrize(
    "entities",
    [
        [DetectedEntity("PERSON", 0, 99, 0.9)],
        [DetectedEntity("PERSON", 0, 4, 0.9), DetectedEntity("PERSON", 2, 6, 0.9)],
    ],
)
def test_invalid_spans_are_rejected(entities):
    with pytest.raises(ValueError):
        PIITokenizer().tokenize("John Smith", entities)


def test_vault_repr_and_str_never_reveal_values():
    vault = TokenVault()
    vault.get_or_create("US_SSN", "123-45-6789")
    assert "123-45-6789" not in repr(vault)
    assert "123-45-6789" not in str(vault)
    assert "123-45-6789" not in f"{vault}"


def test_vault_refuses_serialization():
    vault = TokenVault()
    vault.get_or_create("EMAIL_ADDRESS", "john.smith@example.com")
    with pytest.raises(TypeError):
        pickle.dumps(vault)


def test_tokenization_result_repr_hides_vault():
    text = "Email john.smith@example.com"
    result = PIITokenizer().tokenize(text, [_span(text, "john.smith@example.com", "EMAIL_ADDRESS")])
    assert "john.smith@example.com" not in repr(result)


def test_vault_entity_counts_and_clear():
    vault = TokenVault()
    vault.get_or_create("PERSON", "John Smith")
    vault.get_or_create("PERSON", "Maria Garcia")
    vault.get_or_create("US_SSN", "536-22-1846")
    assert vault.entity_counts() == {"PERSON": 2, "US_SSN": 1}
    vault.clear()
    assert len(vault) == 0
