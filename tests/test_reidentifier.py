"""Controlled re-identification tests (pure unit tests, no NLP model)."""

from __future__ import annotations

import pytest

from app.core.exceptions import ReidentificationError
from app.security.reidentifier import (
    ReidentificationDecision,
    ReidentificationPolicy,
    Reidentifier,
)
from app.security.tokenizer import TokenVault


@pytest.fixture
def vault() -> TokenVault:
    v = TokenVault()
    v.get_or_create("PERSON", "John Smith")
    v.get_or_create("EMAIL_ADDRESS", "john.smith@example.com")
    v.get_or_create("US_SSN", "123-45-6789")
    return v


LLM_OUTPUT = "Hello <PERSON_1>, we will email <EMAIL_1> about SSN <SSN_1>."


def test_allow_all_restores_every_token(vault):
    result = Reidentifier().reidentify(LLM_OUTPUT, vault, ReidentificationPolicy.allow_all())
    assert result.text == (
        "Hello John Smith, we will email john.smith@example.com about SSN 123-45-6789."
    )
    assert result.decision is ReidentificationDecision.ALLOWED
    assert (result.restored_count, result.withheld_count) == (3, 0)


def test_policy_withholds_unauthorized_entity_types(vault):
    policy = ReidentificationPolicy.allow("PERSON", "EMAIL_ADDRESS")
    result = Reidentifier().reidentify(LLM_OUTPUT, vault, policy)
    assert "123-45-6789" not in result.text
    assert "<SSN_1>" in result.text
    assert "John Smith" in result.text
    assert result.decision is ReidentificationDecision.PARTIAL
    assert (result.restored_count, result.withheld_count) == (2, 1)


def test_deny_all_keeps_tokens(vault):
    result = Reidentifier().reidentify(LLM_OUTPUT, vault, ReidentificationPolicy.deny_all())
    assert result.text == LLM_OUTPUT
    assert result.decision is ReidentificationDecision.DENIED


def test_unknown_tokens_from_llm_are_left_untouched(vault):
    result = Reidentifier().reidentify(
        "Also contact <PERSON_7>.", vault, ReidentificationPolicy.allow_all()
    )
    assert result.text == "Also contact <PERSON_7>."
    assert result.unknown_token_count == 1
    assert result.decision is ReidentificationDecision.NOT_REQUIRED


def test_restored_values_are_not_rescanned():
    """A value that itself looks like a token must not trigger a second substitution."""
    v = TokenVault()
    v.get_or_create("PERSON", "<PERSON_2>")
    v.get_or_create("PERSON", "Maria Garcia")
    result = Reidentifier().reidentify("<PERSON_1>", v, ReidentificationPolicy.allow_all())
    assert result.text == "<PERSON_2>"


def test_result_repr_hides_restored_text(vault):
    result = Reidentifier().reidentify(LLM_OUTPUT, vault, ReidentificationPolicy.allow_all())
    assert "123-45-6789" not in repr(result)


def test_tokens_from_another_requests_vault_are_not_resolved(vault):
    other = TokenVault()
    result = Reidentifier().reidentify(LLM_OUTPUT, other, ReidentificationPolicy.allow_all())
    assert result.text == LLM_OUTPUT


@pytest.mark.parametrize(("text", "vault_arg"), [(None, TokenVault()), ("hi", {"<PERSON_1>": "x"})])
def test_invalid_arguments_raise(text, vault_arg):
    with pytest.raises(ReidentificationError):
        Reidentifier().reidentify(text, vault_arg, ReidentificationPolicy.allow_all())  # type: ignore[arg-type]
