"""Controlled re-identification of LLM output.

Runs only in the application security layer, after the LLM has responded.
A policy decides which entity types the caller may see in clear text; tokens
for other types stay masked.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from app.core.config import SUPPORTED_PII_ENTITIES
from app.core.exceptions import ReidentificationError
from app.security.tokenizer import TOKEN_PATTERN, TokenVault

logger = logging.getLogger(__name__)


class ReidentificationDecision(str, Enum):
    """Outcome recorded in the audit log."""

    NOT_REQUIRED = "NOT_REQUIRED"  # response contained no known tokens
    ALLOWED = "ALLOWED"  # every known token was restored
    PARTIAL = "PARTIAL"  # some entity types were withheld by policy
    DENIED = "DENIED"  # no known token was restored


@dataclass(frozen=True)
class ReidentificationPolicy:
    """Entity types the caller is authorized to see in clear text."""

    allowed_entity_types: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def deny_all(cls) -> ReidentificationPolicy:
        return cls(frozenset())

    @classmethod
    def allow_all(cls) -> ReidentificationPolicy:
        return cls(frozenset(SUPPORTED_PII_ENTITIES))

    @classmethod
    def allow(cls, *entity_types: str) -> ReidentificationPolicy:
        return cls(frozenset(entity_types))

    def permits(self, entity_type: str) -> bool:
        return entity_type in self.allowed_entity_types


@dataclass(frozen=True)
class ReidentificationResult:
    """Re-identified text plus audit-safe counters."""

    text: str = field(repr=False)  # may contain PII after restoration
    decision: ReidentificationDecision
    restored_count: int
    withheld_count: int
    unknown_token_count: int


class Reidentifier:
    """Restores PII values for tokens that the policy allows."""

    def reidentify(
        self,
        text: str,
        vault: TokenVault,
        policy: ReidentificationPolicy,
    ) -> ReidentificationResult:
        """Replace authorized tokens in `text` with their original values.

        - Tokens the policy does not allow are left as tokens.
        - Token-shaped strings not in the vault (e.g. hallucinated by the LLM)
          are left untouched and counted.
        - Substitution is single-pass, so restored values are never re-scanned.
        """
        if not isinstance(text, str):
            raise ReidentificationError("LLM response must be a string")
        if not isinstance(vault, TokenVault):
            raise ReidentificationError("A TokenVault is required for re-identification")

        restored = withheld = unknown = 0

        def _substitute(match: re.Match[str]) -> str:
            nonlocal restored, withheld, unknown
            token = match.group(0)
            entity_type = vault.entity_type(token)
            if entity_type is None:
                unknown += 1
                return token
            if not policy.permits(entity_type):
                withheld += 1
                return token
            restored += 1
            return vault.reveal(token) or token

        output = TOKEN_PATTERN.sub(_substitute, text)

        if restored == 0 and withheld == 0:
            decision = ReidentificationDecision.NOT_REQUIRED
        elif withheld == 0:
            decision = ReidentificationDecision.ALLOWED
        elif restored == 0:
            decision = ReidentificationDecision.DENIED
        else:
            decision = ReidentificationDecision.PARTIAL

        logger.debug(
            "Re-identification %s: restored=%d withheld=%d unknown=%d",
            decision.value, restored, withheld, unknown,
        )
        return ReidentificationResult(output, decision, restored, withheld, unknown)
