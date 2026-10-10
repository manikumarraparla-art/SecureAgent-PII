"""PII Security Gateway: the single entry/exit point between users and the LLM.

    raw input -> normalize -> detect -> tokenize -> verify no leak -> sanitized prompt
    LLM output -> policy-controlled re-identification -> response

Everything downstream of `protect()` (RAG, agents, LLM) only ever receives
`ProtectedPrompt.sanitized_text`. The vault stays in this layer.
"""

from __future__ import annotations

import logging
import unicodedata
import uuid
from dataclasses import dataclass, field

from app.core.exceptions import PIILeakError, SecureAgentError, TokenizationError
from app.security.pii_detector import PIIDetector
from app.security.reidentifier import (
    ReidentificationPolicy,
    ReidentificationResult,
    Reidentifier,
)
from app.security.tokenizer import TOKEN_PATTERN, PIITokenizer, TokenVault

logger = logging.getLogger(__name__)


def normalize_input(text: str) -> str:
    """Canonicalize text so invisible or look-alike characters cannot hide PII.

    - NFKC folds compatibility forms (e.g. full-width digits) to plain ASCII.
    - Unicode "format" characters (category Cf: zero-width spaces/joiners,
      bidi controls, soft hyphens) are removed. Without this,
      '123-45-<U+200B>6789' evades detection yet still reads as an SSN.
    """
    if not isinstance(text, str):
        return text
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")


@dataclass(frozen=True)
class ProtectedPrompt:
    """Output of `PIISecurityGateway.protect`.

    Only `sanitized_text` may cross the LLM boundary. `entity_counts` and
    `request_id` are audit-safe; `vault` is not and is hidden from repr.
    """

    request_id: str
    sanitized_text: str
    vault: TokenVault = field(repr=False)
    entity_counts: dict[str, int]

    @property
    def entity_types(self) -> list[str]:
        return sorted(self.entity_counts)

    @property
    def total_entities(self) -> int:
        return sum(self.entity_counts.values())


class PIISecurityGateway:
    """Composes detection, tokenization, leak verification and re-identification."""

    def __init__(
        self,
        detector: PIIDetector,
        tokenizer: PIITokenizer | None = None,
        reidentifier: Reidentifier | None = None,
    ) -> None:
        self._detector = detector
        self._tokenizer = tokenizer or PIITokenizer()
        self._reidentifier = reidentifier or Reidentifier()

    def protect(
        self,
        text: str,
        request_id: str | None = None,
        vault: TokenVault | None = None,
    ) -> ProtectedPrompt:
        """Sanitize `text` for use beyond the security boundary.

        Raises:
            PIIDetectionError: detection failed (fail closed, nothing is sent).
            TokenizationError: tokenization failed (fail closed).
            PIILeakError: a detected value survived tokenization (fail closed).
        """
        request_id = request_id or uuid.uuid4().hex
        text = normalize_input(text)
        entities = self._detector.detect(text)
        try:
            result = self._tokenizer.tokenize(text, entities, vault)
        except SecureAgentError:
            raise
        except Exception:
            # Never chain: the original message could quote the input.
            raise TokenizationError("PII tokenization failed") from None
        self.verify_no_leak(result.sanitized_text, result.vault)

        # Counts reflect entities found in *this* text, not the whole vault.
        counts: dict[str, int] = {}
        for entity in entities:
            counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1

        logger.info(
            "request_id=%s pii_entities=%d types=%s",
            request_id, len(entities), ",".join(sorted(counts)) or "-",
        )
        return ProtectedPrompt(request_id, result.sanitized_text, result.vault, counts)

    @staticmethod
    def verify_no_leak(sanitized_text: str, vault: TokenVault) -> None:
        """Fail closed if any known PII value still appears in the text.

        Tokens themselves are masked out first, so a detected value such as
        'Email' is not mistaken for a leak inside '<EMAIL_1>'. A NUL separator
        keeps text on either side of a token from joining into a false match.
        """
        haystack = TOKEN_PATTERN.sub("\x00", sanitized_text).casefold()
        for token in vault:
            value = vault.reveal(token)
            if value and value.casefold() in haystack:
                entity_type = vault.entity_type(token)
                # Report the type only, never the value.
                raise PIILeakError(f"Sanitized prompt still contains a {entity_type} value")

    def restore(
        self,
        llm_output: str,
        protected: ProtectedPrompt,
        policy: ReidentificationPolicy,
    ) -> ReidentificationResult:
        """Re-identify LLM output according to `policy`."""
        return self._reidentifier.reidentify(llm_output, protected.vault, policy)
