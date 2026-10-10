"""Reversible pseudonymization of detected PII.

Detected values are replaced by typed placeholders such as `<PERSON_1>` and
recorded in a `TokenVault`. The vault lives only in the application security
layer: it is never serialized, logged, or included in an LLM prompt.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from app.security.pii_detector import DetectedEntity

# Presidio entity type -> placeholder label shown to the LLM.
TOKEN_LABELS: dict[str, str] = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "US_SSN": "SSN",
}

TOKEN_PATTERN = re.compile(r"<([A-Z][A-Z_]*?)_(\d+)>")
_TOKEN_SPLIT = re.compile(r"(<[A-Z][A-Z_]*?_\d+>)")

# Entity types whose identity is the digit sequence, regardless of formatting.
_DIGIT_IDENTITY_TYPES = {"PHONE_NUMBER", "US_SSN"}


def _normalize(entity_type: str, value: str) -> str:
    """Canonical form used to give repeated mentions of one value one token."""
    if entity_type in _DIGIT_IDENTITY_TYPES:
        return "".join(ch for ch in value if ch.isdigit())
    return " ".join(value.split()).casefold()


class TokenVault:
    """Per-request (or per-conversation) mapping between tokens and PII values.

    Security properties:
    - `repr()`/`str()` never reveal values, so accidental logging is safe.
    - Pickling is refused, so the vault cannot be serialized into a prompt,
      cache or message queue by mistake.
    """

    __slots__ = ("_token_to_value", "_token_to_type", "_key_to_token", "_counters")

    def __init__(self) -> None:
        self._token_to_value: dict[str, str] = {}
        self._token_to_type: dict[str, str] = {}
        self._key_to_token: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}

    def __repr__(self) -> str:
        return f"TokenVault(<redacted>, size={len(self)})"

    __str__ = __repr__

    def __reduce__(self) -> object:
        raise TypeError("TokenVault must not be serialized")

    def __len__(self) -> int:
        return len(self._token_to_value)

    def __contains__(self, token: object) -> bool:
        return token in self._token_to_value

    def __iter__(self) -> Iterator[str]:
        """Iterate over tokens (not values)."""
        return iter(list(self._token_to_value))

    def get_or_create(self, entity_type: str, value: str, reserved: Iterable[str] = ()) -> str:
        """Return the token for `value`, allocating a new one if needed.

        Tokens listed in `reserved` (e.g. token-like strings the user typed)
        are skipped so a user-supplied `<PERSON_1>` never aliases real PII.
        """
        key = (entity_type, _normalize(entity_type, value))
        existing = self._key_to_token.get(key)
        if existing is not None:
            return existing

        label = TOKEN_LABELS.get(entity_type, entity_type)
        reserved_set = set(reserved)
        index = self._counters.get(label, 0)
        while True:
            index += 1
            token = f"<{label}_{index}>"
            if token not in reserved_set and token not in self._token_to_value:
                break
        self._counters[label] = index

        self._token_to_value[token] = value
        self._token_to_type[token] = entity_type
        self._key_to_token[key] = token
        return token

    def reveal(self, token: str) -> str | None:
        """Return the original value for `token`. Security-layer use only."""
        return self._token_to_value.get(token)

    def entity_type(self, token: str) -> str | None:
        return self._token_to_type.get(token)

    def values(self) -> list[str]:
        """Original values (security-layer use only, e.g. leak verification)."""
        return list(self._token_to_value.values())

    def items(self) -> list[tuple[str, str]]:
        """(token, value) pairs. Security-layer use only."""
        return list(self._token_to_value.items())

    def entity_counts(self) -> dict[str, int]:
        """Number of distinct values per entity type (safe to audit)."""
        counts: dict[str, int] = {}
        for entity_type in self._token_to_type.values():
            counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts

    def clear(self) -> None:
        """Drop all mappings (call when the request/conversation ends)."""
        self._token_to_value.clear()
        self._token_to_type.clear()
        self._key_to_token.clear()
        self._counters.clear()


@dataclass(frozen=True)
class TokenizationResult:
    """Sanitized text plus the vault needed to reverse it."""

    sanitized_text: str
    vault: TokenVault = field(repr=False)
    replaced_count: int


class PIITokenizer:
    """Replaces detected PII spans with consistent, typed tokens."""

    def tokenize(
        self,
        text: str,
        entities: Iterable[DetectedEntity],
        vault: TokenVault | None = None,
    ) -> TokenizationResult:
        """Replace each entity span in `text` with its token.

        Args:
            text: Raw input text.
            entities: Non-overlapping detected spans in `text`.
            vault: Existing vault to extend (multi-turn); a new one if omitted.

        After span replacement, any other case-insensitive occurrence of an
        already-tokenized value is also replaced. This covers mentions the NER
        model missed (e.g. a lowercase repeat of a detected name).
        """
        vault = vault if vault is not None else TokenVault()
        spans = sorted(entities, key=lambda e: e.start)
        self._validate_spans(text, spans)

        reserved = {m.group(0) for m in TOKEN_PATTERN.finditer(text)}

        # Allocate tokens in reading order so numbering is natural (<PERSON_1> first).
        tokens = [vault.get_or_create(e.entity_type, text[e.start : e.end], reserved) for e in spans]

        # Replace right-to-left so earlier offsets stay valid.
        sanitized = text
        for entity, token in sorted(zip(spans, tokens), key=lambda p: p[0].start, reverse=True):
            sanitized = sanitized[: entity.start] + token + sanitized[entity.end :]

        sanitized, propagated = self._propagate(sanitized, vault)
        return TokenizationResult(sanitized, vault, len(spans) + propagated)

    @staticmethod
    def _validate_spans(text: str, spans: list[DetectedEntity]) -> None:
        previous_end = 0
        for entity in spans:
            if not 0 <= entity.start < entity.end <= len(text):
                raise ValueError("Entity span is outside the input text")
            if entity.start < previous_end:
                raise ValueError("Entity spans must not overlap")
            previous_end = entity.end

    @staticmethod
    def _propagate(text: str, vault: TokenVault) -> tuple[str, int]:
        """Replace remaining verbatim occurrences of known values with tokens."""
        total = 0
        # Longest first so 'John Smith' wins over a shorter overlapping value.
        for token, value in sorted(vault.items(), key=lambda kv: len(kv[1]), reverse=True):
            if not value.strip():
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.IGNORECASE)
            # Split on existing tokens (odd indices) and only rewrite the text between them.
            pieces = _TOKEN_SPLIT.split(text)
            for i in range(0, len(pieces), 2):
                pieces[i], count = pattern.subn(token, pieces[i])
                total += count
            text = "".join(pieces)
        return text, total
