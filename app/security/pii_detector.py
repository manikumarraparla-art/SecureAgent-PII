"""PII detection built on Microsoft Presidio Analyzer.

The detector returns entity *positions* only. It never copies the detected
values into its results, so a `DetectedEntity` is safe to log or audit.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import spacy
import tldextract
from presidio_analyzer.predefined_recognizers import EmailRecognizer
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from app.core.config import SUPPORTED_PII_ENTITIES, Settings
from app.core.exceptions import ConfigurationError, PIIDetectionError

logger = logging.getLogger(__name__)

# Presidio loggers can emit analysis details at DEBUG level. Cap them so a
# global DEBUG log level can never write prompt text to the logs.
for _presidio_logger in ("presidio-analyzer", "presidio-anonymizer"):
    logging.getLogger(_presidio_logger).setLevel(logging.WARNING)

LANGUAGE = "en"


@dataclass(frozen=True)
class DetectedEntity:
    """Location and type of one PII entity in a text. Holds no PII value."""

    entity_type: str
    start: int
    end: int
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start


class StrictUsSsnRecognizer(PatternRecognizer):
    """Flags every formatted SSN-shaped string (NNN-NN-NNNN, NNN NN NNNN, NNN.NN.NNNN).

    Presidio's built-in `UsSsnRecognizer` intentionally rejects published
    sample SSNs (e.g. 123-45-6789) and never-issued ranges (000, 666, ...).
    That reduces false positives for data discovery, but a prompt egress
    gateway must fail closed: an SSN-shaped value is masked regardless.
    Unformatted 9-digit numbers are left to Presidio's own recognizer to avoid
    masking every 9-digit identifier.
    """

    # Same delimiter in both positions; dash, space or dot.
    PATTERNS = [
        Pattern(name="ssn_delimited", regex=r"(?<!\d)\d{3}([-. ])\d{2}\1\d{4}(?!\d)", score=0.5),
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="US_SSN",
            patterns=self.PATTERNS,
            context=["ssn", "social", "security"],
            supported_language=LANGUAGE,
            name="StrictUsSsnRecognizer",
        )


class OfflineEmailRecognizer(EmailRecognizer):
    """Validate against the bundled suffix snapshot; never fetch or cache it."""
    def __init__(self):
        super().__init__()
        self._extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

    def validate_result(self, pattern_text):
        return self._extract(pattern_text).fqdn != ""


def build_analyzer(spacy_model: str, strict_ssn_mode: bool = True) -> AnalyzerEngine:
    """Create a Presidio AnalyzerEngine backed by the given spaCy model.

    The model must already be installed. Presidio would otherwise try to
    download it from the internet at runtime (and exit the process if that
    fails), which is not acceptable for an offline security component.
    """
    if not spacy.util.is_package(spacy_model):
        raise PIIDetectionError(
            f"spaCy model {spacy_model!r} is not installed. "
            f"Install it with: python -m spacy download {spacy_model}"
        )
    try:
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": LANGUAGE, "model_name": spacy_model}],
            }
        )
        analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=[LANGUAGE])
    except (OSError, ValueError) as exc:
        raise PIIDetectionError(
            f"Could not load spaCy model {spacy_model!r}. "
            f"Install it with: python -m spacy download {spacy_model}"
        ) from exc

    analyzer.registry.remove_recognizer("EmailRecognizer")
    analyzer.registry.add_recognizer(OfflineEmailRecognizer())

    if strict_ssn_mode:
        analyzer.registry.add_recognizer(StrictUsSsnRecognizer())
    return analyzer


def resolve_overlaps(entities: Iterable[DetectedEntity]) -> list[DetectedEntity]:
    """Keep a non-overlapping subset, preferring higher score, then longer span.

    Example: '078-05-1120' can match both US_SSN (strict, 0.5) and
    PHONE_NUMBER (0.4); only the SSN is kept so the span is tokenized once.
    """
    ranked = sorted(entities, key=lambda e: (-e.score, -e.length, e.start))
    kept: list[DetectedEntity] = []
    for candidate in ranked:
        if all(candidate.end <= k.start or candidate.start >= k.end for k in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda e: e.start)


class PIIDetector:
    """Detects PII entities in free text using Presidio."""

    def __init__(
        self,
        entities: Sequence[str] = SUPPORTED_PII_ENTITIES,
        score_threshold: float = 0.35,
        spacy_model: str = "en_core_web_lg",
        strict_ssn_mode: bool = True,
        analyzer: AnalyzerEngine | None = None,
    ) -> None:
        unsupported = sorted(set(entities) - set(SUPPORTED_PII_ENTITIES))
        if unsupported:
            raise ConfigurationError(f"Unsupported entity types: {unsupported}")
        self.entities: tuple[str, ...] = tuple(entities)
        self.score_threshold = score_threshold
        self._analyzer = analyzer or build_analyzer(spacy_model, strict_ssn_mode)
        # spaCy pipelines are not documented as thread-safe; the API serves
        # requests from a thread pool, so serialize access to the analyzer.
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> PIIDetector:
        return cls(
            entities=settings.pii_entities,
            score_threshold=settings.pii_score_threshold,
            spacy_model=settings.spacy_model,
            strict_ssn_mode=settings.strict_ssn_mode,
        )

    def detect(self, text: str) -> list[DetectedEntity]:
        """Return non-overlapping PII entities found in `text`, ordered by position.

        Raises:
            PIIDetectionError: if analysis fails. Callers must fail closed.
        """
        if not isinstance(text, str):
            raise PIIDetectionError("Input text must be a string")
        if not text.strip():
            return []

        try:
            with self._lock:
                results: list[RecognizerResult] = self._analyzer.analyze(
                    text=text,
                    language=LANGUAGE,
                    entities=list(self.entities),
                    score_threshold=self.score_threshold,
                )
        except Exception as exc:  # Presidio/spaCy raise a variety of types
            # Do not chain the original exception text: it could quote input.
            raise PIIDetectionError(f"PII analysis failed ({type(exc).__name__})") from None

        entities = resolve_overlaps(
            DetectedEntity(r.entity_type, r.start, r.end, float(r.score)) for r in results
        )
        logger.debug("Detected %d PII entities", len(entities))
        return entities
