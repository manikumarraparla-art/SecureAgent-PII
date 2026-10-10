"""PII detection tests for PERSON, EMAIL_ADDRESS, PHONE_NUMBER and US_SSN."""

from __future__ import annotations

import pytest

from app.core.exceptions import ConfigurationError, PIIDetectionError
from app.security.pii_detector import DetectedEntity, PIIDetector, resolve_overlaps


def _found(detector: PIIDetector, text: str) -> list[tuple[str, str]]:
    return [(e.entity_type, text[e.start : e.end]) for e in detector.detect(text)]


@pytest.mark.parametrize(
    ("text", "entity_type", "value"),
    [
        ("Please contact John Smith tomorrow.", "PERSON", "John Smith"),
        ("Meeting notes from Maria Garcia are attached.", "PERSON", "Maria Garcia"),
        ("Send it to john.smith@example.com today.", "EMAIL_ADDRESS", "john.smith@example.com"),
        ("Reply to ops-team+alerts@example.org please.", "EMAIL_ADDRESS", "ops-team+alerts@example.org"),
        ("Call me at (415) 555-0132.", "PHONE_NUMBER", "(415) 555-0132"),
        ("Her mobile is +1 212-555-0187.", "PHONE_NUMBER", "+1 212-555-0187"),
        ("My SSN is 536-22-1846.", "US_SSN", "536-22-1846"),
    ],
)
def test_detects_supported_entity_types(detector, text, entity_type, value):
    assert (entity_type, value) in _found(detector, text)


@pytest.mark.parametrize("ssn", ["123-45-6789", "078-05-1120", "666-12-3456", "000-12-3456"])
def test_strict_mode_masks_ssns_that_presidio_rejects(detector, ssn):
    """Presidio ignores sample/never-issued SSNs; the gateway must not."""
    assert ("US_SSN", ssn) in _found(detector, f"SSN on file: {ssn}")


def test_detects_all_entities_in_combined_example(detector):
    text = (
        "My name is John Smith. My email is john.smith@example.com. "
        "My SSN is 123-45-6789. My phone is 415-555-0132."
    )
    found = _found(detector, text)
    assert ("PERSON", "John Smith") in found
    assert ("EMAIL_ADDRESS", "john.smith@example.com") in found
    assert ("US_SSN", "123-45-6789") in found
    assert ("PHONE_NUMBER", "415-555-0132") in found


def test_results_are_ordered_and_non_overlapping(detector):
    text = "John Smith, john.smith@example.com, 123-45-6789, (415) 555-0132"
    entities = detector.detect(text)
    for left, right in zip(entities, entities[1:]):
        assert left.end <= right.start


def test_detected_entity_holds_no_value(detector):
    text = "Email john.smith@example.com"
    entity = detector.detect(text)[0]
    assert "john.smith" not in repr(entity)


@pytest.mark.parametrize("text", ["", "   ", "The quarterly report is due on Friday."])
def test_no_pii_returns_empty(detector, text):
    assert detector.detect(text) == []


def test_rejects_non_string_input(detector):
    with pytest.raises(PIIDetectionError):
        detector.detect(None)  # type: ignore[arg-type]


def test_rejects_unsupported_entity_configuration(detector):
    with pytest.raises(ConfigurationError):
        PIIDetector(entities=["CREDIT_CARD"], analyzer=detector._analyzer)


def test_analyzer_failure_is_wrapped_without_input_text():
    class BrokenAnalyzer:
        def analyze(self, **kwargs):
            raise RuntimeError(f"boom on {kwargs['text']}")

    broken = PIIDetector(analyzer=BrokenAnalyzer())  # type: ignore[arg-type]
    with pytest.raises(PIIDetectionError) as exc_info:
        broken.detect("John Smith 123-45-6789")
    assert "123-45-6789" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_missing_spacy_model_raises_helpful_error():
    with pytest.raises(PIIDetectionError, match="spacy download"):
        PIIDetector(spacy_model="en_core_web_does_not_exist")


def test_resolve_overlaps_prefers_higher_score_then_longer_span():
    entities = [
        DetectedEntity("PHONE_NUMBER", 0, 11, 0.4),
        DetectedEntity("US_SSN", 0, 11, 0.5),
        DetectedEntity("PERSON", 20, 24, 0.85),
        DetectedEntity("PERSON", 20, 30, 0.85),
    ]
    assert resolve_overlaps(entities) == [
        DetectedEntity("US_SSN", 0, 11, 0.5),
        DetectedEntity("PERSON", 20, 30, 0.85),
    ]
