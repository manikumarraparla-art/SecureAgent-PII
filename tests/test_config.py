"""Configuration loading tests."""

from __future__ import annotations

import pytest

from app.core.config import load_settings
from app.core.exceptions import ConfigurationError

_KEYS = [
    "OLLAMA_MODEL", "PII_ENTITIES", "PII_SCORE_THRESHOLD", "STRICT_SSN_MODE",
    "OLLAMA_BASE_URL", "LOG_LEVEL", "OLLAMA_CONNECT_TIMEOUT", "OLLAMA_REQUEST_TIMEOUT",
    "AUDIT_LOG_PATH", "REIDENTIFICATION_ALLOWED_TYPES",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults_target_local_ollama(tmp_path):
    settings = load_settings(tmp_path / "missing.env")
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "qwen2.5:7b"
    assert settings.ollama_embed_model == "nomic-embed-text"
    assert settings.strict_ssn_mode is True
    assert set(settings.pii_entities) == {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"}


def test_reads_dotenv_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_MODEL=llama3.1:8b\nPII_ENTITIES=person, us_ssn\nLOG_LEVEL=debug\n")
    settings = load_settings(env_file)
    for key in _KEYS:  # load_dotenv mutates os.environ; let monkeypatch undo it
        monkeypatch.delenv(key, raising=False)
    assert settings.ollama_model == "llama3.1:8b"
    assert settings.pii_entities == ("PERSON", "US_SSN")
    assert settings.log_level == "DEBUG"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("PII_ENTITIES", "CREDIT_CARD"),
        ("PII_ENTITIES", " , "),
        ("PII_SCORE_THRESHOLD", "1.5"),
        ("PII_SCORE_THRESHOLD", "high"),
        ("STRICT_SSN_MODE", "maybe"),
        ("OLLAMA_BASE_URL", "https://api.example.com"),
        ("OLLAMA_BASE_URL", "http://10.0.0.5:11434"),
        ("OLLAMA_BASE_URL", "ftp://localhost:11434"),
        ("OLLAMA_CONNECT_TIMEOUT", "0"),
        ("OLLAMA_REQUEST_TIMEOUT", "soon"),
        ("REIDENTIFICATION_ALLOWED_TYPES", "PERSON,CREDIT_CARD"),
    ],
)
def test_invalid_values_raise(tmp_path, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ConfigurationError):
        load_settings(tmp_path / "missing.env")


def test_phase2_defaults(tmp_path):
    settings = load_settings(tmp_path / "missing.env")
    assert settings.ollama_connect_timeout == 5.0
    assert settings.ollama_request_timeout == 120.0
    assert settings.audit_log_path == "logs/audit.jsonl"
    assert settings.reidentification_ceiling == ("PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER")


@pytest.mark.parametrize("url", ["http://localhost:11434/", "http://127.0.0.1:11434", "http://[::1]:11434"])
def test_local_ollama_urls_accepted(tmp_path, monkeypatch, url):
    monkeypatch.setenv("OLLAMA_BASE_URL", url)
    assert load_settings(tmp_path / "missing.env").ollama_base_url == url.rstrip("/")


def test_empty_reidentification_ceiling_means_deny_all(tmp_path, monkeypatch):
    monkeypatch.setenv("REIDENTIFICATION_ALLOWED_TYPES", "")
    assert load_settings(tmp_path / "missing.env").reidentification_ceiling == ()
