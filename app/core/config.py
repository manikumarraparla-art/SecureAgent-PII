"""Application configuration loaded from environment variables / `.env`.

Settings are plain, immutable values. No secrets are required for local
operation; if cloud settings are added later they must come from the
environment, never from source code.
"""

from __future__ import annotations

import os
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from app.core.exceptions import ConfigurationError

SUPPORTED_PII_ENTITIES: tuple[str, ...] = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
)

# Re-identification ceiling: the most a caller may ever request to see in clear
# text. SSNs are excluded by default.
DEFAULT_REIDENTIFICATION_CEILING: tuple[str, ...] = ("PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER")

# The LLM must run on this machine; sanitized prompts never leave it.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    llm_provider: str = "ollama"
    foundry_model: str = "qwen3-1.7b-generic-gpu"
    foundry_base_url: str | None = None
    foundry_connect_timeout: float = 5.0
    foundry_request_timeout: float = 120.0

    max_agent_steps: int = 5
    max_agent_tool_calls: int = 5
    max_tool_result_chars: int = 16000

    rag_documents_dir: str = "data/documents"
    rag_index_path: str = "data/vectorstore/index.sqlite3"
    rag_collection: str = "synthetic"
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 150
    rag_top_k: int = 3

    # Phase 8 orchestration. MCP servers are in-process only; no URL is configurable.
    mcp_tool_timeout: float = 30.0
    approval_ttl_seconds: int = 300
    max_pending_approvals: int = 16
    graph_trace_log_path: str = "logs/graph_trace.jsonl"

    def __post_init__(self):
        if self.llm_provider not in ('ollama', 'foundry'):
            raise ConfigurationError('LLM_PROVIDER must be ollama or foundry')
        if self.foundry_base_url is not None:
            object.__setattr__(self, 'foundry_base_url', _parse_local_url(self.foundry_base_url))
        object.__setattr__(self, 'ollama_base_url', _parse_local_url(self.ollama_base_url))
        for name in ('ollama_model','ollama_embed_model','spacy_model','foundry_model'):
            value=getattr(self,name)
            if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}',value):
                raise ConfigurationError('Invalid model identifier')
        for name in ('ollama_connect_timeout','ollama_request_timeout','foundry_connect_timeout','foundry_request_timeout'):
            _validate_timeout(name,getattr(self,name))
        if type(self.strict_ssn_mode) is not bool:
            raise ConfigurationError('STRICT_SSN_MODE must be a boolean')
        if type(self.pii_score_threshold) not in (int,float) or not math.isfinite(self.pii_score_threshold) or not 0 <= self.pii_score_threshold <= 1:
            raise ConfigurationError('PII_SCORE_THRESHOLD must be between 0.0 and 1.0')
        for name in ('pii_entities','reidentification_ceiling'):
            value=getattr(self,name)
            if not isinstance(value,tuple) or any(v not in SUPPORTED_PII_ENTITIES for v in value) or (name=='pii_entities' and not value):
                raise ConfigurationError('Invalid PII entity configuration')
        if self.log_level not in ('DEBUG','INFO','WARNING','ERROR','CRITICAL'):
            raise ConfigurationError('Invalid LOG_LEVEL')
        if not isinstance(self.rag_collection,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',self.rag_collection):
            raise ConfigurationError('Invalid RAG collection')
        if any(type(v) is not int for v in (self.rag_chunk_size,self.rag_chunk_overlap,self.rag_top_k)):
            raise ConfigurationError('RAG limits must be integers')
        if any(type(v) is not int for v in (self.max_agent_steps,self.max_agent_tool_calls,self.max_tool_result_chars)):
            raise ConfigurationError("Agent limits must be integers")
        if not 1 <= self.max_agent_steps <= 10 or not 1 <= self.max_agent_tool_calls <= 10:
            raise ConfigurationError("Agent limits must be 1..10")
        if not 256 <= self.max_tool_result_chars <= 32000:
            raise ConfigurationError("Tool result limit must be 256..32000")
        if not 1 <= self.rag_chunk_size <= 16000 or not 0 <= self.rag_chunk_overlap < self.rag_chunk_size:
            raise ConfigurationError("Invalid RAG chunk configuration")
        if not 1 <= self.rag_top_k <= 10:
            raise ConfigurationError("Invalid RAG top_k")
        if type(self.mcp_tool_timeout) not in (int,float) or not math.isfinite(self.mcp_tool_timeout) or not 0 < self.mcp_tool_timeout <= 300:
            raise ConfigurationError("MCP_TOOL_TIMEOUT must be > 0 and <= 300 seconds")
        if type(self.approval_ttl_seconds) is not int or not 30 <= self.approval_ttl_seconds <= 3600:
            raise ConfigurationError("HUMAN_APPROVAL_TTL_SECONDS must be 30..3600")
        if type(self.max_pending_approvals) is not int or not 1 <= self.max_pending_approvals <= 100:
            raise ConfigurationError("MAX_PENDING_APPROVALS must be 1..100")

    app_env: str = "development"
    log_level: str = "INFO"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "nomic-embed-text"
    spacy_model: str = "en_core_web_lg"
    pii_entities: tuple[str, ...] = SUPPORTED_PII_ENTITIES
    pii_score_threshold: float = 0.35
    strict_ssn_mode: bool = True
    ollama_connect_timeout: float = 5.0
    ollama_request_timeout: float = 120.0
    audit_log_path: str = "logs/audit.jsonl"
    reidentification_ceiling: tuple[str, ...] = DEFAULT_REIDENTIFICATION_CEILING


def _parse_integer(name, raw):
    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(f"Invalid {name}") from None


def _parse_bool(name: str, raw: str) -> bool:
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ConfigurationError(f"{name} must be a boolean")


def _parse_entities(raw: str, name: str = "PII_ENTITIES", allow_empty: bool = False) -> tuple[str, ...]:
    entities = tuple(e.strip().upper() for e in raw.split(",") if e.strip())
    if not entities and not allow_empty:
        raise ConfigurationError(f"{name} must list at least one entity type")
    unsupported = sorted(set(entities) - set(SUPPORTED_PII_ENTITIES))
    if unsupported:
        raise ConfigurationError(
            f"Unsupported {name}; supported: {list(SUPPORTED_PII_ENTITIES)}"
        )
    return entities


def _parse_timeout(name: str, raw: str) -> float:
    try:
        value = float(raw)
    except (ValueError,TypeError):
        raise ConfigurationError(f"{name} must be a number of seconds") from None
    return _validate_timeout(name,value)


def _validate_timeout(name: str, value: float) -> float:
    if type(value) not in (int,float) or not math.isfinite(value) or not 0.0 < value <= 3600.0:
        raise ConfigurationError(f"{name} must be > 0 and <= 3600 seconds")
    return value


def _parse_local_url(raw: str) -> str:
    """Accept only http(s) URLs that point at this machine."""
    try:
        if not isinstance(raw,str) or any(c.isspace() for c in raw) or '%' in raw or '\\' in raw:
            raise ValueError
        parsed = urlparse(raw)
        if parsed.scheme not in {'http','https'} or parsed.hostname not in _LOCAL_HOSTS:
            raise ValueError
        if parsed.username is not None or parsed.password is not None or parsed.path not in ('','/') or parsed.params or parsed.query or parsed.fragment:
            raise ValueError
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError
        return raw.rstrip('/')
    except (ValueError,TypeError):
        raise ConfigurationError('Local provider URL must be a loopback http(s) base URL without credentials or other URL components') from None


def _parse_threshold(raw: str) -> float:
    try:
        value = float(raw)
    except (ValueError,TypeError):
        raise ConfigurationError("PII_SCORE_THRESHOLD must be a number") from None
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError("PII_SCORE_THRESHOLD must be between 0.0 and 1.0")
    return value


# LangGraph depends on langchain-core, which can export runs to LangSmith when
# these variables are enabled. Graph state is sanitized, but nothing may leave
# this machine, so enabling cloud tracing is a startup error (fail closed).
CLOUD_TRACING_VARIABLES = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2")


def ensure_cloud_tracing_disabled() -> None:
    for name in CLOUD_TRACING_VARIABLES:
        value = os.environ.get(name, "").strip().lower()
        if value and value not in _FALSE_VALUES:
            raise ConfigurationError("Cloud tracing must remain disabled; unset LangSmith/LangChain tracing variables")


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Build settings from the environment, optionally loading a `.env` file.

    Real environment variables take precedence over values in the file.
    """
    load_dotenv(dotenv_path=env_file, override=False)
    ensure_cloud_tracing_disabled()
    defaults = Settings()
    env = os.environ.get

    return Settings(
        llm_provider=env("LLM_PROVIDER", defaults.llm_provider),
        foundry_model=env("FOUNDRY_MODEL", defaults.foundry_model),
        foundry_base_url=env("FOUNDRY_BASE_URL") or None,
        foundry_connect_timeout=_parse_timeout("FOUNDRY_CONNECT_TIMEOUT", env("FOUNDRY_CONNECT_TIMEOUT", str(defaults.foundry_connect_timeout))),
        foundry_request_timeout=_parse_timeout("FOUNDRY_REQUEST_TIMEOUT", env("FOUNDRY_REQUEST_TIMEOUT", str(defaults.foundry_request_timeout))),
        max_agent_steps=_parse_integer("MAX_AGENT_STEPS",env("MAX_AGENT_STEPS",str(defaults.max_agent_steps))),
        max_agent_tool_calls=_parse_integer("MAX_AGENT_TOOL_CALLS",env("MAX_AGENT_TOOL_CALLS",str(defaults.max_agent_tool_calls))),
        max_tool_result_chars=_parse_integer("MAX_TOOL_RESULT_CHARS",env("MAX_TOOL_RESULT_CHARS",str(defaults.max_tool_result_chars))),
        rag_documents_dir=env("RAG_DOCUMENTS_DIR", defaults.rag_documents_dir),
        rag_index_path=env("RAG_INDEX_PATH", defaults.rag_index_path),
        rag_collection=env("RAG_COLLECTION", defaults.rag_collection),
        rag_chunk_size=_parse_integer("RAG_CHUNK_SIZE", env("RAG_CHUNK_SIZE", str(defaults.rag_chunk_size))),
        rag_chunk_overlap=_parse_integer("RAG_CHUNK_OVERLAP", env("RAG_CHUNK_OVERLAP", str(defaults.rag_chunk_overlap))),
        rag_top_k=_parse_integer("RAG_TOP_K", env("RAG_TOP_K", str(defaults.rag_top_k))),
        mcp_tool_timeout=_parse_timeout("MCP_TOOL_TIMEOUT", env("MCP_TOOL_TIMEOUT", str(defaults.mcp_tool_timeout))),
        approval_ttl_seconds=_parse_integer("HUMAN_APPROVAL_TTL_SECONDS", env("HUMAN_APPROVAL_TTL_SECONDS", str(defaults.approval_ttl_seconds))),
        max_pending_approvals=_parse_integer("MAX_PENDING_APPROVALS", env("MAX_PENDING_APPROVALS", str(defaults.max_pending_approvals))),
        graph_trace_log_path=env("GRAPH_TRACE_LOG_PATH", defaults.graph_trace_log_path),
        app_env=env("APP_ENV", defaults.app_env),
        log_level=env("LOG_LEVEL", defaults.log_level).upper(),
        ollama_base_url=_parse_local_url(env("OLLAMA_BASE_URL", defaults.ollama_base_url)),
        ollama_model=env("OLLAMA_MODEL", defaults.ollama_model),
        ollama_embed_model=env("OLLAMA_EMBED_MODEL", defaults.ollama_embed_model),
        spacy_model=env("SPACY_MODEL", defaults.spacy_model),
        pii_entities=_parse_entities(env("PII_ENTITIES", ",".join(defaults.pii_entities))),
        pii_score_threshold=_parse_threshold(
            env("PII_SCORE_THRESHOLD", str(defaults.pii_score_threshold))
        ),
        strict_ssn_mode=_parse_bool(
            "STRICT_SSN_MODE", env("STRICT_SSN_MODE", str(defaults.strict_ssn_mode))
        ),
        ollama_connect_timeout=_parse_timeout(
            "OLLAMA_CONNECT_TIMEOUT", env("OLLAMA_CONNECT_TIMEOUT", str(defaults.ollama_connect_timeout))
        ),
        ollama_request_timeout=_parse_timeout(
            "OLLAMA_REQUEST_TIMEOUT", env("OLLAMA_REQUEST_TIMEOUT", str(defaults.ollama_request_timeout))
        ),
        audit_log_path=env("AUDIT_LOG_PATH", defaults.audit_log_path),
        reidentification_ceiling=_parse_entities(
            env("REIDENTIFICATION_ALLOWED_TYPES", ",".join(defaults.reidentification_ceiling)),
            name="REIDENTIFICATION_ALLOWED_TYPES",
            allow_empty=True,
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings (loaded once)."""
    return load_settings()
