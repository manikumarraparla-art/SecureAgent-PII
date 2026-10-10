"""Application exception hierarchy.

Exception messages must never include raw PII or token mappings: they may end
up in logs, HTTP error responses or tracebacks. Each exception carries a
stable `error_code` that is safe to return to clients and write to audit logs.
"""

from enum import Enum


class ErrorCode(str, Enum):
    """Audit- and client-safe error categories."""

    INTERNAL_ERROR = "INTERNAL_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    PII_DETECTION_FAILED = "PII_DETECTION_FAILED"
    TOKENIZATION_FAILED = "TOKENIZATION_FAILED"
    SECURITY_VALIDATION_FAILED = "SECURITY_VALIDATION_FAILED"
    REIDENTIFICATION_FAILED = "REIDENTIFICATION_FAILED"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_MODEL_UNAVAILABLE = "LLM_MODEL_UNAVAILABLE"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_CAPABILITY_UNSUPPORTED = "LLM_CAPABILITY_UNSUPPORTED"
    LLM_BAD_RESPONSE = "LLM_BAD_RESPONSE"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    VECTOR_STORE_FAILED = "VECTOR_STORE_FAILED"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    INGESTION_FAILED = "INGESTION_FAILED"
    TOOL_UNKNOWN = "TOOL_UNKNOWN"
    TOOL_DENIED = "TOOL_DENIED"
    TOOL_ARGUMENTS_INVALID = "TOOL_ARGUMENTS_INVALID"
    TOOL_FAILED = "TOOL_FAILED"
    TOOL_RESULT_TOO_LARGE = "TOOL_RESULT_TOO_LARGE"
    AGENT_LIMIT_REACHED = "AGENT_LIMIT_REACHED"
    MCP_UNAVAILABLE = "MCP_UNAVAILABLE"
    MCP_TIMEOUT = "MCP_TIMEOUT"
    APPROVAL_DENIED = "APPROVAL_DENIED"
    APPROVAL_NOT_FOUND = "APPROVAL_NOT_FOUND"
    APPROVER_NOT_PERMITTED = "APPROVER_NOT_PERMITTED"


class SecureAgentError(Exception):
    """Base class for all SecureAgent-PII errors."""

    error_code: ErrorCode = ErrorCode.INTERNAL_ERROR


class ConfigurationError(SecureAgentError):
    """Raised when configuration values are missing or invalid."""

    error_code = ErrorCode.CONFIGURATION_ERROR


class PIIDetectionError(SecureAgentError):
    """Raised when PII detection cannot be performed.

    Callers must treat this as fail-closed: the prompt is not sent anywhere.
    """

    error_code = ErrorCode.PII_DETECTION_FAILED


class TokenizationError(SecureAgentError):
    """Raised when detected PII cannot be tokenized (fail closed)."""

    error_code = ErrorCode.TOKENIZATION_FAILED


class PIILeakError(SecureAgentError):
    """Raised when a sanitized prompt still contains a detected PII value."""

    error_code = ErrorCode.SECURITY_VALIDATION_FAILED


class ReidentificationError(SecureAgentError):
    """Raised when re-identification cannot be performed safely."""

    error_code = ErrorCode.REIDENTIFICATION_FAILED


class LLMError(SecureAgentError):
    """Base class for LLM backend failures."""

    error_code = ErrorCode.INTERNAL_ERROR


class LLMUnavailableError(LLMError):
    """The local LLM server cannot be reached."""

    error_code = ErrorCode.LLM_UNAVAILABLE


class LLMModelNotFoundError(LLMError):
    """The configured model is not installed locally (it is never auto-pulled)."""

    error_code = ErrorCode.LLM_MODEL_UNAVAILABLE


class LLMTimeoutError(LLMError):
    """The LLM did not finish generating within the configured timeout."""

    error_code = ErrorCode.LLM_TIMEOUT


class LLMResponseError(LLMError):
    """The LLM server returned an error status or a malformed response."""

    error_code = ErrorCode.LLM_BAD_RESPONSE


class LLMCapabilityError(LLMError):
    """The configured model does not support this operation."""

    error_code = ErrorCode.LLM_CAPABILITY_UNSUPPORTED
