"""Exception handlers that return safe, generic error bodies.

Responses carry a stable error code and a fixed message per code. Exception
messages, stack traces, request bodies and internal state are never returned.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import ErrorCode, SecureAgentError
from app.models.chat import (AgentChatRequest, ApprovalDecisionRequest, ChatRequest,
                             ErrorDetail, ErrorResponse, RAGChatRequest)

logger = logging.getLogger(__name__)

# Unknown JSON keys are caller-controlled and can themselves contain PII.
# Only application-owned schema names and framework location markers are public.
_SAFE_VALIDATION_LOCATIONS = frozenset({'body', 'path', 'query', 'header', 'cookie', 'approval_id'}).union(
    *(model.model_fields for model in (ChatRequest, RAGChatRequest, AgentChatRequest, ApprovalDecisionRequest))
)


_NOT_SENT = "The request could not be processed securely and was not sent to the model."

# error code -> (HTTP status, client-facing message)
ERROR_RESPONSES: dict[ErrorCode, tuple[int, str]] = {
    ErrorCode.TOOL_UNKNOWN: (400,"The model requested an unknown tool."),
    ErrorCode.TOOL_DENIED: (403,"The requested tool is not permitted for this role."),
    ErrorCode.TOOL_ARGUMENTS_INVALID: (400,"The model supplied invalid tool arguments."),
    ErrorCode.TOOL_FAILED: (500,"The tool could not be processed securely."),
    ErrorCode.TOOL_RESULT_TOO_LARGE: (500,"The tool result exceeded the configured limit."),
    ErrorCode.AGENT_LIMIT_REACHED: (409,"The agent reached a configured execution limit."),
    ErrorCode.MCP_UNAVAILABLE: (503,"The local tool service is unavailable."),
    ErrorCode.MCP_TIMEOUT: (504,"The local tool service did not respond in time."),
    ErrorCode.APPROVAL_DENIED: (403,"The requested tool action was not approved."),
    ErrorCode.APPROVAL_NOT_FOUND: (404,"The approval request is unknown, expired or already decided."),
    ErrorCode.APPROVER_NOT_PERMITTED: (403,"The selected demo role may not decide approvals."),
    ErrorCode.EMBEDDING_FAILED: (503, "Local embeddings are unavailable."),
    ErrorCode.VECTOR_STORE_FAILED: (503, "Local retrieval index is unavailable."),
    ErrorCode.RETRIEVAL_FAILED: (503, "Local retrieval is unavailable."),
    ErrorCode.INGESTION_FAILED: (500, _NOT_SENT),
    ErrorCode.PII_DETECTION_FAILED: (500, _NOT_SENT),
    ErrorCode.TOKENIZATION_FAILED: (500, _NOT_SENT),
    ErrorCode.SECURITY_VALIDATION_FAILED: (500, _NOT_SENT),
    ErrorCode.REIDENTIFICATION_FAILED: (500, "The model response could not be safely finalized."),
    ErrorCode.LLM_UNAVAILABLE: (503, "The local language model service is unavailable."),
    ErrorCode.LLM_MODEL_UNAVAILABLE: (503, "The configured model is not available on the local model server."),
    ErrorCode.LLM_TIMEOUT: (504, "The local language model did not respond in time."),
    ErrorCode.LLM_CAPABILITY_UNSUPPORTED: (409, "The configured model does not support structured agent tools."),
    ErrorCode.LLM_BAD_RESPONSE: (502, "The local language model returned an invalid response."),
    ErrorCode.CONFIGURATION_ERROR: (500, "Internal server error."),
    ErrorCode.INTERNAL_ERROR: (500, "Internal server error."),
}


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error(request: Request, status: int, code: str, message: str, fields=None) -> JSONResponse:
    body = ErrorResponse(
        request_id=_request_id(request),
        error=ErrorDetail(code=code, message=message, fields=fields),
    )
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True))


async def handle_secure_agent_error(request: Request, exc: SecureAgentError) -> JSONResponse:
    status, message = ERROR_RESPONSES.get(exc.error_code, ERROR_RESPONSES[ErrorCode.INTERNAL_ERROR])
    if request.url.path == "/api/v1/agent/chat" and exc.error_code in {
        ErrorCode.PII_DETECTION_FAILED, ErrorCode.TOKENIZATION_FAILED, ErrorCode.SECURITY_VALIDATION_FAILED
    }:
        # An earlier sanitized planning turn may already have run. Do not imply
        # that the whole request was never sent when a tool follow-up is blocked.
        message = "The agent request could not be completed securely."
    return _error(request, status, exc.error_code.value, message)


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default 422 body echoes the submitted input, which may be PII.
    # Return allowlisted locations and generic types, never arbitrary input keys.
    fields = [
        {"loc": [str(part) if type(part) is int or (isinstance(part, str) and part in _SAFE_VALIDATION_LOCATIONS)
                 else "unknown_field" for part in err.get("loc", ())], "type": err.get("type", "invalid")}
        for err in exc.errors()
    ]
    return _error(request, 422, "INVALID_REQUEST", "The request body is invalid.", fields)


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error("request_id=%s unhandled %s", _request_id(request), type(exc).__name__)
    return _error(request, 500, ErrorCode.INTERNAL_ERROR.value, "Internal server error.")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(SecureAgentError, handle_secure_agent_error)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, handle_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, handle_unexpected_error)
