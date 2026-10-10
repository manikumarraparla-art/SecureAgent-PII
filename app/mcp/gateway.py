"""MCP Tool Gateway: the only path from the agent to an MCP server.

    discovery -> allowlist + server binding + schema pin -> request catalog
    call      -> catalog check -> MCP tools/call -> strict result decoding

Role authorization, record scope, argument validation and approval run in
`ToolGateway` before `call`; this layer adds the protocol-level controls.
Results are returned raw and must pass the tool-result guard before any
model sees them.
"""
import json
import time
from types import MappingProxyType

import mcp_types as types

from app.agent.errors import ToolArgumentError, ToolDeniedError, ToolExecutionError, ToolResultTooLarge
from app.agent.schemas import ToolName
from app.audit.events import AuditEventType, AuditStatus
from app.core.exceptions import ErrorCode, PIIDetectionError, PIILeakError, SecureAgentError, TokenizationError
from app.mcp.catalog import APPROVED_TOOL_BINDINGS, PINNED_FINGERPRINTS, MCPServerName, schema_fingerprint
from app.mcp.client import LocalMCPClient, MCPUnavailableError
from app.mcp.host import LocalMCPHost
from app.rag.embeddings import EmbeddingError
from app.rag.retriever import RetrievalError
from app.rag.vector_store import VectorStoreError

# Server-reported failure codes that keep their meaning; anything else is TOOL_FAILED.
_SERVER_ERRORS = MappingProxyType({
    ErrorCode.TOOL_ARGUMENTS_INVALID: ToolArgumentError,
    ErrorCode.TOOL_RESULT_TOO_LARGE: ToolResultTooLarge,
    ErrorCode.RETRIEVAL_FAILED: RetrievalError,
    ErrorCode.EMBEDDING_FAILED: EmbeddingError,
    ErrorCode.VECTOR_STORE_FAILED: VectorStoreError,
    ErrorCode.PII_DETECTION_FAILED: PIIDetectionError,
    ErrorCode.TOKENIZATION_FAILED: TokenizationError,
    ErrorCode.SECURITY_VALIDATION_FAILED: PIILeakError,
})


def _reject_constant(value):
    raise ValueError("Non-finite JSON number")


class MCPToolGateway:
    def __init__(self, host: LocalMCPHost, client: LocalMCPClient, max_result_chars: int = 16000):
        self.host = host
        self._client = client
        self._max_result_chars = max_result_chars

    @classmethod
    def from_registry(cls, registry, *, timeout=30.0, max_result_chars=16000, server_overrides=None, untrusted_servers=()):
        return cls(LocalMCPHost(registry, server_overrides, untrusted_servers), LocalMCPClient(timeout), max_result_chars)

    def discover(self, session, record):
        """Approved tools for this request, each bound to its one approved server."""
        catalog = {}
        for binding, server in self.host.servers(session):
            try:
                listing = self._client.list_tools(server)
            except SecureAgentError as exc:
                record(AuditEventType.MCP_TOOL_DISCOVERED, AuditStatus.FAILURE, server=binding, error=exc.error_code)
                continue
            # The configured binding decides identity; a self-declared name must agree.
            trusted = binding is not MCPServerName.UNAPPROVED and listing.server_name == binding.value
            for advertised, schema in listing.tools:
                try:
                    name = ToolName(advertised)
                except (ValueError, TypeError):
                    name = None
                approved = (trusted and name is not None and APPROVED_TOOL_BINDINGS[name] is binding
                            and schema_fingerprint(schema) == PINNED_FINGERPRINTS[name] and name not in catalog)
                if approved:
                    catalog[name] = binding
                # Unapproved names are never recorded: they are server-controlled text.
                record(AuditEventType.MCP_TOOL_DISCOVERED, AuditStatus.SUCCESS if approved else AuditStatus.FAILURE,
                       tool=name, authorized=approved, server=binding)
        if not catalog:
            raise MCPUnavailableError("No approved MCP tools are available")
        return MappingProxyType(catalog)

    def call(self, name: ToolName, arguments: dict, session, record, step: int, catalog) -> dict:
        started = time.perf_counter()
        record(AuditEventType.MCP_TOOL_REQUESTED, AuditStatus.SUCCESS, tool=name, step=step)
        if catalog is None:
            catalog = self.discover(session, record)
        binding = catalog.get(name)
        if binding is None or APPROVED_TOOL_BINDINGS.get(name) is not binding:
            record(AuditEventType.MCP_TOOL_DENIED, AuditStatus.FAILURE, tool=name, step=step, authorized=False)
            raise ToolDeniedError("Tool is not available from an approved MCP server")
        record(AuditEventType.MCP_TOOL_AUTHORIZED, AuditStatus.SUCCESS, tool=name, step=step, authorized=True, server=binding)
        try:
            raw = self._decode(self._client.call_tool(self.host.server(binding, session), name.value, arguments))
        except SecureAgentError as exc:
            record(AuditEventType.MCP_TOOL_COMPLETED, AuditStatus.FAILURE, tool=name, step=step, server=binding,
                   error=exc.error_code, latency_ms=(time.perf_counter() - started) * 1000)
            raise
        record(AuditEventType.MCP_TOOL_COMPLETED, AuditStatus.SUCCESS, tool=name, step=step, server=binding,
               latency_ms=(time.perf_counter() - started) * 1000)
        return raw

    def _decode(self, result) -> dict:
        """Untrusted result: exactly one bounded JSON-object text block."""
        content = getattr(result, "content", None)
        if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], types.TextContent):
            raise ToolExecutionError("Malformed MCP result")
        text = content[0].text
        if not isinstance(text, str):
            raise ToolExecutionError("Malformed MCP result")
        if getattr(result, "is_error", False):
            raise self._server_error(text)
        if len(text) > self._max_result_chars:
            raise ToolResultTooLarge("Tool result too large")
        try:
            value = json.loads(text, parse_constant=_reject_constant)
        except (ValueError, RecursionError):
            raise ToolExecutionError("Malformed MCP result") from None
        if not isinstance(value, dict):
            raise ToolExecutionError("Malformed MCP result")
        return value

    @staticmethod
    def _server_error(text: str) -> SecureAgentError:
        try:
            code = ErrorCode(json.loads(text[:256])["error_code"])
        except (ValueError, TypeError, KeyError, RecursionError):
            code = None
        return _SERVER_ERRORS.get(code, ToolExecutionError)("Tool processing failed")
