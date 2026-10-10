"""Privacy-safe audit event schema.

The schema is closed: every field is an enum, a number, a validated
identifier or a validated model tag. There is no free-text field, so raw
prompts, LLM responses, PII values or token mappings cannot be recorded even
by mistake.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.agent.schemas import Role, ToolName
from app.core.config import SUPPORTED_PII_ENTITIES
from app.core.exceptions import ErrorCode
from app.graph.names import GraphNode, Route
from app.mcp.catalog import MCPServerName
from app.security.reidentifier import ReidentificationDecision

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")


class AuditEventType(str, Enum):
    AGENT_STARTED = "AGENT_STARTED"
    TOOL_REQUESTED = "TOOL_REQUESTED"
    TOOL_AUTHORIZED = "TOOL_AUTHORIZED"
    TOOL_DENIED = "TOOL_DENIED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    AGENT_COMPLETED = "AGENT_COMPLETED"
    AGENT_FAILED = "AGENT_FAILED"
    AGENT_STEP_LIMIT_REACHED = "AGENT_STEP_LIMIT_REACHED"
    PII_SANITIZED = "PII_SANITIZED"
    LLM_REQUEST_COMPLETED = "LLM_REQUEST_COMPLETED"
    LLM_REQUEST_FAILED = "LLM_REQUEST_FAILED"
    REQUEST_COMPLETED = "REQUEST_COMPLETED"
    REQUEST_FAILED = "REQUEST_FAILED"
    # Phase 8 security decisions (agent requests only; never chat/RAG records).
    MCP_TOOL_DISCOVERED = "MCP_TOOL_DISCOVERED"
    MCP_TOOL_REQUESTED = "MCP_TOOL_REQUESTED"
    MCP_TOOL_AUTHORIZED = "MCP_TOOL_AUTHORIZED"
    MCP_TOOL_DENIED = "MCP_TOOL_DENIED"
    MCP_TOOL_COMPLETED = "MCP_TOOL_COMPLETED"
    HUMAN_APPROVAL_REQUIRED = "HUMAN_APPROVAL_REQUIRED"
    HUMAN_APPROVAL_GRANTED = "HUMAN_APPROVAL_GRANTED"
    HUMAN_APPROVAL_DENIED = "HUMAN_APPROVAL_DENIED"


class AuditStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class AuditEvent:
    """One audit record. Validation rejects anything outside the schema."""

    request_id: str
    event: AuditEventType
    status: AuditStatus
    provider: str | None = None
    model: str | None = None
    pii_types: tuple[str, ...] = ()
    pii_count: int = 0
    reidentification: ReidentificationDecision | None = None
    latency_ms: float | None = None
    error_category: ErrorCode | None = None
    retrieval_enabled: bool = False
    documents_retrieved: int = 0
    chunk_count: int = 0
    embedding_model: str | None = None
    tool_name: ToolName | None = None
    role: Role | None = None
    authorized: bool | None = None
    agent_step: int | None = None
    mcp_server: MCPServerName | None = None
    timestamp: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.mcp_server is not None and not isinstance(self.mcp_server, MCPServerName):
            raise ValueError("Invalid audit MCP server")
        if self.provider is not None and self.provider not in ("ollama", "foundry"):
            raise ValueError("Invalid audit provider")
        if self.tool_name is not None and not isinstance(self.tool_name,ToolName):
            raise ValueError("Invalid audit tool")
        if self.role is not None and not isinstance(self.role,Role):
            raise ValueError("Invalid audit role")
        if self.authorized is not None and type(self.authorized) is not bool:
            raise ValueError("Invalid authorization flag")
        if self.agent_step is not None and (type(self.agent_step) is not int or not 0 <= self.agent_step <= 10):
            raise ValueError("Invalid agent step")
        if type(self.retrieval_enabled) is not bool:
            raise ValueError("Invalid retrieval flag")
        if any(type(x) is not int or x < 0 for x in (self.documents_retrieved, self.chunk_count)):
            raise ValueError("Invalid retrieval counts")
        if self.embedding_model is not None and not _MODEL_RE.fullmatch(self.embedding_model):
            raise ValueError("Invalid embedding model")
        if not _REQUEST_ID_RE.fullmatch(self.request_id):
            raise ValueError("Invalid audit request_id")
        if not isinstance(self.event, AuditEventType) or not isinstance(self.status, AuditStatus):
            raise ValueError("Audit event and status must be enum members")
        if self.model is not None and not _MODEL_RE.fullmatch(self.model):
            raise ValueError("Invalid audit model name")
        if not set(self.pii_types) <= set(SUPPORTED_PII_ENTITIES):
            raise ValueError("Audit pii_types must be supported entity type names")
        if not isinstance(self.pii_count, int) or self.pii_count < 0:
            raise ValueError("Audit pii_count must be a non-negative integer")
        if self.reidentification is not None and not isinstance(
            self.reidentification, ReidentificationDecision
        ):
            raise ValueError("Audit reidentification must be a ReidentificationDecision")
        if self.error_category is not None and not isinstance(self.error_category, ErrorCode):
            raise ValueError("Audit error_category must be an ErrorCode")
        # Normalize ordering so records are deterministic.
        object.__setattr__(self, "pii_types", tuple(sorted(set(self.pii_types))))

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict; optional fields that are unset are omitted."""
        record: dict[str, Any] = {
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "event": self.event.value,
            "status": self.status.value,
            "pii_types": list(self.pii_types),
            "pii_count": self.pii_count,
        }
        if self.provider is not None:
            record["provider"] = self.provider
        if self.model is not None:
            record["model"] = self.model
        if self.reidentification is not None:
            record["reidentification"] = self.reidentification.value
        if self.latency_ms is not None:
            record["latency_ms"] = round(self.latency_ms, 1)
        if self.error_category is not None:
            record["error_category"] = self.error_category.value
        if self.retrieval_enabled:
            record.update(retrieval_enabled=True, documents_retrieved=self.documents_retrieved,
                          chunk_count=self.chunk_count, embedding_model=self.embedding_model)
        if self.tool_name is not None: record['tool_name']=self.tool_name.value
        if self.role is not None: record['role']=self.role.value
        if self.authorized is not None: record['authorized']=self.authorized
        if self.agent_step is not None: record['agent_step']=self.agent_step
        if self.mcp_server is not None: record['mcp_server']=self.mcp_server.value
        return record


class GraphTraceType(str, Enum):
    GRAPH_STARTED = "GRAPH_STARTED"
    GRAPH_ROUTED = "GRAPH_ROUTED"
    GRAPH_NODE_ENTERED = "GRAPH_NODE_ENTERED"
    GRAPH_NODE_COMPLETED = "GRAPH_NODE_COMPLETED"
    GRAPH_INTERRUPTED = "GRAPH_INTERRUPTED"
    GRAPH_RESUMED = "GRAPH_RESUMED"
    GRAPH_COMPLETED = "GRAPH_COMPLETED"
    GRAPH_LIMIT_REACHED = "GRAPH_LIMIT_REACHED"


@dataclass(frozen=True)
class GraphTraceEvent:
    """Orchestration trace record, kept apart from the security audit log.

    Closed schema like `AuditEvent`: node and route names are enums, so graph
    state, prompts, tool payloads and model output cannot be recorded.
    """

    request_id: str
    event: GraphTraceType
    status: AuditStatus
    node: GraphNode | None = None
    route: Route | None = None
    step: int | None = None
    tool_name: ToolName | None = None
    provider: str | None = None
    model: str | None = None
    latency_ms: float | None = None
    error_category: ErrorCode | None = None
    timestamp: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not _REQUEST_ID_RE.fullmatch(self.request_id):
            raise ValueError("Invalid trace request_id")
        if not isinstance(self.event, GraphTraceType) or not isinstance(self.status, AuditStatus):
            raise ValueError("Trace event and status must be enum members")
        for value, kind in ((self.node, GraphNode), (self.route, Route), (self.tool_name, ToolName),
                            (self.error_category, ErrorCode)):
            if value is not None and not isinstance(value, kind):
                raise ValueError("Invalid trace field")
        if self.step is not None and (type(self.step) is not int or not 0 <= self.step <= 100):
            raise ValueError("Invalid trace step")
        if self.provider is not None and self.provider not in ("ollama", "foundry"):
            raise ValueError("Invalid trace provider")
        if self.model is not None and not _MODEL_RE.fullmatch(self.model):
            raise ValueError("Invalid trace model")
        if self.latency_ms is not None and (type(self.latency_ms) not in (int, float) or self.latency_ms < 0):
            raise ValueError("Invalid trace latency")

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {"timestamp": self.timestamp, "request_id": self.request_id,
                                  "event": self.event.value, "status": self.status.value}
        for key, value in (("node", self.node), ("route", self.route), ("tool_name", self.tool_name),
                           ("error_category", self.error_category)):
            if value is not None:
                record[key] = value.value
        for key, value in (("step", self.step), ("provider", self.provider), ("model", self.model)):
            if value is not None:
                record[key] = value
        if self.latency_ms is not None:
            record["latency_ms"] = round(self.latency_ms, 1)
        return record
