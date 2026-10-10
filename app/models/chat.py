"""Public API schemas.

Response models list their fields explicitly, so internal objects (token
vaults, sanitized prompts, detection scores, settings) cannot be serialized
into a response by accident.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.version import VERSION

EntityType = Literal["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"]

MAX_MESSAGE_CHARS = 8000


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    allowed_reidentification_types: list[EntityType] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Entity types the caller asks to see in clear text. Intersected with "
            "the server-side ceiling (REIDENTIFICATION_ALLOWED_TYPES). Default: none."
        ),
    )

    include_orchestration: bool = Field(
        default=False, strict=True,
        description="Return safe LangGraph/MCP execution metadata (route, node names, MCP servers).",
    )

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


GraphNodeName = Literal["input_guard", "router", "secure_chat", "secure_rag", "tool_discovery", "agent_planner",
                        "policy_gate", "human_approval", "mcp_tool", "tool_result_guard", "output_guard",
                        "finalize", "safe_error"]


class OrchestrationSummary(BaseModel):
    """Control-flow metadata only: no state, prompts, reasoning or tool payloads."""

    engine: Literal["langgraph"]
    route: Literal["chat", "rag", "agent"]
    nodes: list[GraphNodeName] = Field(max_length=128)
    tool_protocol: Literal["mcp"] | None = None
    mcp_transport: Literal["in-process"] | None = None
    mcp_servers: list[Literal["claims", "policy", "knowledge"]] | None = None
    approval: Literal["NOT_REQUIRED", "GRANTED", "DENIED"] | None = None


class SecuritySummary(BaseModel):
    pii_detected: bool
    entity_types: list[str]
    reidentification_decision: str


class ChatResponse(BaseModel):
    request_id: str
    response: str
    security: SecuritySummary
    model: str
    orchestration: OrchestrationSummary | None = None


class HealthResponse(BaseModel):
    version: str = VERSION
    status: Literal["healthy", "degraded"]
    ollama: str
    provider: Literal["ollama", "foundry"] = "ollama"
    llm: str = "unknown"
    tool_calling: bool = False
    model: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    fields: list[dict[str, object]] | None = None


class ErrorResponse(BaseModel):
    request_id: str | None
    error: ErrorDetail


class RAGChatRequest(ChatRequest):
    top_k: int | None = Field(default=None, ge=1, le=10, strict=True)

class SourceReference(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    score: float = Field(ge=-1, le=1, allow_inf_nan=False)

class RAGSecuritySummary(SecuritySummary):
    document_pii_detected: bool

class RAGChatResponse(ChatResponse):
    sources: list[SourceReference]
    security: RAGSecuritySummary


from app.agent.schemas import Role, ToolName

class AgentChatRequest(ChatRequest):
    include_sources: bool = Field(default=False,strict=True)
    role: Role = Role.CUSTOMER_SERVICE

class AgentSecuritySummary(RAGSecuritySummary):
    tool_pii_detected: bool

class AgentChatResponse(ChatResponse):
    sources: list[SourceReference] | None = None
    tools_used: list[ToolName]
    agent_steps: int = Field(ge=1,le=10)
    security: AgentSecuritySummary


class ApprovalSummary(BaseModel):
    approval_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    tool: ToolName
    record_id: str | None = Field(default=None, pattern=r"^(?:CLM|POL)-[0-9]{4}$")
    reason: Literal["RESTRICTED_RECORD"]
    approver_role: Literal["supervisor"]
    expires_in_seconds: int = Field(ge=1, le=3600)


class AgentApprovalResponse(BaseModel):
    """HTTP 202: the graph is paused before the MCP call until an approver decides."""

    request_id: str
    status: Literal["APPROVAL_REQUIRED"]
    approval: ApprovalSummary
    model: str


class ApprovalDecisionRequest(BaseModel):
    """Demonstration approval: the approver role is a demo label, not an identity."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "deny"]
    approver_role: Role
