"""Validate the public API contract without importing backend implementation."""
from enum import Enum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class Mode(str, Enum):
    CHAT = "Secure Chat"
    RAG = "Secure RAG"
    AGENT = "Secure Agent"

ENDPOINTS = {Mode.CHAT:"/api/v1/chat", Mode.RAG:"/api/v1/rag/chat", Mode.AGENT:"/api/v1/agent/chat"}
# Demonstration approval endpoint; the id segment is validated before use.
APPROVALS_PATH = "/api/v1/agent/approvals/"
APPROVAL_ID_PATTERN = r"^[0-9a-f]{32}$"
ROLES = ("customer_service","claims_adjuster","supervisor")
PII_TYPES = ("PERSON","EMAIL_ADDRESS","PHONE_NUMBER","US_SSN")
RESTORABLE_TYPES = PII_TYPES[:3]
Tool = Literal["get_claim_status","get_policy_details","search_knowledge"]
Entity = Literal["PERSON","EMAIL_ADDRESS","PHONE_NUMBER","US_SSN"]
GraphNode = Literal["input_guard","router","secure_chat","secure_rag","tool_discovery","agent_planner","policy_gate",
                    "human_approval","mcp_tool","tool_result_guard","output_guard","finalize","safe_error"]

class PublicModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

class SecurityMetadata(PublicModel):
    pii_detected: bool
    entity_types: list[Entity] = Field(max_length=4)
    reidentification_decision: Literal["ALLOWED","PARTIAL","DENIED","NOT_REQUIRED"]
    document_pii_detected: bool = False
    tool_pii_detected: bool = False

class Source(PublicModel):
    document_id: str = Field(pattern=r"^[0-9a-f]{32}$", max_length=32)
    filename: str = Field(pattern=r"^(?:(?:claim|policy|customer)_[0-9]+|procedures_[a-z_]+|malicious_[a-z_]+)\.txt$",max_length=128)
    chunk_id: str = Field(pattern=r"^[0-9a-f]{32}-[0-9]+$",max_length=64)
    score: float = Field(ge=-1,le=1,allow_inf_nan=False)

class Orchestration(PublicModel):
    """Control-flow names only; never state, prompts, reasoning or tool payloads."""
    engine: Literal["langgraph"]
    route: Literal["chat","rag","agent"]
    nodes: list[GraphNode] = Field(max_length=128)
    tool_protocol: Literal["mcp"] | None = None
    mcp_transport: Literal["in-process"] | None = None
    mcp_servers: list[Literal["claims","policy","knowledge"]] = Field(default_factory=list,max_length=3)
    approval: Literal["NOT_REQUIRED","GRANTED","DENIED"] | None = None

class APIResponse(PublicModel):
    request_id: str = Field(pattern=r"^[A-Za-z0-9-]{1,64}$",max_length=64)
    response: str = Field(max_length=64000,repr=False)
    security: SecurityMetadata
    model: str = Field(pattern=r"^[A-Za-z0-9._:/-]{1,128}$",max_length=128)
    # Agent budgets allow up to ten searches with ten hits each.
    sources: list[Source] = Field(default_factory=list,max_length=100)
    tools_used: list[Tool] = Field(default_factory=list,max_length=10)
    agent_steps: int | None = Field(default=None,ge=1,le=10)
    orchestration: Orchestration | None = None

class ApprovalRequest(PublicModel):
    approval_id: str = Field(pattern=APPROVAL_ID_PATTERN,max_length=32)
    tool: Tool
    record_id: str | None = Field(default=None,pattern=r"^(?:CLM|POL)-[0-9]{4}$",max_length=8)
    reason: Literal["RESTRICTED_RECORD"]
    approver_role: Literal["supervisor"]
    expires_in_seconds: int = Field(ge=1,le=3600)

class ApprovalResponse(PublicModel):
    request_id: str = Field(pattern=r"^[A-Za-z0-9-]{1,64}$",max_length=64)
    status: Literal["APPROVAL_REQUIRED"]
    approval: ApprovalRequest
    model: str = Field(pattern=r"^[A-Za-z0-9._:/-]{1,128}$",max_length=128)

class HealthResponse(PublicModel):
    version: str = Field(default="unknown", pattern=r"^(?:unknown|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})$")
    status: Literal["healthy","degraded"]
    provider: Literal["ollama", "foundry"] = "ollama"
    llm: Literal["available","unavailable","model_missing","unknown"] | None = None
    tool_calling: bool = True
    ollama: Literal["available","unavailable","model_missing","unknown"]
    model: str = Field(pattern=r"^[A-Za-z0-9._:/-]{1,128}$",max_length=128)
