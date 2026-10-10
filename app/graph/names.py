"""Closed vocabularies for graph routing and privacy-safe tracing (no LangGraph import)."""
from enum import Enum


class Route(str, Enum):
    CHAT = "chat"
    RAG = "rag"
    AGENT = "agent"


class GraphNode(str, Enum):
    INPUT_GUARD = "input_guard"
    ROUTER = "router"
    SECURE_CHAT = "secure_chat"
    SECURE_RAG = "secure_rag"
    TOOL_DISCOVERY = "tool_discovery"
    AGENT_PLANNER = "agent_planner"
    POLICY_GATE = "policy_gate"
    HUMAN_APPROVAL = "human_approval"
    MCP_TOOL = "mcp_tool"
    TOOL_RESULT_GUARD = "tool_result_guard"
    OUTPUT_GUARD = "output_guard"
    FINALIZE = "finalize"
    SAFE_ERROR = "safe_error"


class ApprovalStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    GRANTED = "GRANTED"
    DENIED = "DENIED"
