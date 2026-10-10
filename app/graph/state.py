"""Typed LangGraph state: sanitized data and control metadata only.

State is checkpointed (in memory) and may be inspected while debugging, so it
never holds raw user input, token vaults, raw MCP results, untrusted model
output or the re-identified response. Those live in the request-scoped
`RequestContext`, which LangGraph passes by reference and never checkpoints.
"""
import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    request_id: str
    mode: str                    # requested route, supplied by the application
    route: str                   # validated route
    role: str                    # informational; policy reads the trusted context
    sanitized_user_message: str
    messages: Annotated[list[dict], operator.add]   # sanitized, verified agent transcript
    sources: list[dict]          # document references only (no text)
    approved_tools: list[str]    # MCP tools that passed discovery allowlisting
    requested_tools: list[str]   # tool names proposed in the latest planner turn
    pending_calls: int           # authorized calls awaiting MCP execution
    tools_used: Annotated[list[str], operator.add]
    mcp_servers_used: Annotated[list[str], operator.add]
    agent_steps: int
    tool_calls: int
    planner_decision: str        # "tool" | "answer"
    policy_decision: str         # "allow" | "approval_required" | "deny"
    approval_status: str
    reidentification_decision: str
    security_metadata: dict      # booleans and entity type names only
    node_path: Annotated[list[str], operator.add]
    error_code: str | None


def initial_state(request_id: str, mode: str) -> GraphState:
    return GraphState(request_id=request_id, mode=mode, messages=[], tools_used=[], mcp_servers_used=[],
                      node_path=[], agent_steps=0, tool_calls=0, pending_calls=0, error_code=None,
                      approval_status="NOT_REQUIRED")
