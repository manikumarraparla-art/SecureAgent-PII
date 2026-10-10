"""Graph topology: explicit nodes and conditional edges (see docs/langgraph-mcp.md).

    START -> input_guard -> router -+-> secure_chat ----------------------------+
                                    +-> secure_rag -----------------------------+-> output_guard -> finalize -> END
                                    +-> tool_discovery -> agent_planner -+------+
                                                            ^            | tool
                                                            |       policy_gate -+-> human_approval -+
                                                            |            | allow                     | approve
                                                            |         mcp_tool <---------------------+
                                                            |            |
                                                            +--- tool_result_guard (loops to mcp_tool while calls remain)

Any node failure routes to safe_error -> END. Policy denial and approval
denial are failures, so they take the same safe path.
"""
from langgraph.graph import END, START, StateGraph

from app.graph.context import RequestContext
from app.graph.names import GraphNode as N
from app.graph.state import GraphState

_ROUTES = {"chat": N.SECURE_CHAT.value, "rag": N.SECURE_RAG.value, "agent": N.TOOL_DISCOVERY.value}


def _failed(state) -> bool:
    return bool(state.get("error_code"))


def _next_or_error(target: N):
    def edge(state):
        return N.SAFE_ERROR.value if _failed(state) else target.value
    return edge


def route_after_router(state):
    if _failed(state):
        return N.SAFE_ERROR.value
    # Allowlisted routes only; anything else fails closed.
    return _ROUTES.get(state.get("route"), N.SAFE_ERROR.value)


def route_after_planner(state):
    if _failed(state):
        return N.SAFE_ERROR.value
    return N.POLICY_GATE.value if state.get("planner_decision") == "tool" else N.OUTPUT_GUARD.value


def route_after_policy(state):
    if _failed(state) or state.get("policy_decision") == "deny":
        return N.SAFE_ERROR.value
    if state.get("policy_decision") == "approval_required":
        return N.HUMAN_APPROVAL.value
    return N.MCP_TOOL.value if state.get("policy_decision") == "allow" else N.SAFE_ERROR.value


def route_after_guard(state):
    if _failed(state):
        return N.SAFE_ERROR.value
    return N.MCP_TOOL.value if state.get("pending_calls", 0) > 0 else N.AGENT_PLANNER.value


def build_secure_graph(nodes) -> StateGraph:
    graph = StateGraph(GraphState, context_schema=RequestContext)
    for node in N:
        graph.add_node(node.value, nodes.wrap(node, getattr(nodes, node.value)))
    graph.add_edge(START, N.INPUT_GUARD.value)
    graph.add_conditional_edges(N.INPUT_GUARD.value, _next_or_error(N.ROUTER), [N.ROUTER.value, N.SAFE_ERROR.value])
    graph.add_conditional_edges(N.ROUTER.value, route_after_router, [*_ROUTES.values(), N.SAFE_ERROR.value])
    for node in (N.SECURE_CHAT, N.SECURE_RAG):
        graph.add_conditional_edges(node.value, _next_or_error(N.OUTPUT_GUARD), [N.OUTPUT_GUARD.value, N.SAFE_ERROR.value])
    graph.add_conditional_edges(N.TOOL_DISCOVERY.value, _next_or_error(N.AGENT_PLANNER),
                                [N.AGENT_PLANNER.value, N.SAFE_ERROR.value])
    graph.add_conditional_edges(N.AGENT_PLANNER.value, route_after_planner,
                                [N.POLICY_GATE.value, N.OUTPUT_GUARD.value, N.SAFE_ERROR.value])
    graph.add_conditional_edges(N.POLICY_GATE.value, route_after_policy,
                                [N.MCP_TOOL.value, N.HUMAN_APPROVAL.value, N.SAFE_ERROR.value])
    graph.add_conditional_edges(N.HUMAN_APPROVAL.value, _next_or_error(N.MCP_TOOL), [N.MCP_TOOL.value, N.SAFE_ERROR.value])
    graph.add_conditional_edges(N.MCP_TOOL.value, _next_or_error(N.TOOL_RESULT_GUARD),
                                [N.TOOL_RESULT_GUARD.value, N.SAFE_ERROR.value])
    graph.add_conditional_edges(N.TOOL_RESULT_GUARD.value, route_after_guard,
                                [N.MCP_TOOL.value, N.AGENT_PLANNER.value, N.SAFE_ERROR.value])
    graph.add_conditional_edges(N.OUTPUT_GUARD.value, _next_or_error(N.FINALIZE), [N.FINALIZE.value, N.SAFE_ERROR.value])
    graph.add_edge(N.FINALIZE.value, END)
    graph.add_edge(N.SAFE_ERROR.value, END)
    return graph
