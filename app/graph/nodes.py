"""LangGraph nodes. Each has one narrow responsibility.

Nodes reach the model only through the SecureLLMService boundary and tools
only through the ToolGateway (policy -> MCP gateway). They never import a
provider client, MCP client/server or concrete tool. A node failure becomes
`error_code` in state and a conditional edge routes to `safe_error`.
"""
import secrets
import time

from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from app.agent.approval import may_approve
from app.agent.errors import AgentLimitError, ApprovalDeniedError, ToolArgumentError, ToolDeniedError, UnknownToolError
from app.agent.planner import AGENT_INSTRUCTIONS, assistant_tool_message, call_key, request_scope, tool_message
from app.agent.schemas import Role, ToolName, tool_definitions
from app.audit.events import AuditEventType, AuditStatus, GraphTraceType
from app.core.exceptions import LLMCapabilityError, LLMResponseError, SecureAgentError
from app.graph.names import ApprovalStatus, GraphNode, Route
from app.llm.base import ModelTurn
from app.mcp.catalog import MCP_TRANSPORT

_POLICY_DENIALS = (ToolDeniedError, UnknownToolError, ToolArgumentError)


def _tool_label(name) -> str:
    # Model-proposed names are untrusted text; only approved names are kept.
    try:
        return ToolName(name).value
    except (ValueError, TypeError):
        return "unknown"


class SecureGraphNodes:
    def __init__(self, boundary, tools, max_steps: int, max_calls: int, trace):
        self.boundary = boundary
        self.tools = tools
        self.max_steps = max_steps
        self.max_calls = max_calls
        self._trace = trace

    # ------------------------------------------------------------ plumbing

    def trace(self, ctx, event, status, **fields) -> None:
        self._trace(ctx, event, status, **fields)

    def wrap(self, node: GraphNode, fn):
        def run(state, runtime):
            ctx = runtime.context
            self.trace(ctx, GraphTraceType.GRAPH_NODE_ENTERED, AuditStatus.SUCCESS, node=node,
                       step=state.get("agent_steps"))
            started = time.perf_counter()
            try:
                update = dict(fn(state, ctx) or {})
                status = AuditStatus.SUCCESS
            except GraphBubbleUp:
                raise  # LangGraph interrupt (human approval pause), not a failure
            except SecureAgentError as exc:
                ctx.error, status = exc, AuditStatus.FAILURE
                update = {"error_code": exc.error_code.value}
                if node is GraphNode.POLICY_GATE and isinstance(exc, _POLICY_DENIALS):
                    update["policy_decision"] = "deny"
                if isinstance(exc, ApprovalDeniedError):
                    update["approval_status"] = ApprovalStatus.DENIED.value
            except Exception:
                ctx.error, status = self.boundary.internal_error(ctx.mode), AuditStatus.FAILURE
                update = {"error_code": ctx.error.error_code.value}
            update["node_path"] = [node.value]
            self.trace(ctx, GraphTraceType.GRAPH_NODE_COMPLETED, status, node=node, step=state.get("agent_steps"),
                       latency_ms=(time.perf_counter() - started) * 1000,
                       error_category=ctx.error.error_code if status is AuditStatus.FAILURE else None)
            return update
        return run

    def record(self, ctx):
        return self.boundary.agent_recorder(ctx)

    # ------------------------------------------------------------ shared

    def input_guard(self, state, ctx):
        """Protect user input before anything else sees it; derive agent record scope."""
        if ctx.mode is Route.AGENT:
            ctx.role = Role(ctx.role)
            ctx.session = self.boundary.new_tool_session(ctx)
        protected = self.boundary.protect_input(ctx)
        if ctx.mode is Route.CHAT:
            self.boundary.record_chat_sanitized(ctx)
        update = {"sanitized_user_message": protected.sanitized_text,
                  "security_metadata": {"pii_detected": protected.total_entities > 0,
                                        "entity_types": list(protected.entity_types)}}
        if ctx.mode is Route.AGENT:
            update["role"] = ctx.role.value
            self.record(ctx)(AuditEventType.AGENT_STARTED, AuditStatus.SUCCESS)
            # Input policy: one claim and one policy per request, never a bulk sweep.
            ctx.scope = request_scope(protected.sanitized_text)
        return update

    def router(self, state, ctx):
        """Route selection only. Authorization is never decided here."""
        route = Route(state["mode"])
        if route is not ctx.mode:
            raise ValueError("Route mismatch")
        if route is Route.AGENT and (self.tools is None or not self.boundary.tool_calling_supported):
            raise LLMCapabilityError("Structured agent capability unavailable")
        self.trace(ctx, GraphTraceType.GRAPH_ROUTED, AuditStatus.SUCCESS, route=route)
        return {"route": route.value}

    def secure_chat(self, state, ctx):
        ctx.draft = self.boundary.generate_chat(ctx)
        return {}

    def secure_rag(self, state, ctx):
        # Existing Phase 3 retrieval and document protection, unchanged.
        ctx.draft = self.boundary.generate_rag(ctx)
        metadata = dict(state.get("security_metadata") or {}, document_pii_detected=bool(ctx.document_counts))
        return {"sources": [dict(source) for source in ctx.sources], "security_metadata": metadata}

    def output_guard(self, state, ctx):
        """Fail closed on guessed protected values, then apply the re-identification policy."""
        text = ctx.draft
        if ctx.mode is Route.AGENT:
            self.boundary.verify_agent_answer(ctx, text)
        ctx.restored = self.boundary.restore(ctx, text)
        ctx.draft = None
        return {"reidentification_decision": ctx.restored.decision.value}

    def finalize(self, state, ctx):
        summary = self._orchestration_summary(state) if ctx.include_orchestration else None
        ctx.result = self.boundary.complete_request(ctx, state, summary)
        return {}

    def safe_error(self, state, ctx):
        self.boundary.record_failure(ctx)
        return {}

    # ------------------------------------------------------------ agent

    def tool_discovery(self, state, ctx):
        """MCP discovery filtered through the application allowlist."""
        ctx.catalog = self.tools.discover(ctx.session, self.record(ctx))
        return {"approved_tools": [tool.value for tool in ctx.catalog]}

    def agent_planner(self, state, ctx):
        """One structured model turn: either a final answer or proposed tool calls."""
        step = state.get("agent_steps", 0) + 1
        history = list(state.get("messages") or [])
        new = [] if history else [{"role": "user", "content": state["sanitized_user_message"]}]
        messages = [{"role": "system", "content": self.boundary.system_prompt + AGENT_INSTRUCTIONS}, *history, *new]
        turn = self.boundary.complete_agent(ctx, messages, tool_definitions(ctx.catalog))
        if not isinstance(turn, ModelTurn):
            raise LLMResponseError("Invalid model turn")
        update = {"agent_steps": step, "messages": new}
        if not turn.tool_calls:
            if not turn.content.strip():
                raise LLMResponseError("Empty agent answer")
            ctx.draft = turn.content
            return {**update, "planner_decision": "answer", "requested_tools": []}
        # A final answer also consumes a step; requesting a tool on the last step stops.
        if step == self.max_steps or state.get("tool_calls", 0) + len(turn.tool_calls) > self.max_calls:
            self.record(ctx)(AuditEventType.AGENT_STEP_LIMIT_REACHED, AuditStatus.FAILURE, step=step)
            raise AgentLimitError("Agent budget reached")
        ctx.proposed = tuple(turn.tool_calls)
        return {**update, "planner_decision": "tool",
                "requested_tools": [_tool_label(call.name) for call in turn.tool_calls]}

    def policy_gate(self, state, ctx):
        """Deterministic policy for every proposed call, before any of them executes."""
        step, record = state["agent_steps"], self.record(ctx)
        proposed, ctx.proposed = ctx.proposed, ()
        prepared = [self.tools.authorize_and_validate(call, ctx.role, record, step, ctx.scope) for call in proposed]
        keys = [call_key(name, arguments) for name, arguments in prepared]
        if len(set(keys)) != len(keys) or any(key in ctx.seen_calls for key in keys):
            record(AuditEventType.AGENT_STEP_LIMIT_REACHED, AuditStatus.FAILURE, step=step)
            raise AgentLimitError("Repeated tool request")
        ctx.seen_calls.update(keys)
        ctx.prepared = list(prepared)
        restricted = [call for call in prepared if self.tools.approval_required(call, ctx.role)
                      and call_key(*call) not in ctx.approved_calls]
        if not restricted:
            return {"policy_decision": "allow", "pending_calls": len(prepared)}
        ctx.approval_id = secrets.token_hex(16)
        name, arguments = restricted[0]
        ctx.approval_request = {"tool": name.value,
                                "record_id": arguments.get("claim_id") or arguments.get("policy_id")}
        for name, _ in restricted:
            record(AuditEventType.HUMAN_APPROVAL_REQUIRED, AuditStatus.SUCCESS, tool=name, step=step)
        return {"policy_decision": "approval_required", "pending_calls": len(prepared),
                "approval_status": ApprovalStatus.PENDING.value}

    def human_approval(self, state, ctx):
        """Pause the graph until an approver decides. Must have no side effects before
        `interrupt()`: LangGraph re-executes this node from the top on resume."""
        decision = interrupt({"approval_id": ctx.approval_id, **(ctx.approval_request or {})})
        try:
            approve = decision["decision"]
            approver = Role(decision["approver_role"])
        except (KeyError, TypeError, ValueError):
            raise ApprovalDeniedError("Invalid approval decision") from None
        if approve not in ("approve", "deny") or not may_approve(approver):
            raise ApprovalDeniedError("Invalid approval decision")
        step, record = state["agent_steps"], self.record(ctx)
        restricted = [call for call in ctx.prepared if self.tools.approval_required(call, ctx.role)]
        if approve == "deny":
            for name, _ in restricted:
                record(AuditEventType.HUMAN_APPROVAL_DENIED, AuditStatus.FAILURE, tool=name, step=step, authorized=False)
            raise ApprovalDeniedError("Approval denied")
        ctx.approved_calls = ctx.approved_calls | {call_key(*call) for call in restricted}
        for name, _ in restricted:
            record(AuditEventType.HUMAN_APPROVAL_GRANTED, AuditStatus.SUCCESS, tool=name, step=step, authorized=True)
        return {"approval_status": ApprovalStatus.GRANTED.value}

    def mcp_tool(self, state, ctx):
        """Execute exactly one authorized call through the MCP gateway; quarantine the raw result."""
        step = state["agent_steps"]
        prepared = ctx.prepared.pop(0)
        started = time.perf_counter()
        name, arguments, raw = self.tools.execute_raw(prepared, ctx.session, self.record(ctx), step, ctx.role,
                                                      ctx.scope, ctx.catalog, ctx.approved_calls)
        ctx.quarantine.append((name, arguments, raw, started))
        return {"pending_calls": len(ctx.prepared), "tool_calls": state.get("tool_calls", 0) + 1,
                "mcp_servers_used": [ctx.catalog[name].value]}

    def tool_result_guard(self, state, ctx):
        """Untrusted MCP output -> TOOL-namespace tokens; only then may it enter state."""
        name, arguments, raw, started = ctx.quarantine.pop(0)
        safe = self.tools.guard_result(name, raw, ctx.session, self.record(ctx), state["agent_steps"], started)
        ctx.batch.append((name, arguments, safe))
        update = {"tools_used": [name.value]}
        if ctx.prepared:
            return update
        calls, ctx.batch = ctx.batch, []
        update["messages"] = [assistant_tool_message([(n, a) for n, a, _ in calls]),
                              *(tool_message(n, s) for n, _, s in calls)]
        return update

    # ------------------------------------------------------------ metadata

    @staticmethod
    def _orchestration_summary(state) -> dict:
        route = state.get("route")
        summary = {"engine": "langgraph", "route": route,
                   "nodes": [*state.get("node_path", []), GraphNode.FINALIZE.value]}
        if route == Route.AGENT.value:
            summary.update(tool_protocol="mcp", mcp_transport=MCP_TRANSPORT,
                           mcp_servers=sorted(set(state.get("mcp_servers_used", []))),
                           approval=state.get("approval_status", ApprovalStatus.NOT_REQUIRED.value))
        return summary
