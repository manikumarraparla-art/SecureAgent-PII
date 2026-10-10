"""Executes the compiled secure graph for one request at a time.

- Each run gets a fresh thread id; its checkpoint is deleted when the run
  ends, so no graph state outlives the request (no cross-request leakage).
- The checkpointer is in memory only and stores sanitized state. It exists for
  one reason: a human-approval pause must be resumable. Nothing is written to disk.
- Checkpoints are written only when a run exits or pauses (durability="exit").
- LangGraph's recursion limit bounds total node executions in addition to the
  agent's own step and tool-call budgets.
- LangSmith tracing is forced off for every invocation.
"""
from collections import OrderedDict
from dataclasses import dataclass
import threading
import time
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from langsmith.run_helpers import tracing_context

from app.agent.approval import may_approve
from app.agent.errors import AgentLimitError, ApprovalNotFoundError, ApproverNotPermittedError
from app.audit.events import AuditEventType, AuditStatus, GraphTraceType
from app.graph.builder import build_secure_graph
from app.graph.names import GraphNode, Route
from app.graph.state import initial_state


def recursion_limit_for(max_steps: int, max_calls: int) -> int:
    """input_guard, router, discovery, output_guard, finalize (+ slack), and per
    agent step: planner, policy gate, approval, then mcp_tool + guard per call."""
    return 10 + max_steps * (3 + 2 * max_calls)


@dataclass(frozen=True)
class ApprovalPending:
    request_id: str
    approval_id: str
    tool: str
    record_id: str | None
    expires_in_seconds: int


@dataclass
class _Paused:
    thread_id: str
    context: object
    expires_at: float


class ApprovalStore:
    """Bounded, expiring, single-use store of paused runs (memory only).

    A paused run keeps its request context (and so its token vaults) until it
    is decided, expires or is evicted; then the context is cleared and its
    checkpoint deleted.
    """

    def __init__(self, ttl_seconds: int, capacity: int, discard):
        self._ttl, self._capacity, self._discard = ttl_seconds, capacity, discard
        self._items: OrderedDict[str, _Paused] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, approval_id: str, paused: _Paused) -> None:
        with self._lock:
            dropped = self._expire()
            while len(self._items) >= self._capacity:
                dropped.append(self._items.popitem(last=False)[1])
            self._items[approval_id] = paused
        for item in dropped:
            self._discard(item)

    def get(self, approval_id: str) -> _Paused | None:
        with self._lock:
            dropped = self._expire()
            paused = self._items.get(approval_id)
        for item in dropped:
            self._discard(item)
        return paused

    def take(self, approval_id: str) -> _Paused | None:
        with self._lock:
            dropped = self._expire()
            paused = self._items.pop(approval_id, None)
        for item in dropped:
            self._discard(item)
        return paused

    def purge(self, everything: bool = False) -> None:
        with self._lock:
            dropped = list(self._items.values()) if everything else self._expire()
            if everything:
                self._items.clear()
        for item in dropped:
            self._discard(item)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def _expire(self) -> list[_Paused]:
        now = time.monotonic()
        expired = [key for key, item in self._items.items() if item.expires_at <= now]
        return [self._items.pop(key) for key in expired]


class SecureGraphRuntime:
    def __init__(self, nodes, *, recursion_limit: int, approval_ttl_seconds: int = 300, max_pending_approvals: int = 16):
        self._nodes = nodes
        self._saver = InMemorySaver()
        self.graph = build_secure_graph(nodes).compile(checkpointer=self._saver)
        self.recursion_limit = recursion_limit
        self._ttl = approval_ttl_seconds
        self.approvals = ApprovalStore(approval_ttl_seconds, max_pending_approvals, self._discard)

    @property
    def checkpointer(self) -> InMemorySaver:
        return self._saver

    def run(self, ctx):
        self.approvals.purge()  # expired pauses lose their vaults promptly
        self._nodes.trace(ctx, GraphTraceType.GRAPH_STARTED, AuditStatus.SUCCESS, route=ctx.mode)
        return self._execute(initial_state(ctx.request_id, ctx.mode.value), ctx, uuid.uuid4().hex)

    def resume(self, approval_id: str, decision: str, approver_role):
        # Check the approver before consuming the approval, so a wrong role cannot burn it.
        if self.approvals.get(approval_id) is None:
            raise ApprovalNotFoundError("Unknown approval")
        if not may_approve(approver_role):
            raise ApproverNotPermittedError("Approver role not permitted")
        paused = self.approvals.take(approval_id)  # single use
        if paused is None:
            raise ApprovalNotFoundError("Unknown approval")
        ctx = paused.context
        self._nodes.trace(ctx, GraphTraceType.GRAPH_RESUMED, AuditStatus.SUCCESS, node=GraphNode.HUMAN_APPROVAL)
        command = Command(resume={"decision": decision, "approver_role": approver_role.value})
        return self._execute(command, ctx, paused.thread_id)

    def close(self) -> None:
        """Destroy every paused run (vaults and checkpoints) at shutdown."""
        self.approvals.purge(everything=True)

    def _execute(self, graph_input, ctx, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": self.recursion_limit}
        paused = False
        try:
            with tracing_context(enabled=False):
                try:
                    output = self.graph.invoke(graph_input, config, context=ctx, durability="exit")
                except GraphRecursionError:
                    output = self._limit_reached(ctx)
                except Exception:
                    # Defensive: node failures are already routed to safe_error.
                    ctx.error = self._nodes.boundary.internal_error(ctx.mode)
                    self._nodes.boundary.record_failure(ctx)
                    output = {}
            if output.get("__interrupt__") and ctx.error is None and ctx.approval_id:
                paused = True
                return self._pause(ctx, thread_id)
            if ctx.error is None and ctx.result is None:
                ctx.error = self._nodes.boundary.internal_error(ctx.mode)
                self._nodes.boundary.record_failure(ctx)
            failed = ctx.error is not None
            self._nodes.trace(ctx, GraphTraceType.GRAPH_COMPLETED, AuditStatus.FAILURE if failed else AuditStatus.SUCCESS,
                              route=ctx.mode, latency_ms=(time.perf_counter() - ctx.started) * 1000,
                              error_category=ctx.error.error_code if failed else None)
            if failed:
                raise ctx.error
            return ctx.result
        finally:
            if not paused:
                self._saver.delete_thread(thread_id)
                ctx.clear()

    def _limit_reached(self, ctx) -> dict:
        ctx.error = AgentLimitError("Graph execution limit reached")
        self._nodes.trace(ctx, GraphTraceType.GRAPH_LIMIT_REACHED, AuditStatus.FAILURE,
                          error_category=ctx.error.error_code)
        if ctx.mode is Route.AGENT and ctx.protected is not None:
            self._nodes.record(ctx)(AuditEventType.AGENT_STEP_LIMIT_REACHED, AuditStatus.FAILURE)
        self._nodes.boundary.record_failure(ctx)
        return {}

    def _pause(self, ctx, thread_id: str) -> ApprovalPending:
        self.approvals.add(ctx.approval_id, _Paused(thread_id, ctx, time.monotonic() + self._ttl))
        self._nodes.trace(ctx, GraphTraceType.GRAPH_INTERRUPTED, AuditStatus.SUCCESS, node=GraphNode.HUMAN_APPROVAL)
        request = ctx.approval_request or {}
        return ApprovalPending(ctx.request_id, ctx.approval_id, request.get("tool", ""), request.get("record_id"), self._ttl)

    def _discard(self, paused: _Paused) -> None:
        paused.context.clear()
        self._saver.delete_thread(paused.thread_id)
