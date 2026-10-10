"""The single controlled path from the application to an LLM.

    API -> SecureLLMService (request facade) -> LangGraph runtime -> nodes
                                                                   |
        nodes -> SecureLLMService boundary -> PIISecurityGateway -> LLMClient

`SecureLLMService` is the only component that holds an `LLMClient` and the
only caller of `LLMClient.generate()` / `complete()` in the code base
(enforced by tests/test_architecture.py). The API layer receives this
service, never the LLM client, the graph internals or an MCP component.

Each request method builds one `RequestContext` and runs the compiled
LangGraph. LangGraph owns routing and state; the boundary methods below own
PII protection, leak verification, model invocation and re-identification.

Token-mapping lifecycle: one `TokenVault` per request inside its
`RequestContext` (never in graph state), cleared when the run ends whether it
succeeds or fails. A run paused for human approval keeps its context in a
bounded, expiring in-memory store until it is decided.
"""

from __future__ import annotations

import logging
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.audit.events import AuditEvent, AuditEventType, AuditStatus, GraphTraceEvent
from app.audit.logger import AuditLogger
from app.core.exceptions import (
    ErrorCode,
    LLMError,
    LLMResponseError,
    ReidentificationError,
    SecureAgentError,
)
from app.llm.base import LLMClient, LLMHealth, SupportsHealthCheck, SupportsToolCalling
from app.agent.tool_gateway import ToolGateway
from app.agent.schemas import Role
from app.graph.context import RequestContext
from app.graph.names import Route
from app.graph.nodes import SecureGraphNodes
from app.graph.runtime import ApprovalPending, SecureGraphRuntime, recursion_limit_for
from app.security.tool_results import ToolSecuritySession
from app.rag.prompt import build_rag_prompt, RAG_INSTRUCTIONS
from app.rag.chunking import Chunk
from app.security.documents import protect_retrieved
from app.security.tokenizer import TokenVault, TOKEN_PATTERN
from app.security.gateway import PIISecurityGateway, ProtectedPrompt
from app.security.reidentifier import ReidentificationDecision, ReidentificationPolicy

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful enterprise assistant. For privacy, personal data in the "
    "user's message has been replaced with placeholders such as <PERSON_1>, "
    "<EMAIL_1>, <PHONE_1> or <SSN_1>. When you refer to that data, copy the "
    "placeholders exactly as written, including the angle brackets. Never guess, "
    "invent, alter or explain the real values behind placeholders, and never "
    "create new placeholders."
)

_INTERNAL_MESSAGES = {
    Route.CHAT: "Unexpected processing failure",
    Route.RAG: "Secure RAG processing failed",
    Route.AGENT: "Secure agent processing failed",
}


class InternalProcessingError(SecureAgentError):
    """Unexpected failure inside the secure pipeline (details withheld)."""


@dataclass(frozen=True)
class ChatResult:
    """Outcome of a secure chat request. `response` may contain restored PII."""

    request_id: str
    response: str = field(repr=False)
    model: str
    pii_detected: bool
    entity_types: list[str]
    reidentification_decision: ReidentificationDecision
    latency_ms: float
    orchestration: dict | None = None


class SecureLLMService:
    """Request facade over the LangGraph runtime, and the sole LLM boundary."""

    def __init__(
        self,
        gateway: PIISecurityGateway,
        llm: LLMClient,
        audit: AuditLogger,
        reidentification_ceiling: Iterable[str],
        system_prompt: str = SYSTEM_PROMPT,
        retriever=None,
        rag_top_k: int = 3,
        tool_registry=None,
        max_agent_steps: int = 5,
        max_agent_tool_calls: int = 5,
        max_tool_result_chars: int = 16000,
        *,
        tool_gateway: ToolGateway | None = None,
        trace: AuditLogger | None = None,
        mcp_tool_timeout: float = 30.0,
        approval_ttl_seconds: int = 300,
        max_pending_approvals: int = 16,
    ) -> None:
        if tool_gateway is None and tool_registry is not None:
            tool_gateway = ToolGateway(tool_registry, timeout=mcp_tool_timeout, max_result_chars=max_tool_result_chars)
        self._tools = tool_gateway
        self._max_tool_result_chars = max_tool_result_chars
        self._retriever = retriever
        self._rag_top_k = rag_top_k
        self._gateway = gateway
        self._llm = llm
        self._audit = audit
        self._trace = trace or AuditLogger([])
        self._ceiling = frozenset(reidentification_ceiling)
        self._system_prompt = system_prompt
        nodes = SecureGraphNodes(self, self._tools, max_agent_steps, max_agent_tool_calls, self._trace_event)
        self._runtime = SecureGraphRuntime(
            nodes, recursion_limit=recursion_limit_for(max_agent_steps, max_agent_tool_calls),
            approval_ttl_seconds=approval_ttl_seconds, max_pending_approvals=max_pending_approvals)

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    @property
    def provider_name(self) -> str:
        return getattr(self._llm, "provider_name", "ollama")

    @property
    def tool_calling_supported(self) -> bool:
        return isinstance(self._llm, SupportsToolCalling) and getattr(self._llm, "tool_calling_supported", True)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def tool_gateway(self) -> ToolGateway | None:
        return self._tools

    @property
    def runtime(self) -> SecureGraphRuntime:
        return self._runtime

    def llm_health(self) -> LLMHealth:
        if isinstance(self._llm, SupportsHealthCheck):
            return self._llm.check_health()
        return LLMHealth.UNKNOWN

    def close(self) -> None:
        self._runtime.close()
        if self._retriever is not None:
            self._retriever.close()
        close = getattr(self._llm, "close", None)
        if callable(close):
            close()

    def effective_policy(self, requested_types: Iterable[str]) -> ReidentificationPolicy:
        """Caller request intersected with the server-side ceiling."""
        return ReidentificationPolicy(frozenset(requested_types) & self._ceiling)

    # ------------------------------------------------------------ request facades

    def chat(self, message: str, requested_types: Iterable[str], request_id: str,
             include_orchestration: bool = False) -> ChatResult:
        """Process one chat message through the secure graph.

        Raises:
            SecureAgentError: any failure, already mapped to a safe error code.
                Nothing reaches the LLM unless sanitization fully succeeded.
        """
        return self._runtime.run(RequestContext(request_id, Route.CHAT, tuple(requested_types), message=message,
                                                include_orchestration=include_orchestration))

    def rag_chat(self, message, requested_types, request_id, top_k=None, include_orchestration=False):
        return self._runtime.run(RequestContext(request_id, Route.RAG, tuple(requested_types), message=message,
                                                top_k=top_k, include_orchestration=include_orchestration))

    def agent_chat(self, message, requested_types, request_id, role=Role.CUSTOMER_SERVICE, include_sources=False,
                   include_orchestration=False):
        # Role is application-controlled and validated in the graph; no model field can affect it.
        result = self._runtime.run(RequestContext(request_id, Route.AGENT, tuple(requested_types), role=role,
                                                  message=message, include_sources=include_sources,
                                                  include_orchestration=include_orchestration))
        return self._pending(result) if isinstance(result, ApprovalPending) else result

    def decide_approval(self, approval_id: str, decision: str, approver_role) -> dict:
        """Resume a run paused for human approval (demonstration approval only)."""
        result = self._runtime.resume(approval_id, decision, approver_role)
        return self._pending(result) if isinstance(result, ApprovalPending) else result

    def _pending(self, pending: ApprovalPending) -> dict:
        return {"request_id": pending.request_id, "status": "APPROVAL_REQUIRED", "model": self.model_name,
                "approval": {"approval_id": pending.approval_id, "tool": pending.tool, "record_id": pending.record_id,
                             "reason": "RESTRICTED_RECORD", "approver_role": Role.SUPERVISOR.value,
                             "expires_in_seconds": pending.expires_in_seconds}}

    # ------------------------------------------------------------ trusted boundary (graph nodes)

    def internal_error(self, mode: Route) -> InternalProcessingError:
        return InternalProcessingError(_INTERNAL_MESSAGES.get(mode, "Unexpected processing failure"))

    def new_tool_session(self, ctx: RequestContext) -> ToolSecuritySession:
        return ToolSecuritySession(ctx.request_id, self._gateway, self._max_tool_result_chars)

    def protect_input(self, ctx: RequestContext) -> ProtectedPrompt:
        """Normalize, detect, tokenize, verify (raises -> fail closed)."""
        ctx.protected = self._gateway.protect(ctx.message, request_id=ctx.request_id, vault=ctx.vault)
        ctx.message = ""  # the raw input is no longer needed
        return ctx.protected

    def record_chat_sanitized(self, ctx: RequestContext) -> None:
        self._record(AuditEventType.PII_SANITIZED, AuditStatus.SUCCESS, ctx.request_id, ctx.protected)

    def generate_chat(self, ctx: RequestContext) -> str:
        # Defence in depth: re-verify the exact payload at the LLM boundary.
        prompt = ctx.protected.sanitized_text
        self._gateway.verify_no_leak(prompt, ctx.protected.vault)
        # Only sanitized text crosses the boundary.
        return self._invoke_llm(prompt, ctx.request_id, ctx.protected)

    def generate_rag(self, ctx: RequestContext) -> str:
        """Phase 3 secure retrieval, document protection and generation."""
        protected, request_id = ctx.protected, ctx.request_id
        if self._retriever is None:
            from app.rag.retriever import RetrievalError
            raise RetrievalError("Retrieval unavailable")
        chosen_k = self._rag_top_k if ctx.top_k is None else ctx.top_k
        if type(chosen_k) is not int or not 1 <= chosen_k <= 10:
            raise ValueError("Invalid top_k")
        hits = self._retriever.retrieve(protected.sanitized_text, chosen_k)
        if len(hits) > chosen_k:
            raise ValueError("Too many retrieved chunks")
        ctx.hits = list(hits)
        contexts, sources = [], []
        for hit in hits:
            chunk = hit.chunk
            # Revalidate untrusted persisted metadata before API or prompt use.
            Chunk(chunk.document_id, chunk.filename, chunk.chunk_id, chunk.text)
            document_vault = ctx.document_vaults.setdefault(chunk.document_id, TokenVault())
            safe, _ = protect_retrieved(chunk.text, self._gateway, chunk.document_id, request_id, document_vault)
            labels = {"PERSON":"PERSON", "EMAIL":"EMAIL_ADDRESS", "PHONE":"PHONE_NUMBER", "SSN":"US_SSN"}
            for match in TOKEN_PATTERN.finditer(safe):
                entity = labels.get(match.group(1).removeprefix("DOC_"))
                if entity:
                    ctx.document_counts[entity] = ctx.document_counts.get(entity, 0) + 1
            contexts.append({"source": chunk.filename, "chunk_id": chunk.chunk_id, "text": safe})
            sources.append({"document_id":chunk.document_id,"filename":chunk.filename,
                            "chunk_id":chunk.chunk_id,"score":hit.score})
        prompt = build_rag_prompt(protected.sanitized_text, contexts)
        self._gateway.verify_no_leak(prompt, ctx.vault)
        for document_vault in ctx.document_vaults.values():
            self._gateway.verify_no_leak(prompt, document_vault)
        ctx.sources = sources
        return self._invoke_llm(prompt, request_id, protected, system=self._system_prompt + RAG_INSTRUCTIONS)

    def agent_recorder(self, ctx: RequestContext):
        protected, role = ctx.protected, ctx.role

        def record(event, status, tool=None, step=0, authorized=None, counts=None, latency_ms=None, error=None,
                   decision=None, server=None):
            counts = counts or {}
            types = set(protected.entity_types) | set(counts)
            self._audit.record(AuditEvent(request_id=ctx.request_id, event=event, status=status, model=self.model_name,
                provider=self.provider_name, role=role, tool_name=tool, authorized=authorized, agent_step=step,
                pii_types=tuple(sorted(types)), pii_count=protected.total_entities + sum(counts.values()),
                latency_ms=latency_ms, error_category=error, reidentification=decision, mcp_server=server))
        return record

    def complete_agent(self, ctx: RequestContext, messages: list[dict], tools: list[dict]):
        """One native structured turn. Every message is re-verified first."""
        session = ctx.session
        for msg in messages:
            text = msg.get('content', '')
            self._gateway.verify_no_leak(text, ctx.vault)
            session.verify(text)
            for call in msg.get('tool_calls', []):
                arguments = json.dumps(call['function']['arguments'], ensure_ascii=False)
                self._gateway.verify_no_leak(arguments, ctx.vault)
                session.verify(arguments)
        try:
            return self._llm.complete(messages, tools)
        except SecureAgentError:
            raise
        except Exception:
            raise LLMResponseError("Structured model invocation failed") from None

    def verify_agent_answer(self, ctx: RequestContext, text: str) -> None:
        # Fail closed if a model guesses or reconstructs a known protected value.
        ctx.session.verify(text)
        self._gateway.verify_no_leak(text, ctx.vault)

    def restore(self, ctx: RequestContext, text: str):
        """Controlled re-identification in the security layer (user vault only)."""
        try:
            return self._gateway.restore(text, ctx.protected, self.effective_policy(ctx.requested_types))
        except SecureAgentError:
            raise
        except Exception:
            raise ReidentificationError("Re-identification failed") from None

    def complete_request(self, ctx: RequestContext, state: dict, orchestration: dict | None):
        """Success audit and the mode's public result (the only place restored text is placed)."""
        protected, restored = ctx.protected, ctx.restored
        latency_ms = (time.perf_counter() - ctx.started) * 1000
        extra = {"orchestration": orchestration} if orchestration else {}
        if ctx.mode is Route.CHAT:
            self._record(AuditEventType.REQUEST_COMPLETED, AuditStatus.SUCCESS, ctx.request_id, protected,
                         reidentification=restored.decision, latency_ms=latency_ms)
            return ChatResult(request_id=ctx.request_id, response=restored.text, model=self.model_name,
                              pii_detected=protected.total_entities > 0, entity_types=protected.entity_types,
                              reidentification_decision=restored.decision, latency_ms=latency_ms,
                              orchestration=orchestration)
        if ctx.mode is Route.RAG:
            self._record_rag(ctx.request_id, protected, ctx.document_counts, ctx.hits, ctx.started, restored.decision)
            return {"request_id":ctx.request_id, "response":restored.text, "sources":list(ctx.sources),
                    "security":{"pii_detected":protected.total_entities > 0,
                    "document_pii_detected":bool(ctx.document_counts),
                    "entity_types":sorted(set(protected.entity_types) | set(ctx.document_counts)),
                    "reidentification_decision":restored.decision.value}, "model":self.model_name, **extra}
        session = ctx.session
        self.agent_recorder(ctx)(AuditEventType.AGENT_COMPLETED, AuditStatus.SUCCESS, step=state["agent_steps"],
                                 counts=session.pii_counts, latency_ms=latency_ms, decision=restored.decision)
        return {"request_id":ctx.request_id,"response":restored.text,"tools_used":list(state.get("tools_used", [])),
            "agent_steps":state["agent_steps"],**({"sources":list(session.sources)} if ctx.include_sources else {}),
            "security":{"pii_detected":protected.total_entities>0,
            "tool_pii_detected":session.tool_pii_detected,"document_pii_detected":session.document_pii_detected,
            "entity_types":sorted(set(protected.entity_types)|set(session.pii_counts)),
            "reidentification_decision":restored.decision.value},"model":self.model_name, **extra}

    def record_failure(self, ctx: RequestContext) -> None:
        """Mode-specific failure audit; error codes only, never exception text."""
        code = ctx.error.error_code if isinstance(ctx.error, SecureAgentError) else ErrorCode.INTERNAL_ERROR
        if ctx.mode is Route.CHAT:
            self._record_failure(ctx.request_id, ctx.protected, code, ctx.started)
        elif ctx.mode is Route.RAG:
            self._record_rag(ctx.request_id, ctx.protected, ctx.document_counts, ctx.hits, ctx.started, error=code)
        else:
            self._audit.record(AuditEvent(request_id=ctx.request_id, event=AuditEventType.AGENT_FAILED,
                status=AuditStatus.FAILURE, model=self.model_name, provider=self.provider_name,
                role=ctx.role if isinstance(ctx.role, Role) else None, error_category=code,
                latency_ms=(time.perf_counter() - ctx.started) * 1000))

    # ------------------------------------------------------------------ helpers

    def _trace_event(self, ctx: RequestContext, event, status, **fields) -> None:
        try:
            self._trace.record(GraphTraceEvent(request_id=ctx.request_id, event=event, status=status,
                                               provider=self.provider_name, model=self.model_name, **fields))
        except ValueError:
            logger.warning("request_id=%s graph trace event rejected", ctx.request_id)

    def _invoke_llm(self, prompt: str, request_id: str, protected: ProtectedPrompt, *, system: str | None = None) -> str:
        started = time.perf_counter()
        try:
            output = self._llm.generate(prompt, system=system or self._system_prompt)
            if not isinstance(output, str):
                raise LLMResponseError("LLM returned a non-text response")
        except LLMError as exc:
            self._record(
                AuditEventType.LLM_REQUEST_FAILED, AuditStatus.FAILURE, request_id, protected,
                latency_ms=(time.perf_counter() - started) * 1000, error_category=exc.error_code,
            )
            raise
        except Exception:
            self._record(
                AuditEventType.LLM_REQUEST_FAILED, AuditStatus.FAILURE, request_id, protected,
                latency_ms=(time.perf_counter() - started) * 1000,
                error_category=ErrorCode.LLM_BAD_RESPONSE,
            )
            raise LLMResponseError("LLM invocation failed") from None

        self._record(
            AuditEventType.LLM_REQUEST_COMPLETED, AuditStatus.SUCCESS, request_id, protected,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return output

    def _record_rag(self, request_id, protected, counts, hits, started, decision=None, error=None):
        self._audit.record(AuditEvent(
            request_id=request_id, event=AuditEventType.REQUEST_FAILED if error else AuditEventType.REQUEST_COMPLETED,
            status=AuditStatus.FAILURE if error else AuditStatus.SUCCESS, model=self.model_name, provider=self.provider_name,
            pii_types=tuple(sorted(set(protected.entity_types if protected else []) | set(counts))),
            pii_count=(protected.total_entities if protected else 0) + sum(counts.values()),
            reidentification=decision, latency_ms=(time.perf_counter()-started)*1000, error_category=error,
            retrieval_enabled=True, documents_retrieved=len({h.chunk.document_id for h in hits}),
            chunk_count=len(hits), embedding_model=self._retriever.embedding_model if self._retriever else None,
        ))

    def _record(
        self,
        event: AuditEventType,
        status: AuditStatus,
        request_id: str,
        protected: ProtectedPrompt | None,
        *,
        reidentification: ReidentificationDecision | None = None,
        latency_ms: float | None = None,
        error_category: ErrorCode | None = None,
    ) -> None:
        self._audit.record(
            AuditEvent(
                request_id=request_id,
                event=event,
                status=status,
                model=self.model_name, provider=self.provider_name,
                pii_types=tuple(protected.entity_types) if protected else (),
                pii_count=protected.total_entities if protected else 0,
                reidentification=reidentification,
                latency_ms=latency_ms,
                error_category=error_category,
            )
        )

    def _record_failure(
        self,
        request_id: str,
        protected: ProtectedPrompt | None,
        error_code: ErrorCode,
        started: float,
    ) -> None:
        self._record(
            AuditEventType.REQUEST_FAILED, AuditStatus.FAILURE, request_id, protected,
            latency_ms=(time.perf_counter() - started) * 1000, error_category=error_code,
        )
        logger.warning("request_id=%s failed error_code=%s", request_id, error_code.value)
