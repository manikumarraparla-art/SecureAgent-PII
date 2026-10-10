"""FastAPI application factory and composition root.

This module is the only place where the LLM client is constructed; it is
handed straight to `SecureLLMService`, which is the only object the API
layer can reach. The service owns the compiled LangGraph runtime, and its
ToolGateway owns the in-process MCP servers built from the tool registry.

Run locally:
    uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from app.agent.tool_registry import build_registry
from app.api.errors import register_error_handlers
from app.api.routes import router
from app.version import VERSION
from app.audit.logger import AuditLogger, JsonlAuditSink
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.llm.factory import create_llm_client
from app.rag.embeddings import OllamaEmbeddingClient
from app.rag.vector_store import SQLiteVectorStore
from app.rag.retriever import Retriever
from app.security.gateway import PIISecurityGateway
from app.security.pii_detector import PIIDetector
from app.security.secure_llm import SecureLLMService


def build_service(settings: Settings) -> SecureLLMService:
    """Wire the production pipeline from settings."""
    gateway=PIISecurityGateway(PIIDetector.from_settings(settings))
    retriever=Retriever(OllamaEmbeddingClient(settings.ollama_base_url,settings.ollama_embed_model,
                        settings.ollama_request_timeout),
                        SQLiteVectorStore(settings.rag_index_path,settings.rag_collection,settings.ollama_embed_model))
    return SecureLLMService(
        gateway=gateway,
        llm=create_llm_client(settings),
        audit=AuditLogger([JsonlAuditSink(settings.audit_log_path)]),
        reidentification_ceiling=settings.reidentification_ceiling,
        retriever=retriever,
        rag_top_k=settings.rag_top_k,
        tool_registry=build_registry(retriever,gateway,settings.rag_top_k),
        max_agent_steps=settings.max_agent_steps,
        max_agent_tool_calls=settings.max_agent_tool_calls,
        max_tool_result_chars=settings.max_tool_result_chars,
        trace=AuditLogger([JsonlAuditSink(settings.graph_trace_log_path)]),
        mcp_tool_timeout=settings.mcp_tool_timeout,
        approval_ttl_seconds=settings.approval_ttl_seconds,
        max_pending_approvals=settings.max_pending_approvals,
    )


def create_app(settings: Settings | None = None, service: SecureLLMService | None = None) -> FastAPI:
    """Create the app. Settings are validated here, so bad config fails at startup."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Loads the spaCy model once at startup rather than on first request.
        app.state.service = service or build_service(settings)
        try:
            yield
        finally:
            app.state.service.close()

    app = FastAPI(title="SecureAgent-PII", version=VERSION, lifespan=lifespan)

    @app.middleware("http")
    async def assign_request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Always server-generated: a client-supplied ID could smuggle PII into audit logs.
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    register_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()
