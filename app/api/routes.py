"""HTTP routes.

Routes depend on `SecureLLMService` only. They never receive or construct an
LLM client, so there is no API -> LLM path that bypasses the gateway.
"""

from __future__ import annotations

from fastapi import APIRouter, Path, Request
from fastapi.responses import JSONResponse

from app.llm.base import LLMHealth
from app.models.chat import ChatRequest, ChatResponse, HealthResponse, SecuritySummary
from app.security.secure_llm import SecureLLMService

router = APIRouter()


def _service(request: Request) -> SecureLLMService:
    return request.app.state.service


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> JSONResponse:
    """Service health: selected local provider and configured model only."""
    service = _service(request)
    ollama = service.llm_health()
    healthy = ollama is LLMHealth.AVAILABLE
    body = HealthResponse(
        status="healthy" if healthy else "degraded",
        ollama=ollama.value if service.provider_name == "ollama" else "unknown",
        provider=service.provider_name,
        llm=ollama.value,
        tool_calling=service.tool_calling_supported,
        model=service.model_name,
    )
    return JSONResponse(status_code=200 if healthy else 503, content=body.model_dump())


# Sync handler: FastAPI runs it in a worker thread, so each request has its
# own call stack and its own request-scoped token vault.
@router.post("/api/v1/chat", response_model=ChatResponse, response_model_exclude_none=True)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Secure chat: PII is tokenized before the LLM and restored per policy after."""
    result = _service(request).chat(
        message=body.message,
        requested_types=body.allowed_reidentification_types,
        request_id=request.state.request_id,
        include_orchestration=body.include_orchestration,
    )
    return ChatResponse(
        request_id=result.request_id,
        response=result.response,
        security=SecuritySummary(
            pii_detected=result.pii_detected,
            entity_types=result.entity_types,
            reidentification_decision=result.reidentification_decision.value,
        ),
        model=result.model,
        orchestration=result.orchestration,
    )


from app.models.chat import RAGChatRequest, RAGChatResponse

@router.post("/api/v1/rag/chat", response_model=RAGChatResponse, response_model_exclude_none=True)
def rag_chat(body: RAGChatRequest, request: Request) -> RAGChatResponse:
    result = _service(request).rag_chat(body.message, body.allowed_reidentification_types,
                                       request.state.request_id, body.top_k,
                                       include_orchestration=body.include_orchestration)
    return RAGChatResponse(**result)


from app.models.chat import (AgentApprovalResponse, AgentChatRequest, AgentChatResponse,
                             ApprovalDecisionRequest)


def _agent_result(result: dict):
    # A run paused for human approval is accepted, not completed: 202 and no answer.
    if result.get("status") == "APPROVAL_REQUIRED":
        body = AgentApprovalResponse(**result)
        return JSONResponse(status_code=202, content=body.model_dump(mode="json", exclude_none=True))
    return AgentChatResponse(**result)


@router.post("/api/v1/agent/chat",response_model=AgentChatResponse,response_model_exclude_none=True,
             responses={202: {"model": AgentApprovalResponse}})
def agent_chat(body: AgentChatRequest, request: Request) -> AgentChatResponse:
    result=_service(request).agent_chat(body.message,body.allowed_reidentification_types,
                                      request.state.request_id,body.role,include_sources=body.include_sources,
                                      include_orchestration=body.include_orchestration)
    return _agent_result(result)


@router.post("/api/v1/agent/approvals/{approval_id}",response_model=AgentChatResponse,
             response_model_exclude_none=True,responses={202: {"model": AgentApprovalResponse}})
def decide_approval(body: ApprovalDecisionRequest, request: Request,
                    approval_id: str = Path(pattern=r"^[0-9a-f]{32}$")) -> AgentChatResponse:
    """Local demonstration of human approval; the role is a demo label, not an identity."""
    return _agent_result(_service(request).decide_approval(approval_id, body.decision, body.approver_role))
