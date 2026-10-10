"""Only fixed local FastAPI endpoints. No direct AI or private-module access."""
from dataclasses import dataclass
from enum import Enum
import math
import re
import time
from urllib.parse import urlsplit
import httpx
from pydantic import ValidationError
from secureagent_ui.contracts import (APIResponse, ApprovalResponse, HealthResponse, Mode, ENDPOINTS, ROLES,
                                     RESTORABLE_TYPES, APPROVALS_PATH, APPROVAL_ID_PATTERN)

class FailureCode(str,Enum):
    CONFIGURATION = "CONFIGURATION"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    INVALID_REQUEST = "INVALID_REQUEST"
    BACKEND_INCOMPATIBLE = "BACKEND_INCOMPATIBLE"
    TOOL_DENIED = "TOOL_DENIED"
    RAG_UNAVAILABLE = "RAG_UNAVAILABLE"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    CAPABILITY_UNSUPPORTED = "CAPABILITY_UNSUPPORTED"
    LIMIT_REACHED = "LIMIT_REACHED"
    SERVER_ERROR = "SERVER_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    TOOLS_UNAVAILABLE = "TOOLS_UNAVAILABLE"
    APPROVAL_DENIED = "APPROVAL_DENIED"
    APPROVAL_EXPIRED = "APPROVAL_EXPIRED"

MESSAGES = {
    FailureCode.CONFIGURATION:"UI configuration must use a local FastAPI URL and valid timeouts.",
    FailureCode.BACKEND_UNAVAILABLE:"Backend unavailable. Start it with: uvicorn app.main:app --host 127.0.0.1 --port 8000",
    FailureCode.TIMEOUT:"The local request timed out. Try again or reduce the request size.",
    FailureCode.INVALID_REQUEST:"Check your message and selected request options.",
    FailureCode.BACKEND_INCOMPATIBLE:"The running backend does not support Phase 8 requests. Restart FastAPI from the current SecureAgent-PII checkout with: python -m uvicorn app.main:app --host 127.0.0.1 --port 8000",
    FailureCode.TOOL_DENIED:"Tool access denied for the selected demo role or requested record scope.",
    FailureCode.RAG_UNAVAILABLE:"Local knowledge search is unavailable. Rebuild the index with: python -m app.rag",
    FailureCode.LLM_UNAVAILABLE:"The configured local model is unavailable. Check the backend provider and loaded model.",
    FailureCode.CAPABILITY_UNSUPPORTED:"The configured model does not support Secure Agent tools. Use Secure Chat or Secure RAG.",
    FailureCode.LIMIT_REACHED:"The agent reached its execution limit. Try a more specific question.",
    FailureCode.SERVER_ERROR:"The request could not be completed securely. Please try again.",
    FailureCode.INVALID_RESPONSE:"The backend returned an unexpected response. No internal details are displayed.",
    FailureCode.TOOLS_UNAVAILABLE:"The local MCP tool servers are unavailable. Try again shortly.",
    FailureCode.APPROVAL_DENIED:"The supervisor did not approve this tool action. No record was retrieved.",
    FailureCode.APPROVAL_EXPIRED:"This approval request expired or was already decided. Ask the question again.",
}

class UIFailure(Exception):
    def __init__(self,code:FailureCode):
        self.code=code
        super().__init__(MESSAGES[code])


def local_backend_url(raw:str) -> str:
    try:
        if not isinstance(raw,str) or any(c.isspace() for c in raw) or '%' in raw or '\\' in raw:
            raise ValueError("Invalid URL")
        url=urlsplit(raw)
        if url.scheme not in ('http','https') or url.hostname not in ('localhost','127.0.0.1','::1'):
            raise ValueError("Not loopback")
        if url.username is not None or url.password is not None or url.path not in ('','/') or url.query or url.fragment:
            raise ValueError("URL contains unsupported components")
        if url.port is not None and (not 1<=url.port<=65535 or url.port==11434):
            raise ValueError("Not an API port")
        return raw.rstrip('/')
    except (ValueError,TypeError):
        raise UIFailure(FailureCode.CONFIGURATION) from None


def safe_error(status:int,body) -> FailureCode:
    # Inspect fixed codes and schema metadata only. Never show backend messages/body.
    code=body.get('error',{}).get('code') if isinstance(body,dict) and isinstance(body.get('error'),dict) else None
    # Match only fixed schema metadata; never display input, message or arbitrary locations.
    fields = body['error'].get('fields') if code == 'INVALID_REQUEST' else None
    if status == 422 and isinstance(fields, list) and any(
        isinstance(field, dict) and field.get('loc') == ['body', 'include_orchestration']
        and field.get('type') == 'extra_forbidden' for field in fields
    ):
        return FailureCode.BACKEND_INCOMPATIBLE
    if code=='TOOL_DENIED': return FailureCode.TOOL_DENIED
    if code in ('EMBEDDING_FAILED','VECTOR_STORE_FAILED','RETRIEVAL_FAILED'): return FailureCode.RAG_UNAVAILABLE
    if code=='LLM_TIMEOUT' or status==504: return FailureCode.TIMEOUT
    if code in ('LLM_UNAVAILABLE','LLM_MODEL_UNAVAILABLE'): return FailureCode.LLM_UNAVAILABLE
    if code=='LLM_CAPABILITY_UNSUPPORTED': return FailureCode.CAPABILITY_UNSUPPORTED
    if code=='AGENT_LIMIT_REACHED': return FailureCode.LIMIT_REACHED
    if code=='MCP_UNAVAILABLE': return FailureCode.TOOLS_UNAVAILABLE
    if code=='APPROVAL_DENIED': return FailureCode.APPROVAL_DENIED
    if code=='APPROVAL_NOT_FOUND': return FailureCode.APPROVAL_EXPIRED
    if status in (400,422): return FailureCode.INVALID_REQUEST
    return FailureCode.SERVER_ERROR

@dataclass(frozen=True)
class ReceivedResponse:
    result: APIResponse
    latency_ms: float

@dataclass(frozen=True)
class PendingApproval:
    """A paused agent run awaiting the demonstration supervisor decision."""
    approval: ApprovalResponse

@dataclass(frozen=True)
class BackendHealth:
    backend_available: bool
    ollama_available: bool
    model: str = "qwen2.5:7b"
    provider: str = "ollama"
    tool_calling: bool = True
    version: str = "unknown"

    @property
    def llm_available(self):
        return self.ollama_available

class APIClient:
    def __init__(self,base_url='http://127.0.0.1:8000',connect_timeout=5.0,response_timeout=180.0,transport=None):
        base_url=local_backend_url(base_url)
        if any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in (connect_timeout,response_timeout)):
            raise UIFailure(FailureCode.CONFIGURATION)
        if connect_timeout>30 or response_timeout>600: raise UIFailure(FailureCode.CONFIGURATION)
        self._client=httpx.Client(base_url=base_url,trust_env=False,follow_redirects=False,
            timeout=httpx.Timeout(response_timeout,connect=connect_timeout),transport=transport)
        self._health_timeout=connect_timeout

    def close(self): self._client.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()

    def health(self) -> BackendHealth:
        try:
            response=self._client.get('/health',timeout=self._health_timeout)
            if response.status_code not in (200,503) or len(response.content)>4096:
                return BackendHealth(False,False)
            health=HealthResponse.model_validate(response.json())
            return BackendHealth(True,(health.llm or health.ollama)=='available',health.model,health.provider,health.tool_calling,health.version)
        except Exception:
            return BackendHealth(False,False)

    def send(self,mode,message,role='customer_service',allowed_types=(),top_k=3) -> ReceivedResponse:
        try:
            mode=Mode(mode)
            if not isinstance(message,str) or not message.strip() or len(message)>8000:
                raise ValueError("Invalid message")
            if not isinstance(allowed_types,(list,tuple)) or len(allowed_types)>3 or not set(allowed_types)<=set(RESTORABLE_TYPES):
                raise ValueError("Invalid restoration options")
            payload={'message':message,'allowed_reidentification_types':list(allowed_types),
                     'include_orchestration':True}
            if mode is Mode.AGENT:
                if role not in ROLES: raise ValueError("Invalid role")
                payload.update(role=role,include_sources=True)
            if mode is Mode.RAG:
                if type(top_k) is not int or not 1<=top_k<=10: raise ValueError("Invalid top_k")
                payload['top_k']=top_k
        except (ValueError,TypeError):
            raise UIFailure(FailureCode.INVALID_REQUEST) from None
        return self._post(ENDPOINTS[mode],payload,mode)

    def decide(self,approval_id,decision):
        """Demonstration approval: always acts as the supervisor demo role."""
        try:
            if not isinstance(approval_id,str) or not re.fullmatch(APPROVAL_ID_PATTERN,approval_id):
                raise ValueError("Invalid approval id")
            if decision not in ('approve','deny'): raise ValueError("Invalid decision")
        except (ValueError,TypeError):
            raise UIFailure(FailureCode.INVALID_REQUEST) from None
        return self._post(APPROVALS_PATH+approval_id,{'decision':decision,'approver_role':'supervisor'},Mode.AGENT)

    def _post(self,path,payload,mode):
        started=time.perf_counter()
        try:
            response=self._client.post(path,json=payload)
        except httpx.TimeoutException:
            raise UIFailure(FailureCode.TIMEOUT) from None
        except httpx.HTTPError:
            raise UIFailure(FailureCode.BACKEND_UNAVAILABLE) from None
        if len(response.content)>1_000_000:
            raise UIFailure(FailureCode.INVALID_RESPONSE)
        try: body=response.json()
        except (ValueError,TypeError): body=None
        if mode is Mode.AGENT and response.status_code==202:
            try: return PendingApproval(ApprovalResponse.model_validate(body))
            except (ValidationError,ValueError,TypeError):
                raise UIFailure(FailureCode.INVALID_RESPONSE) from None
        if response.status_code!=200:
            raise UIFailure(safe_error(response.status_code,body)) from None
        try:
            result=APIResponse.model_validate(body)
            if mode is Mode.RAG and 'sources' not in body: raise ValueError("Missing RAG metadata")
            if mode is Mode.AGENT and ('tools_used' not in body or 'agent_steps' not in body):
                raise ValueError("Missing agent metadata")
            for source in result.sources:
                if not source.chunk_id.startswith(source.document_id+'-'): raise ValueError("Invalid source relation")
            return ReceivedResponse(result,(time.perf_counter()-started)*1000)
        except (ValidationError,ValueError,TypeError,AttributeError):
            raise UIFailure(FailureCode.INVALID_RESPONSE) from None
