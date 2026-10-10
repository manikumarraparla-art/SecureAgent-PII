"""Tool policy enforcement point: authorize -> validate -> scope -> approval -> MCP -> sanitize.

Every decision here is deterministic application policy; the model only
proposes calls. Execution is delegated to the MCP Tool Gateway, and every
result passes `guard_result` before it can reach a model.
"""
import json
import time
from pydantic import ValidationError
from app.agent.approval import requires_approval
from app.agent.planner import call_key
from app.agent.authorization import permits
from app.agent.schemas import Role, ToolName, SCHEMAS
from app.agent.errors import UnknownToolError, ToolDeniedError, ToolArgumentError, ToolExecutionError
from app.audit.events import AuditEventType, AuditStatus
from app.core.exceptions import SecureAgentError


def _silent(*args, **kwargs):
    return None


class ToolGateway:
    def __init__(self,registry=None,mcp=None,**mcp_options):
        if mcp is None:
            from app.mcp.gateway import MCPToolGateway
            mcp=MCPToolGateway.from_registry(registry,**mcp_options)
        self.mcp=mcp

    def discover(self,session,record):
        return self.mcp.discover(session,record)

    def authorize_and_validate(self,call,role,record,step,scope):
        try: name=ToolName(call.name)
        except (ValueError,TypeError):
            record(AuditEventType.TOOL_REQUESTED,AuditStatus.FAILURE,step=step)
            record(AuditEventType.TOOL_DENIED,AuditStatus.FAILURE,step=step,authorized=False)
            raise UnknownToolError("Unknown tool") from None
        record(AuditEventType.TOOL_REQUESTED,AuditStatus.SUCCESS,tool=name,step=step)
        if not permits(role,name):
            record(AuditEventType.TOOL_DENIED,AuditStatus.FAILURE,tool=name,step=step,authorized=False)
            raise ToolDeniedError("Tool denied")
        record(AuditEventType.TOOL_AUTHORIZED,AuditStatus.SUCCESS,tool=name,step=step,authorized=True)
        try:
            if type(call.arguments) is not dict or len(json.dumps(call.arguments,ensure_ascii=False))>4096:
                raise ValueError("Invalid arguments")
            args=SCHEMAS[name].model_validate(call.arguments).model_dump()
        except (ValidationError,ValueError,TypeError,OverflowError,RecursionError):
            raise ToolArgumentError("Invalid tool arguments") from None
        key="claim_id" if name is ToolName.CLAIM else "policy_id" if name is ToolName.POLICY else None
        if key and args[key] not in scope[key]:
            record(AuditEventType.TOOL_DENIED,AuditStatus.FAILURE,tool=name,step=step,authorized=False)
            raise ToolDeniedError("Requested record outside user scope")
        return name,args

    @staticmethod
    def approval_required(prepared,role):
        name,args=prepared
        return requires_approval(role,name,args)

    def execute_raw(self,prepared,session,record,step,role,scope,catalog=None,approved=frozenset()):
        """Recheck policy at dispatch and invoke MCP. The result is raw and untrusted."""
        # Recheck so even a forged/mutated prepared tuple cannot bypass role,
        # argument, record-scope or approval enforcement.
        from app.llm.base import ToolCall
        name,args=prepared
        name,args=self.authorize_and_validate(ToolCall(name.value,args),role,_silent,step,scope)
        if requires_approval(role,name,args) and call_key(name,args) not in approved:
            record(AuditEventType.TOOL_DENIED,AuditStatus.FAILURE,tool=name,step=step,authorized=False)
            raise ToolDeniedError("Tool action requires human approval")
        try:
            # Model-produced query arguments are untrusted and sanitized before
            # retrieval or replay in assistant history. Identifiers have fixed schemas.
            if name is ToolName.KNOWLEDGE:
                args=session.sanitize(args)
            return name,args,self.mcp.call(name,args,session,record,step,catalog)
        except SecureAgentError:
            record(AuditEventType.TOOL_COMPLETED,AuditStatus.FAILURE,tool=name,step=step)
            raise
        except Exception:
            record(AuditEventType.TOOL_COMPLETED,AuditStatus.FAILURE,tool=name,step=step)
            raise ToolExecutionError("Tool processing failed") from None

    @staticmethod
    def guard_result(name,raw,session,record,step,started):
        """Tool-result guard: PII tokenization into the TOOL namespace, leak and size checks."""
        try:
            safe=session.sanitize(raw,preserve_documents=name is ToolName.KNOWLEDGE)
        except SecureAgentError:
            record(AuditEventType.TOOL_COMPLETED,AuditStatus.FAILURE,tool=name,step=step)
            raise
        except Exception:
            record(AuditEventType.TOOL_COMPLETED,AuditStatus.FAILURE,tool=name,step=step)
            raise ToolExecutionError("Tool processing failed") from None
        record(AuditEventType.TOOL_COMPLETED,AuditStatus.SUCCESS,tool=name,step=step,
               counts=session.pii_counts,latency_ms=(time.perf_counter()-started)*1000)
        return safe

    def execute(self,prepared,session,record,step,role,scope,catalog=None,approved=frozenset()):
        started=time.perf_counter()
        name,args,raw=self.execute_raw(prepared,session,record,step,role,scope,catalog,approved)
        return name,args,self.guard_result(name,raw,session,record,step,started)
