"""Explicit allowlist of first-party handlers; only the MCP host dispatches them."""
from types import MappingProxyType
from app.agent.schemas import ToolName

class ToolRegistry:
    def __init__(self, handlers):
        if set(handlers) != set(ToolName):
            raise ValueError("Registry must contain exactly the Phase 4 tools")
        if not all(callable(handler) for handler in handlers.values()):
            raise ValueError("Invalid tool handler")
        self._handlers=MappingProxyType(dict(handlers))

    def dispatch(self, name: ToolName, arguments: dict, session):
        return self._handlers[name](arguments,session)


def build_registry(retriever, gateway, top_k):
    from app.tools.claims import get_claim_status
    from app.tools.policies import get_policy_details
    from app.tools.knowledge import search_knowledge
    return ToolRegistry({ToolName.CLAIM:get_claim_status,ToolName.POLICY:get_policy_details,
        ToolName.KNOWLEDGE:lambda args,session:search_knowledge(args,session,retriever,gateway,top_k)})
