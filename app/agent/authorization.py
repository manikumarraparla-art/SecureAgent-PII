"""Deterministic policy. Role is supplied by the application, never the model."""
from app.agent.schemas import Role, ToolName

PERMISSIONS = {
    Role.CUSTOMER_SERVICE: frozenset({ToolName.POLICY,ToolName.KNOWLEDGE}),
    Role.CLAIMS_ADJUSTER: frozenset(ToolName),
    Role.SUPERVISOR: frozenset(ToolName),
}

def permits(role: Role, tool: ToolName) -> bool:
    return isinstance(role,Role) and tool in PERMISSIONS.get(role, frozenset())
