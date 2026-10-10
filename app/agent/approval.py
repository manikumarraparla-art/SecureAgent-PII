"""Deterministic human-approval policy (demonstration, not production identity).

Approval is evaluated only after role authorization, argument validation and
record scope have already allowed a call. It can add a gate; it can never
grant a capability that policy denied. Roles are demo labels, not identities.
"""
from app.agent.schemas import Role, ToolName

# Synthetic record-sensitivity classification owned by the application, not
# derived from tool output: claims referred to special investigation.
RESTRICTED_CLAIMS = frozenset({"CLM-1004"})
# Roles that may read restricted claims without a second person's approval.
DIRECT_ACCESS_ROLES = frozenset({Role.SUPERVISOR})
# Roles that may decide a pending approval.
APPROVER_ROLES = frozenset({Role.SUPERVISOR})


def requires_approval(role: Role, tool: ToolName, arguments: dict) -> bool:
    return (tool is ToolName.CLAIM and arguments.get("claim_id") in RESTRICTED_CLAIMS
            and role not in DIRECT_ACCESS_ROLES)


def may_approve(role: object) -> bool:
    return isinstance(role, Role) and role in APPROVER_ROLES
