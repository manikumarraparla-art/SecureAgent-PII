"""Application-owned MCP allowlist. Pure data: no SDK import, no I/O.

A discovered MCP tool is usable only if its name is an approved ToolName, it
is advertised by the one server bound to it here, and its input schema matches
the locally pinned schema. Server-supplied descriptions never reach the model.
"""
import hashlib
import json
from enum import Enum
from types import MappingProxyType

from app.agent.schemas import SCHEMAS, ToolName


class MCPServerName(str, Enum):
    CLAIMS = "claims"
    POLICY = "policy"
    KNOWLEDGE = "knowledge"
    # Audit label for any server outside the approved bindings; its own
    # self-declared name is untrusted and never recorded.
    UNAPPROVED = "unapproved"


# Transport label used in API metadata and reports.
MCP_TRANSPORT = "in-process"

APPROVED_TOOL_BINDINGS = MappingProxyType({
    ToolName.CLAIM: MCPServerName.CLAIMS,
    ToolName.POLICY: MCPServerName.POLICY,
    ToolName.KNOWLEDGE: MCPServerName.KNOWLEDGE,
})


def pinned_input_schema(tool: ToolName) -> dict:
    return SCHEMAS[tool].model_json_schema()


def schema_fingerprint(schema: object) -> str:
    """Canonical hash so a changed advertised schema (rug pull) is rejected."""
    try:
        encoded = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return ""
    return hashlib.sha256(encoded.encode()).hexdigest()


PINNED_FINGERPRINTS = MappingProxyType({tool: schema_fingerprint(pinned_input_schema(tool)) for tool in ToolName})
