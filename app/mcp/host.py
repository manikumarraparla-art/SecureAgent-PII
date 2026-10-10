"""In-process host for the approved first-party MCP servers.

The only code that dispatches registry handlers. Servers are rebuilt for each
request (and each call) and bound to that request's tool-security session,
so no server instance holds state across requests. No listener, port,
subprocess or URL is involved: nothing outside this process can reach them.
"""
from types import MappingProxyType

from app.agent.schemas import ToolName
from app.mcp.catalog import APPROVED_TOOL_BINDINGS, MCPServerName
from app.mcp.servers.claims import build_claims_server
from app.mcp.servers.knowledge import build_knowledge_server
from app.mcp.servers.policy import build_policy_server

_BUILDERS = MappingProxyType({
    MCPServerName.CLAIMS: build_claims_server,
    MCPServerName.POLICY: build_policy_server,
    MCPServerName.KNOWLEDGE: build_knowledge_server,
})
_TOOLS = MappingProxyType({server: tool for tool, server in APPROVED_TOOL_BINDINGS.items()})


class LocalMCPHost:
    def __init__(self, registry, server_overrides=None, untrusted_servers=()):
        """`server_overrides` / `untrusted_servers` exist to exercise hostile or
        compromised servers in tests; untrusted servers never receive request data."""
        self.registry = registry
        self._overrides = MappingProxyType(dict(server_overrides or {}))
        self._untrusted = tuple(untrusted_servers)

    def server(self, binding: MCPServerName, session):
        if binding not in _BUILDERS:
            raise ValueError("Unknown MCP server binding")
        tool: ToolName = _TOOLS[binding]
        registry = self.registry

        def handler(arguments):
            return registry.dispatch(tool, arguments, session)

        return self._overrides.get(binding, _BUILDERS[binding])(handler)

    def servers(self, session):
        """(configured binding, server) pairs to discover. The binding is
        assigned here, never taken from the server's self-declared name."""
        approved = [(binding, self.server(binding, session)) for binding in _BUILDERS]
        return approved + [(MCPServerName.UNAPPROVED, factory()) for factory in self._untrusted]
