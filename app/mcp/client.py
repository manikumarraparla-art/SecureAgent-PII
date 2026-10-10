"""Synchronous facade over the official MCP Python SDK client.

Transport: the SDK's in-memory transport in `legacy` mode, i.e. real JSON-RPC
2.0 messages (`initialize`, `tools/list`, `tools/call`) over in-process memory
streams. Only in-process `Server` objects are accepted: URLs, stdio commands
and other transports are refused, so no remote or subprocess server can be
configured. No sampling, elicitation or roots callback is registered, so an
MCP server cannot ask this application to call an LLM or read local files.
"""
from dataclasses import dataclass
import logging

import anyio
from mcp import Client
from mcp.server import Server

from app.core.exceptions import ErrorCode, SecureAgentError

# The SDK logs handler tracebacks; our servers never raise, but keep any
# future SDK diagnostics from echoing payload-derived text into app logs.
logging.getLogger("mcp").setLevel(logging.WARNING)

MAX_LISTED_TOOLS = 64


class MCPUnavailableError(SecureAgentError):
    error_code = ErrorCode.MCP_UNAVAILABLE


class MCPTimeoutError(SecureAgentError):
    error_code = ErrorCode.MCP_TIMEOUT


@dataclass(frozen=True)
class ServerListing:
    server_name: str
    protocol_version: str
    tools: tuple[tuple[str, object], ...]  # (advertised name, advertised input schema)


def _contains_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    inner = getattr(exc, "exceptions", None)  # exception groups from task groups
    return isinstance(inner, tuple) and any(_contains_timeout(e) for e in inner)


class LocalMCPClient:
    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout

    def _run(self, server, operation):
        if not isinstance(server, Server):
            raise MCPUnavailableError("Only in-process MCP servers are permitted")

        async def run():
            with anyio.fail_after(self._timeout):
                async with Client(server, mode="legacy", cache=None) as client:
                    return await operation(client)

        try:
            return anyio.run(run)
        except SecureAgentError:
            raise
        except Exception as exc:
            # Never chain: SDK errors can carry server-supplied text.
            if _contains_timeout(exc):
                raise MCPTimeoutError("MCP request timed out") from None
            raise MCPUnavailableError("MCP request failed") from None

    def list_tools(self, server) -> ServerListing:
        async def operation(client):
            listing = await client.list_tools()
            info = client.server_info
            tools = tuple((tool.name, tool.input_schema) for tool in listing.tools[:MAX_LISTED_TOOLS])
            return ServerListing(info.name if info else "", client.protocol_version, tools)
        return self._run(server, operation)

    def call_tool(self, server, name: str, arguments: dict):
        async def operation(client):
            return await client.call_tool(name, arguments)
        return self._run(server, operation)
