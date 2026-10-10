"""Shared builder for the single-tool, in-process MCP servers.

Servers expose a capability; they do not authorize. Authorization, record
scope and approval run in the application before a server is ever reached.
Servers still validate their own input (they trust no client) and never let
exception text cross the protocol, because it could quote sensitive data.
"""
import json

import anyio
import mcp_types as types
from mcp.server import Server
from pydantic import ValidationError

from app.agent.schemas import DESCRIPTIONS, SCHEMAS, ToolName
from app.core.exceptions import ErrorCode, SecureAgentError
from app.mcp.catalog import MCPServerName, pinned_input_schema

from app.version import VERSION as SERVER_VERSION


def error_result(code: ErrorCode) -> types.CallToolResult:
    return types.CallToolResult(is_error=True,
                                content=[types.TextContent(type="text", text=json.dumps({"error_code": code.value}))])


def build_tool_server(server_name: MCPServerName, tool: ToolName, handler) -> Server:
    """`handler(arguments) -> dict` is bound by the host to one request."""
    definition = types.Tool(name=tool.value, description=DESCRIPTIONS[tool], input_schema=pinned_input_schema(tool))

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[definition])

    async def call_tool(ctx, params) -> types.CallToolResult:
        if params.name != tool.value:
            return error_result(ErrorCode.TOOL_UNKNOWN)
        try:
            arguments = SCHEMAS[tool].model_validate(params.arguments or {}).model_dump()
        except (ValidationError, ValueError, TypeError):
            return error_result(ErrorCode.TOOL_ARGUMENTS_INVALID)
        try:
            # Off the event loop, so the client's timeout can abandon a slow lookup.
            result = await anyio.to_thread.run_sync(handler, arguments, abandon_on_cancel=True)
            text = json.dumps(result, ensure_ascii=False, allow_nan=False)
        except SecureAgentError as exc:
            return error_result(exc.error_code)
        except Exception:
            return error_result(ErrorCode.TOOL_FAILED)
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

    return Server(server_name.value, version=SERVER_VERSION, on_list_tools=list_tools, on_call_tool=call_tool)
