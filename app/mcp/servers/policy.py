"""Policy MCP server: exact-key synthetic insurance-policy lookup."""
from app.agent.schemas import ToolName
from app.mcp.catalog import MCPServerName
from app.mcp.servers._common import build_tool_server


def build_policy_server(handler):
    return build_tool_server(MCPServerName.POLICY, ToolName.POLICY, handler)
