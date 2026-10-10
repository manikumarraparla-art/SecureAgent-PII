"""Claims MCP server: exact-key synthetic claim status lookup."""
from app.agent.schemas import ToolName
from app.mcp.catalog import MCPServerName
from app.mcp.servers._common import build_tool_server


def build_claims_server(handler):
    return build_tool_server(MCPServerName.CLAIMS, ToolName.CLAIM, handler)
