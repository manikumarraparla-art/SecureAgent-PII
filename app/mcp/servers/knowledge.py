"""Knowledge MCP server: local document search reusing Phase 3 document protection."""
from app.agent.schemas import ToolName
from app.mcp.catalog import MCPServerName
from app.mcp.servers._common import build_tool_server


def build_knowledge_server(handler):
    return build_tool_server(MCPServerName.KNOWLEDGE, ToolName.KNOWLEDGE, handler)
