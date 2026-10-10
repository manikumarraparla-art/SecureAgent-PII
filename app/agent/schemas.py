"""Strict argument schemas and server-defined capability names."""
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator

class Role(str, Enum):
    CUSTOMER_SERVICE = "customer_service"
    CLAIMS_ADJUSTER = "claims_adjuster"
    SUPERVISOR = "supervisor"

class ToolName(str, Enum):
    CLAIM = "get_claim_status"
    POLICY = "get_policy_details"
    KNOWLEDGE = "search_knowledge"

class StrictArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

class ClaimArguments(StrictArguments):
    claim_id: str = Field(pattern=r"^CLM-[0-9]{4}$", min_length=8, max_length=8)

class PolicyArguments(StrictArguments):
    policy_id: str = Field(pattern=r"^POL-[0-9]{4}$", min_length=8, max_length=8)

class KnowledgeArguments(StrictArguments):
    query: str = Field(min_length=1, max_length=2000)

    @field_validator("query")
    @classmethod
    def nonblank(cls,value):
        if not value.strip(): raise ValueError("Blank knowledge query")
        return value

SCHEMAS = {ToolName.CLAIM:ClaimArguments, ToolName.POLICY:PolicyArguments,
           ToolName.KNOWLEDGE:KnowledgeArguments}
DESCRIPTIONS = {
    ToolName.CLAIM:"Look up one synthetic insurance claim by its exact CLM-0000 identifier.",
    ToolName.POLICY:"Look up one synthetic insurance policy by its exact POL-0000 identifier.",
    ToolName.KNOWLEDGE:"Search local insurance procedures and documents with a concise question.",
}

def tool_definitions(names=None):
    """Locally defined tool contracts; MCP server descriptions are never forwarded."""
    return [{"type":"function","function":{"name":name.value,"description":DESCRIPTIONS[name],
             "parameters":schema.model_json_schema()}} for name,schema in SCHEMAS.items()
            if names is None or name in names]
