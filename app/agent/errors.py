"""Safe agent failures, never including model names/arguments/results."""
from app.core.exceptions import SecureAgentError, ErrorCode

class UnknownToolError(SecureAgentError): error_code=ErrorCode.TOOL_UNKNOWN
class ToolDeniedError(SecureAgentError): error_code=ErrorCode.TOOL_DENIED
class ToolArgumentError(SecureAgentError): error_code=ErrorCode.TOOL_ARGUMENTS_INVALID
class ToolExecutionError(SecureAgentError): error_code=ErrorCode.TOOL_FAILED
class ToolResultTooLarge(SecureAgentError): error_code=ErrorCode.TOOL_RESULT_TOO_LARGE
class AgentLimitError(SecureAgentError): error_code=ErrorCode.AGENT_LIMIT_REACHED
class ApprovalDeniedError(SecureAgentError): error_code=ErrorCode.APPROVAL_DENIED
class ApprovalNotFoundError(SecureAgentError): error_code=ErrorCode.APPROVAL_NOT_FOUND
class ApproverNotPermittedError(SecureAgentError): error_code=ErrorCode.APPROVER_NOT_PERMITTED
