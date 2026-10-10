"""Per-browser-session in-memory display history. Never sent as model context."""
from dataclasses import dataclass,field
from secureagent_ui.contracts import Mode
from secureagent_ui.api_client import ReceivedResponse, PendingApproval, FailureCode, MESSAGES

@dataclass(frozen=True)
class Message:
    speaker: str
    text: str = field(repr=False)
    mode: Mode
    received: ReceivedResponse | None = field(default=None,repr=False)
    error: FailureCode | None = None
    pending: PendingApproval | None = None

class Conversation:
    def __init__(self):
        self._messages=[]
        self._decided=set()
    @property
    def messages(self): return tuple(self._messages)
    def add_user(self,text,mode): self._messages.append(Message('user',text,Mode(mode)))
    def add_response(self,received,mode):
        self._messages.append(Message('assistant',received.result.response,Mode(mode),received))
    def add_error(self,code,mode):
        self._messages.append(Message('assistant',MESSAGES[code],Mode(mode),error=code))
    def add_pending(self,pending,mode):
        self._messages.append(Message('assistant','Supervisor approval required (demo).',Mode(mode),pending=pending))
    def mark_decided(self,approval_id): self._decided.add(approval_id)
    def is_decided(self,approval_id): return approval_id in self._decided
    def clear(self):
        self._messages.clear()
        self._decided.clear()
    def __repr__(self): return f"Conversation(<redacted>, messages={len(self._messages)})"
