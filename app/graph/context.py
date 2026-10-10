"""Trusted request-scoped security context (LangGraph runtime context).

One instance per request. It holds everything that must not enter graph
state: the raw message until it is protected, token vaults, quarantined raw
tool output, the untrusted model draft and the re-identified result. It is
cleared when the request ends; a TokenVault refuses serialization, so it
cannot be copied into a checkpoint by mistake.
"""
from dataclasses import dataclass, field
import time

from app.graph.names import Route
from app.security.tokenizer import TokenVault


@dataclass(eq=False)
class RequestContext:
    request_id: str
    mode: Route
    requested_types: tuple = ()
    role: object = None
    include_sources: bool = False
    include_orchestration: bool = False
    top_k: int | None = None
    message: str = field(default="", repr=False)
    started: float = field(default_factory=time.perf_counter)
    vault: TokenVault = field(default_factory=TokenVault, repr=False)
    protected: object = field(default=None, repr=False)
    session: object = field(default=None, repr=False)
    # Secure RAG progress, kept for safe failure auditing.
    document_vaults: dict = field(default_factory=dict, repr=False)
    document_counts: dict = field(default_factory=dict)
    hits: list = field(default_factory=list, repr=False)
    sources: list = field(default_factory=list)
    # Agent: record scope, MCP catalog and the call pipeline.
    scope: dict = field(default_factory=dict)
    catalog: object = None
    proposed: tuple = field(default=(), repr=False)      # untrusted model proposals
    prepared: list = field(default_factory=list, repr=False)  # authorized, validated calls
    quarantine: list = field(default_factory=list, repr=False)  # raw MCP output awaiting the guard
    batch: list = field(default_factory=list, repr=False)  # guarded results of this planner turn
    seen_calls: set = field(default_factory=set, repr=False)
    approved_calls: frozenset = field(default=frozenset(), repr=False)
    approval_id: str | None = field(default=None, repr=False)
    approval_request: dict | None = None
    draft: str | None = field(default=None, repr=False)   # untrusted model answer
    restored: object = field(default=None, repr=False)    # may contain re-identified PII
    result: object = field(default=None, repr=False)
    error: Exception | None = field(default=None, repr=False)

    def clear(self) -> None:
        self.message = ""
        self.vault.clear()
        if self.session is not None:
            self.session.clear()
        for vault in self.document_vaults.values():
            vault.clear()
        self.document_vaults.clear()
        self.hits.clear()
        self.proposed = ()
        self.prepared.clear()
        self.quarantine.clear()
        self.batch.clear()
        self.draft = None
        self.restored = None
