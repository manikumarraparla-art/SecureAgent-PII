# Production readiness: local reference implementation

Version 1.0 is a portfolio release of tested local behavior, not authorization to
use real customer data or expose the service remotely. **Copilot Studio is NOT
integrated.** Microsoft Foundry cloud and other managed Azure services are not
integrated either. Foundry Local is an actual local inference adapter.

## IMPLEMENTED

- Strict FastAPI input/response contracts; safe fixed errors and redacted unknown
  validation field names; release/provider
  identity and a read-only live-contract check.
- Presidio/spaCy detection, normalization, strict SSN recognition, tokenization,
  known-value boundary checks and user-only restoration intersected with server
  policy. SSNs are excluded by the default ceiling, not by an immutable ban.
- Document sanitization before embeddings and again before generation; TOOL/DOC
  namespaces never restored by user restoration requests.
- Actual LangGraph execution for all modes, bounded agent loops, explicit error
  paths, sanitized state, request-local vaults outside checkpoints, cleanup and
  tracing disabled for cloud sinks.
- Deterministic role, schema, record-scope and approval checks, independent of
  model prose. MCP allowlist/server binding/pinned schemas, strict result decoding,
  real SDK transport, tool-result guard and dispatch rechecks.
- Memory-only approval pause/resume, expiry, capacity limits and single-use
  decisions. Foundry reasoning-block removal and native-agent capability rejection.
- Metadata-only audit/trace records; HTTP-only UI; loopback validation; no proxy
  inheritance, redirects, auto-download or remote fallback.
- Model-free regression suite and CI definition; separate real local provider,
  startup, UI and failure-path verification. See [Phase 9 evidence](phase-9-report.md).

## DEMONSTRATION / LOCAL

| Element | Actual boundary |
|---|---|
| Demo roles and supervisor buttons | Caller-selected labels, no authenticated user or approver |
| Record scope | One explicitly named claim/policy per request; not proof of ownership |
| Claims/policies/documents | Synthetic fixtures only; no enterprise systems or RAG ACLs |
| MCP servers | First-party Python servers within the API process; no remote transport or cryptographic server identity |
| Inference | Ollama/qwen2.5:7b and Foundry Local/qwen3-1.7b; Ollama embeddings for both |
| Persistence | Local SQLite retrieval index; audit/trace files; no durable approval or conversation store |
| UI/runtime | Trusted local Streamlit/browser and one local API process; synchronous requests |
| Security evidence | Specific tested invariants; not certification, perfect PII detection or general prompt-injection immunity |

## PRODUCTION REPLACEMENTS

| Required work | What must replace or extend the demonstration |
|---|---|
| Enterprise IdP/OAuth/OIDC | Validate tokens, derive roles server-side, bind tenant/user/approver identity; separation of duties |
| Enterprise record access | Real claims/policy APIs with tenant, record and field permissions; document ACLs before retrieval |
| Remote MCP infrastructure | Authenticated/encrypted transport, reviewed server identity/contracts, least privilege, isolation and supply-chain controls |
| Managed secrets | Workload identity, secret lifecycle/rotation, encrypted storage and backups; never credentials in prompts |
| Central observability | Reviewed redaction/retention, metrics and alerting, explicit mandatory-audit failure policy; no raw prompts/reasoning by default |
| Persistent state where justified | Protected checkpoint/approval storage, lifecycle and deletion controls; raw PII/vaults must remain outside graph state |
| Enterprise deployment | Hardened runtime, TLS, ingress/egress controls, tenant isolation, availability, upgrade and rollback procedures |
| Abuse/resource controls | Rate limits, total body/response limits, concurrency budgets, cancellation and denial-of-service testing |
| Model/data governance | Reviewed models/licenses, provenance, evaluations, adversarial tests, quality monitoring, human review and change control |
| Formal assurance | Security/privacy review, penetration testing, retention/deletion obligations and relevant compliance assessment |

A production replacement is a future project and a new trust-boundary review,
not a feature silently implied by the portfolio diagram. Audit sink failures
currently permit requests to continue. PII detection can miss values; semantic
injection and hallucination remain possible. Foundry Agent remains unavailable.
See the [threat model](threat-model.md) for residual risks and the
[Microsoft mapping](microsoft-production-mapping.md) for optional future services.
