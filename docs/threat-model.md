# Threat model

Scope: a security-focused local enterprise AI reference implementation using
only fictional data. This is not a certified or production-ready security product.
Assets: user PII, source documents, tool records, request-local token mappings,
approved response values, embeddings and privacy-safe audit metadata.
Attackers: malicious callers, poisoned source authors, hostile model content and
misconfigured environments. Operators and the local OS are trusted for this phase.

**MITIGATED** means a specific application invariant has implemented/tested
enforcement, not that the entire threat class is impossible. **PARTIALLY
MITIGATED** means controls reduce exposure but leave material risk.
**OUT OF SCOPE** means the prototype does not implement the required control.
See [security boundaries](security-boundaries.md) and [data flows](data-flow.md).

| Threat / status | Attack example | Existing mitigation | Residual risk | Production recommendation |
|---|---|---|---|---|
| 1. Raw user PII to LLM — PARTIALLY MITIGATED | Enter names, email and dotted/zero-width SSN variants | Gateway normalization, Presidio/spaCy, strict SSNs, tokenization and known-value leak validation before secure service invocation | Undetected/obfuscated PII can remain; detection is not perfect | Evaluate organization-specific recognizers and adversarial datasets; add reviewed output policies |
| 2. Document PII to embeddings — PARTIALLY MITIGATED | Put contact details in a claim document | Whole-document sanitization before deterministic chunking and local embeddings; vault discarded | Missed details/semantics may be encoded; sanitization is not anonymization | Apply classification, ingestion approvals, encryption and embedding access policy |
| 3. Document PII to generation — PARTIALLY MITIGATED | Tamper with stored text after ingestion | Revalidate sources and sanitize retrieved chunks again; prompt leak checks | Detector blind spots and semantic reconstruction | Add authorized retrieval filters and independent boundary evaluation |
| 4. Tool-result PII — PARTIALLY MITIGATED | Return a record with raw name/SSN | Normalize, detect/tokenize, TOOL namespace, result/history leak validation | Unknown PII and excessive business information may remain | Minimize upstream tool fields and enforce record/field permissions |
| 5. Token mapping disclosure — MITIGATED | Ask the model/UI/audit for the private vault | Mappings stay in security memory; no public schema or prompt field; cleanup in finally | Local debugger/process memory or malicious code can read it | Restrict process privileges; encrypted/short-lived vault if persistence ever becomes necessary |
| 6. Namespace collision — MITIGATED | Insert token-looking user text into documents/tools | Separate USER/DOC/TOOL handling, literal escaping, separate vaults and user-only restoration | Arbitrary malicious application changes defeat the invariant | Keep namespace validation and independent regression checks |
| 7. User prompt injection — PARTIALLY MITIGATED | Ask to ignore policy and disclose PII | Fixed system instructions plus deterministic security/authorization independent of model | Model may follow semantic instructions, hallucinate or expose undetected data | Red-team prompts; restrict capabilities; enforce output/data policies |
| 8. RAG instruction injection — PARTIALLY MITIGATED | Promote a retrieved paragraph to system authority | Fixed system role; escaped JSON context treated as data; malicious fixture tests | Models may still obey text semantically | Curate/sign document provenance; authorization filtering and retrieval evaluation |
| 9. Tool-result injection — PARTIALLY MITIGATED | Tool or MCP result says to call a different capability or enumerate customers | Untrusted tool-role JSON; structural sanitization; policy_gate and dispatch recheck every requested action; record scope blocks enumeration | Model can choose a harmful allowed action in future richer tools | Least privilege and human approval for consequential writes |
| 10. Unauthorized capability — MITIGATED | customer_service requests claim lookup | ToolGateway applies deterministic role policy before execution, including batch preflight and dispatch recheck | Caller can select a different unauthenticated demo role | Bind roles to verified identity, not request fields |
| 11. Role spoofing — OUT OF SCOPE | Caller selects supervisor in the API/UI | Closed role enum blocks invented roles; model cannot choose the application role | No authentication; valid demo-role impersonation is possible | Entra ID token validation and server-derived permissions |
| 12. Record-scope bypass — PARTIALLY MITIGATED | Model invents CLM-1002 while user named CLM-1001 | Exact user-derived IDs, one claim/policy per request, dispatch scope recheck | Explicitly naming a record does not prove ownership; RAG has no ACLs | Enforce tenant/user record ownership and document ACLs before reads |
| 13. Argument manipulation — MITIGATED | Extra fields, nested code-like values or malformed IDs | Strict flat Pydantic schemas, ASCII ID formats, bounded serialization, no eval/exec | Future tool schemas or business rules may be too permissive | Schema versioning, business invariants and API contract review |
| 14. Tool enumeration — PARTIALLY MITIGATED | Repeatedly probe unknown tool names or many records | Exactly three public schemas; unknown names rejected, not logged verbatim; bulk record scope refused | Tool schema names are intentionally visible to the model; no request rate limit | Limit sessions/rates and avoid sensitive capability metadata |
| 15. Infinite agent loops — MITIGATED | Repeated identical calls or endless model decisions | Step/call ceilings, last-step refusal, repeated-call detection, LangGraph recursion limit | Bounded requests still consume resources; many requests cause load | Global budgets, rate limits, cancellation and concurrency limits |
| 16. Oversized responses — PARTIALLY MITIGATED | Return huge tool data/model history | Tool size checked before/after sanitization; native response and argument limits | Raw HTTP bodies read before validation; no total API body/memory budget | Streaming size limits and ingress/process resource controls |
| 17. Sensitive logging — PARTIALLY MITIGATED | Log full prompts/results or exception strings | Closed audit schema, metadata-only application logging, no UI history logs; tests inspect content | Operator/framework diagnostics or unrelated browser software may leak data; audit sink failures continue | Review all telemetry sinks, redaction, retention and mandatory audit requirements |
| 18. Error leakage — MITIGATED | Backend failure embeds secret/PII in exception text | Fixed public errors, allowlisted validation locations (unknown input keys redacted), redacted config errors, mapped UI messages, hidden framework details | Local operator tracebacks/system logs remain accessible | Disable production debug pages and centralize safe exception handling |
| 19. Proxy routing — MITIGATED | Set HTTPS_PROXY to an attacker host | All model/embedding/UI clients disable environment proxies and redirects | OS/network tampering or operator code changes remain possible | Egress controls and monitored private service routing |
| 20. Remote endpoint configuration — MITIGATED | Configure a remote Ollama or UI URL | Loopback-only URL checks at config/client construction, reject credentials and extra URL components, no fallback | Local endpoint impersonation and localhost resolver tampering | Workload identity, authenticated TLS endpoints, network allowlists |
| 21. Concurrent request bleed — MITIGATED | Request A receives B's PII through reused counters | Per-request IDs, vaults/history/agent state, cleanup and overlapping-request tests | Trusted shared code could introduce a new global cache | Load/concurrency tests and request-scoped dependency design |
| 22. Poisoned documents — PARTIALLY MITIGATED | Inject bogus facts, unsafe filenames, symlink escape | Approved basenames, root-anchored no-follow regular-file reads, size/type checks, revalidation and sanitization | Allowed text can be misleading; local index/document files are not cryptographically authenticated | Provenance, signed ingestion, content review, ACLs and integrity monitoring |
| 23. Local compromise — OUT OF SCOPE | Read browser memory, monkeypatch gateway, replace localhost process | Loopback binding and new owner-only index/audit files reduce incidental exposure | Trusted OS/process access defeats application protections; existing file permissions/symlinks need operator review | Endpoint hardening, least-privilege accounts, encrypted storage and isolation |
| 24. Model uncertainty — PARTIALLY MITIGATED | Invent identifiers, ignore sources, omit token brackets | Strict action schemas, bounded loop, grounded context, known-value final agent validation | Incorrect answers/citations and invented PII remain possible; ordinary Chat/RAG are not a complete output DLP system | Task evaluation, source checks, human review and governed model versions |

### Phase 8 additions: LangGraph and MCP

| Threat / status | Attack example | Existing mitigation | Residual risk | Production recommendation |
|---|---|---|---|---|
| 25. Malicious MCP tool advertisement — MITIGATED | A server advertises `execute_shell` or `read_filesystem` | Application allowlist; only approved names from their bound server enter the catalog; rejected names not logged | A future approved tool could itself be over-privileged | Review every new tool and server before adding it to the allowlist |
| 26. MCP server impersonation / tool shadowing — MITIGATED | An extra server calls itself `claims`, or the policy server also offers `get_claim_status` | Identity comes from configured binding, self-declared name must match, first approved binding wins | In-process binding is configuration, not cryptographic identity | Authenticated server identity (mTLS / signed manifests) for remote servers |
| 27. Tool schema rug pull — MITIGATED | A server loosens `claim_id` to any string | Advertised schema must match a pinned fingerprint; strict local schema validation at gate, dispatch and server | Pinned schemas need review on legitimate change | Version and sign tool contracts |
| 28. Tool-description injection — MITIGATED | Server description says "ignore previous instructions" | Model receives local tool definitions only; server descriptions never forwarded | Tool names remain visible to the model | Keep descriptions application-owned |
| 29. MCP server reaches the model or host — MITIGATED | Server requests sampling, elicitation or roots | Client registers no such callbacks; servers cannot import model/security/OS modules; in-process only | Trusted Python code could be modified | Sandbox third-party servers; deny sampling by policy |
| 30. Hostile or malformed MCP results — MITIGATED | Non-JSON, arrays, images, NaN, huge text, PII inside error text | One bounded JSON-object text block; allowlisted error codes; generic exceptions; structural guard | Semantically misleading but well-formed data | Upstream data contracts and output validation |
| 31. Graph state / checkpoint leakage — MITIGATED | Raw PII or vaults persisted in checkpoints or traces | State holds sanitized data only; vaults in non-checkpointed context; in-memory saver, per-run deletion; closed trace schema; cloud tracing refused | Process memory remains readable by local compromise | Encrypted, access-controlled checkpoint store if persistence is ever required |
| 32. Approval bypass or abuse — PARTIALLY MITIGATED | Execute CLM-1004 without approval, reuse or forge an approval | Graph pauses before MCP; dispatch recheck requires recorded approval; 32-hex single-use ids; supervisor-only decisions; TTL and capacity bounds | Demo roles are unauthenticated: any caller can claim supervisor | Authenticated approvers, separation of duties, durable approval records |
| 33. Remote MCP endpoint configuration — MITIGATED | Configure an HTTP or stdio MCP server | No MCP URL setting; client refuses URLs, stdio parameters and non-server objects; architecture tests ban remote transports | Code changes could add one | Treat any remote MCP server as a new reviewed trust boundary |

## Production release gate

Before real data or remote access: implement verified identity and tenant/record
authorization, retrieval ACLs, secrets/secure storage management, ingress/egress
controls, rate limits, mandatory audit policy where needed, monitoring, data/model
governance, penetration testing, formal security review and compliance review.
Test failure handling and deletion/retention obligations. Microsoft platform
features can support these controls; they do not replace application decisions.
No additional production service is implemented by this phase.


## Phase 9 release review

Review reproduced an error-location leak: a caller could put PII in an unknown
JSON key and see it copied into the 422 location. The handler now allowlists
application-owned field names and redacts unknown keys. Input validation is
unchanged; three endpoint regressions and both real-provider matrices confirm
rejection without echoing PII or invoking a model.

The existing controls above remain in place. Release identity is now explicit
in health, OpenAPI and the UI; a read-only runtime check rejects a stale version,
wrong provider or incompatible orchestration schemas. This reduces accidental
version mismatch, not malicious localhost impersonation: a hostile process can
lie about its identity. Port/PID inspection and a trusted local OS remain necessary.

Final local verification covers tokenization/restoration and default SSN
protection, document/tool namespaces, deterministic authorization/scope denial,
approval pause/approve/deny, malformed requests, real provider connection failure
and a failing handler across actual MCP. Fault injection is limited to test-owned
components. Existing tests cover hostile tool advertisements/results, injection,
malformed model output, graph-state isolation, reasoning-block removal and trace
redaction. See [Phase 9 evidence and limits](phase-9-report.md).

No new claim of general injection prevention, authentication, perfect detection,
mandatory audit durability, secure erasure or production readiness is made.
No remote MCP infrastructure or Copilot Studio integration is added.
