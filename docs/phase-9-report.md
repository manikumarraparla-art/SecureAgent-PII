# Phase 9 — final release and portfolio delivery

Status: implementation and local verification complete (2026-10-05). This is a local v1.0.0 release candidate, not a published tag or
production certification. No commit, merge, push or tag is authorized in this phase.

## Starting evidence

- Branch: `main`; working tree clean before any Phase 9 changes.
- Complete baseline: **719 passed in 10.47s**, no pytest warnings.
- Initial repository scan: PASS, 147 candidate files, no findings; two intentionally
  synthetic negative path fixtures were reported separately.
- Initial `pip check`: no broken requirements. The sandbox reported a pip-cache
  permission warning; it did not affect consistency checks.
- README, architecture, CI, ignore rules, dependencies, startup/smoke commands,
  threat model and repository checker inspected before changes.

## Implementation and defects

- Release identity 1.0.0 shared by FastAPI/OpenAPI, health and first-party MCP;
  package metadata is checked for consistency and the UI shows the live version.
- Read-only runtime check validates version, provider readiness and strict schemas.
  It never kills a process or sends a prompt.
- Corrected startup verification: previously it could reuse an old backend and
  only verify UI rendering. It now creates fresh API/UI processes, submits real
  WebSocket chat and verifies a completed LangGraph trace.
- Added an isolated real-provider release matrix with approval/denial, scope,
  malformed input, restoration/SSN and real transport/MCP-handler failure checks.
- Fixed a reproduced privacy defect: unknown JSON field names were copied into
  validation-error locations and could contain PII. Locations now retain only
  application-owned field names/framework markers or numeric indexes; unknown
  keys become `unknown_field`. Error types and HTTP 422 rejection are preserved.
  Three regressions verify response/log privacy and zero generation for all modes.
- Corrected a new smoke-harness assertion, not security behavior: Foundry can omit
  the SSN placeholder, making restoration `ALLOWED` rather than `PARTIAL` for the
  returned tokens. Checks now require restored name/email, an SSN-excluding policy
  and no raw SSN, accepting either valid outcome without forcing model wording.
- Removed the unused presidio-anonymizer requirement. Made existing direct anyio
  and mcp-types imports explicit at their installed versions. No package upgrade,
  model download, environment uninstall or new architecture component.
- Replaced historical architecture narration with a concise final trust-boundary
  design and updated README, demo, threat/hygiene review and production gaps.

## Verification

### Automated suite and release checks

- Baseline: **719 passed**. New: **14** (11 release identity/diagnostic/UI cases,
  3 validation-key privacy cases). Final complete suite: **733 passed in 11.17s**;
  zero skips, failures or pytest warnings.
- Two existing health assertions now include the release version. The existing
  UI-contract test expects a redacted unknown location; it still requires 422,
  `extra_forbidden`, no PII and no graph/model execution.
- `python scripts/check_repository.py`: **PASS**, 155 files, zero findings.
  The two existing negative-test home-path fixtures remain explicitly identified.
- `python -m pip --no-cache-dir check`: no broken requirements. Compilation of
  app/UI/scripts/tests and `git diff --check`: PASS. Compilation caches for the
  manual check were directed outside the repository.
- Ignore checks passed for .env, audit/trace logs, vector SQLite, virtualenv,
  pytest cache, model weights, private keys and Streamlit secrets. No tracked .env,
  logs or vector-store entries. Ten required documents all retain their synthetic
  fixture labels; sample/test/tool identities are not replaced with real records.
- Existing GitHub Actions definition remains unchanged: SHA-pinned actions,
  read-only permissions, Python 3.14, hygiene/compile/dependency/full-suite checks,
  no generation models or excluded security tests. Workflow validation passes;
  no hosted run for this unpushed revision is claimed.

### Real local runtime evidence

All runs used existing local models and the existing synthetic index. No model
was downloaded, daemon stopped, request-validation rule relaxed or security
component bypassed. Temporary test-owned API/UI processes were closed afterward.

| Requirement | Observed evidence |
|---|---|
| Fresh application start | New Uvicorn and Streamlit OS processes; health/OpenAPI 1.0.0; real WebSocket chat submission; rendered LangGraph Chat metadata and GRAPH_COMPLETED trace |
| Ollama Chat/RAG/Agent | Release matrix: 18 scenario outcomes PASS; Chat/RAG/authorized Agent HTTP 200; 10 actual generation-boundary requests inspected |
| Real UI with Ollama | Chat, RAG, authorized Agent, knowledge-source Agent, unauthorized Agent and clear-conversation flows PASS; seven generation requests inspected |
| MCP authorization | Authorized claim tool passed through real SDK/in-process MCP; customer_service lookup HTTP 403 TOOL_DENIED, no handler execution |
| Record scope | Real API bulk-ID request HTTP 403 TOOL_DENIED and no tool execution; existing deterministic tests also cover model-proposed out-of-scope IDs |
| Human approval | CLM-1004 HTTP 202 with zero execution before decision; approve → 200/granted/one execution; separate pause then deny → 403 APPROVAL_DENIED/no execution |
| Foundry Local | Release matrix: 11 outcomes PASS; qwen3-1.7b-generic-gpu Chat/RAG/restoration HTTP 200; three generation-boundary requests inspected |
| Foundry UI | Real Streamlit AppTest widgets over HTTP Chat/RAG PASS, correct provider/model display, safe Agent rejection; separately launched UI WebSocket render PASS |
| Foundry Agent | HTTP 409 LLM_CAPABILITY_UNSUPPORTED before inference/tool execution; no text-to-tool workaround |
| PII/restoration/SSN | Name/email/SSN tokens observed before inference; permitted user name/email restored after inference; SSN not returned and excluded by effective server policy |
| Document/tool PII | DOC/TOOL namespaces observed at model boundary; protected source/tool values absent from answers; user restoration cannot expose them |
| Malformed requests | Strict flag types and PII-bearing extra keys rejected with 422 in every mode/provider; no generation for rejected requests |
| Provider failure | Test-owned client pointed to a reserved, non-listening loopback socket: real connection failure → fixed 503 LLM_UNAVAILABLE; model daemon untouched |
| MCP/tool failure | Deliberately failing test-owned handler across real SDK/server transport → fixed 500 TOOL_FAILED; exception PII absent; no follow-up generation |
| Audit/trace/reasoning | Raw synthetic PII absent from provider payloads and audit/trace; no reasoning tags in API responses; dedicated existing tests cover malformed/unclosed reasoning and hostile output |
| Stale-process check | Existing developer API on 8000 was correctly rejected as an older release by the read-only checker; left untouched |

Commands executed successfully:

```bash
python -m pytest -q
python scripts/check_repository.py
python -m pip --no-cache-dir check
python -m compileall -q app secureagent_ui scripts tests
python -m scripts.local_ui_startup_smoke
python -m scripts.local_release_smoke --provider ollama
python -m scripts.local_release_smoke --provider foundry
python scripts/local_ui_smoke.py
python -m scripts.local_provider_smoke --provider foundry
git diff --check
```

The intentional check against the older port-8000 API exited 1, as expected;
fresh test-owned release APIs passed the same checker. Do not confuse that
expected stale-process rejection with a failing v1.0 runtime.

### Warnings and evidence limits

No pytest warnings. Streamlit AppTest emitted its expected bare-thread
`missing ScriptRunContext` warning; the real cold WebSocket execution had no
script exception. Deliberate provider-failure scenarios emitted fixed
`LLM_UNAVAILABLE` application warnings without exception/input text. The initial
sandbox pip-cache warning disappeared with `--no-cache-dir`. The first Foundry
smoke attempt exposed the assertion issue described above and passed after its
correction; no security invariant was relaxed.

Real generation remained real throughout the release matrix. Failure injection
was confined to an isolated client connection/MCP handler and is labeled as such.
Ordinary regression tests use scripted model/retrieval boundaries so CI needs no
LLM. No pristine dependency reinstall or standalone CVE/history scan was performed;
`pip check` proves installed consistency, not vulnerability absence or a universal
cross-platform lockfile. Tests and privacy observations apply to the included
synthetic scenarios, not every possible PII/injection input.

## Files and intentional scope

Created: `app/version.py`, `scripts/check_runtime.py`, `scripts/local_release_smoke.py`,
`tests/test_release.py`, `docs/startup.md`, `docs/production-readiness.md`,
`docs/release-checklist.md`, `docs/phase-9-report.md`.

Modified: `app/main.py`, `app/models/chat.py`, `app/api/errors.py`, `app/mcp/servers/_common.py`,
`pyproject.toml`, `secureagent_ui/contracts.py`, `secureagent_ui/api_client.py`,
`secureagent_ui/workspace.py`, `tests/test_api.py`, `tests/test_phase8_ui_contract.py`, `requirements.txt`,
`scripts/local_ui_startup_smoke.py`, `README.md`, `architecture.md`,
`docs/demo-guide.md`, `docs/repository-hygiene.md`, `docs/threat-model.md`.

Intentionally untouched: security gateway/tokenizer/restoration, LangGraph
nodes/state/routing, policy/authorization/record scope/approval, MCP dispatch and
result guard, provider clients/reasoning removal, API request validation,
synthetic fixtures, RAG storage, `.gitignore`, `.env.example`, Streamlit security
settings, `requirements-dev.txt`, GitHub Actions and the repository checker.
Existing ignore/checker behavior already meets the reviewed requirements.

## Remaining limits and review gates

Demo identities, no ownership/RAG ACLs, detector blind spots, semantic injection,
model variability, in-memory approvals, local process trust and continue-on-audit-
failure policy remain. SSN protection depends on the configured server ceiling.
Foundry Agent is intentionally unsupported. Copilot Studio is not integrated.
The bundled scan is heuristic, not a dependency CVE or complete history audit.
Hosted CI on this unpushed revision cannot be confirmed locally. Owner review,
independent publication scans, final commit/merge/push and optional tag remain open.

## Exact startup

From the repository root, after identifying/stopping only your stale application
processes using [startup.md](startup.md), with approved models already installed:

```bash
source .venv/bin/activate
ollama list
# If the daemon is absent, run ollama serve in its own terminal.
python -m app.rag
LLM_PROVIDER=ollama python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Separate terminal:

```bash
source .venv/bin/activate
python -m scripts.check_runtime --provider ollama
UI_BACKEND_URL=http://127.0.0.1:8000 python -m streamlit run app/ui/app.py --server.port 8501
```

Foundry: start/load the existing daemon/model as described in startup.md, stop
your old API, and substitute `LLM_PROVIDER=foundry`; verify with
`python -m scripts.check_runtime --provider foundry`. Keep Ollama for embeddings.

## Git status and runtime handoff

Branch remains `main`. **17 tracked files modified, 8 new files untracked**
(the complete lists are above). No files staged, no commit, merge, push or tag.
`git diff --check` is clean. Review with `git status --short --branch` and
`git diff`; include the untracked files in the review.

Existing developer processes on 8000/8501 and model daemons were left running.
They can still hold the previous source in memory. Before using the final UI,
safely identify and stop only your old API/UI, restart with the commands above,
and require the read-only runtime check to pass and the UI to display 1.0.0.
All v1.0 runtime evidence was gathered from fresh, isolated release processes.
