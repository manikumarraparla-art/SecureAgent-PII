# Repository hygiene and dependency review

All identities, claims, policies, SSNs, emails and other personal information
included in this repository are **synthetic demonstration data**. The fixtures
must not be replaced by employer/client records. This includes tests, read-only
tool dictionaries, sample prompts and historical reports, not just documents.

## Before sharing changes

From the repository root:

```bash
python scripts/check_repository.py
python -m compileall -q app secureagent_ui scripts tests
python -m pip check
python -m pytest -v
git diff --check
git status --short --branch
```

The offline checker inspects tracked and nonignored untracked working-tree files
using Git's file inventory, including hidden project files. It rejects private
artifacts, common credential assignments, provider key formats, private keys,
authorization header values, credential URLs and developer-home paths outside
tests. It prints only filename, line number and category, never matched values.
It fails for symlinks, binary/unreadable and oversized release candidates rather
than silently following or skipping them. It deliberately does not classify
synthetic names/emails/SSNs as credentials. Tests of credential rejection assemble
fictional values at runtime; no real secret belongs in a test fixture.

This is a limited heuristic, not a full secret detector, dependency vulnerability
scan, PII classifier or Git-history scanner. Review code/data provenance manually.
Use dedicated free secret scanning and history scanning before public release;
if a real secret was ever committed, revoke it and separately review history.
The checker cannot prove data is fictional or detect every format of secret.

`.gitignore` excludes environments, caches, audit JSONL, logs, local databases,
vector indexes, temporary/build files, model weights, key files and Streamlit
secrets/local config overrides. The safe `.env.example`, reviewed
`.streamlit/config.toml`, UI/source files and ten synthetic documents remain
visible. Ignore rules do not untrack previously committed files; the checker
catches tracked artifacts as well. Local ignored files are operator-owned and
not scanned by this release checker; the Phase 6 audit separately inventories
runtime files and checks audit/index content without printing it.

## Dependencies

| Group | Files / packages | Decision |
|---|---|---|
| Runtime | requirements.txt: Presidio, spaCy/English NER data, tldextract, FastAPI, uvicorn, httpx, Pydantic, dotenv, Streamlit | Exact versions retained; Pydantic made an explicit direct pin at the installed 2.13.5 version |
| Developer | requirements-dev.txt: runtime include, pytest 9.1.1, toml 0.10.2, websockets 16.1.1, PyYAML 6.0.3 | Test runner moved out of runtime requirements; existing directly used test/smoke imports and workflow parser explicitly pinned |
| Removed unused declaration | presidio-anonymizer | Phase 9 import review found no usage; tokenizer/restoration are application code. Removed from requirements; no environment uninstall or package upgrade. |
| Standard library | SQLite, chunking, audit persistence, repository checker | No vector DB or security-scanner dependency added |
| Transitive | Streamlit dataframe/chart dependencies and Presidio/spaCy dependencies | No broad upgrades; no complete cross-platform transitive lockfile supplied |

No duplicate direct requirements or newly unbounded package requirements were
introduced. The development file includes runtime requirements once. The spaCy
model is a versioned official GitHub wheel URL; installation needs network access
and substantial disk space. It is NER data, not an Ollama generation model.
The tokenizer has no dependence on a cloud anonymization API.

Python 3.14.6 on macOS ARM64 is the verified environment; Python 3.14 is the
recommended target and CI target. The project metadata retains its earlier
3.10–3.14 range, but other versions/platforms need their own verification.
Binary-only installation fails explicitly if a supported wheel is unavailable.
It never silently downgrades Python. Streamlit 1.64.0 remains unchanged.

## CI and local verification

The GitHub Actions workflow uses Ubuntu 24.04/Python 3.14, read-only repository
permissions, SHA-pinned checkout/setup actions and no persisted checkout
credentials. It checks release hygiene, installs development requirements,
checks imports by compilation and runs the full suite. There is no Ollama
installation, model pull, security-test skip, cloud deployment or smoke test in CI.
CI may download packages and the spaCy model for setup; application inference
still remains local. Python setup follows
[GitHub's Python workflow guidance](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).

Real local scripts require existing qwen2.5:7b/nomic-embed-text models and an
ingested index: local_rag_smoke.py, local_agent_smoke.py, local_ui_smoke.py and
local_ui_startup_smoke.py. They are developer-machine verification, not ordinary
unit tests. The UI startup script creates fresh API and UI processes on temporary ports and
submits a real chat through the Streamlit browser protocol. No LLM is downloaded.
Workflow execution on GitHub must be checked after the owner's reviewed push;
Phase 6 does not push or claim a hosted CI run.


## Final direct-dependency decisions (Phase 9)

| Dependency | Why it exists |
|---|---|
| fastapi / uvicorn | Validated HTTP API and local ASGI runtime |
| streamlit | Local UI, session display state and widget testing |
| presidio-analyzer / spacy / en_core_web_lg | PII recognizers and local English person detection; the wheel is NER data |
| tldextract | Explicit offline email-domain suffix validation; no live suffix-list fetch |
| pydantic | Strict API, tool and public metadata models |
| httpx | Controlled local HTTP clients with proxy inheritance and redirects disabled |
| python-dotenv | Optional local configuration loading |
| langgraph | Actual state graph, conditional transitions and interrupt/resume |
| mcp / mcp-types / anyio | Official local MCP protocol/server/client, directly imported wire types and async runtime |
| pytest / toml / websockets / PyYAML (dev) | Tests, project/config inspection, actual Streamlit browser protocol, CI YAML validation |

`anyio==4.15.1` and `mcp-types==2.3.0` were already installed; Phase 9 makes
their direct imports explicit without upgrades. `presidio-anonymizer` was unused
and is no longer requested. All other declared versions remain unchanged. This
is not a full transitive lockfile or a vulnerability audit. No package upgrade
was justified by a discovered compatibility/security defect.

LangGraph requires langchain-core and LangSmith transitively. The application
uses only LangSmith's local `tracing_context(enabled=False)` helper to suppress
tracing, not its cloud client/service. That existing transitive suppression
integration is intentionally not promoted to a standalone cloud dependency;
startup rejects tracing-enabling environment variables. Architecture tests
continue to reject cloud SDK/service use. No new cloud dependency is introduced.

Review of required synthetic fixtures, public sample prompts and test identities
is separate from credential-pattern scanning. The checker cannot certify that
all PII is fictional, examine ignored private data safely on someone else's
machine, or establish dependency CVE status. These limits remain release gates.
