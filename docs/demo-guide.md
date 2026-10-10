# Five-minute synthetic demo

Prepare models, index and processes before the clock starts using the
[startup guide](startup.md). Confirm backend **1.0.0**, Ollama, qwen2.5:7b and
available status. Load the existing Foundry model ahead of time if showing the
provider switch. All names, emails, SSNs and records below are synthetic.
Latency and model wording vary; allow extra time on a slower machine.

| Time | UI selection | Exact prompt / action | Expected behavior and capability |
|---|---|---|---|
| 0:00–0:50 | Secure Chat; restoration empty | `My name is David Chen. My email is david.chen@example.com. My SSN is 123-45-6789. Please acknowledge my claim inquiry.` | Security panel detects PERSON/EMAIL_ADDRESS/US_SSN; placeholders stay protected. Orchestration shows input_guard → router → secure_chat → output_guard → finalize. Demonstrates PII protection before inference. |
| 0:50–1:30 | Secure RAG; Sources to retrieve 3 | `What is the status of claim CLM-1001?` | Grounded answer normally says APPROVED, with claim_1001.txt among safe sources. Document PII remains protected. Demonstrates secured embeddings/retrieval and the graph RAG route. |
| 1:30–2:10 | Secure Agent; Demo Role claims_adjuster | `What is the status of claim CLM-1001?` | get_claim_status executes via claims MCP. Show policy_gate, mcp_tool and tool_result_guard in Orchestration. Demonstrates an actual SDK tool call after deterministic authorization. |
| 2:10–2:40 | Secure Agent; Demo Role customer_service | `What is the status of claim CLM-1001?` | A requested claim tool is denied before execution; fixed safe error. The role gate is code, not the model's opinion. |
| 2:40–3:40 | Secure Agent; Demo Role claims_adjuster | `What is the status of claim CLM-1004?` | Pause before MCP. Click **Approve as supervisor (demo)** to resume; granted approval appears in metadata. Submit again and click **Deny**: safe denial, no read. Demonstrates interrupt/resume; buttons do not authenticate anyone. |
| 3:40–5:00 | Stop your API with Ctrl+C; switch as below; refresh UI; Secure Chat, then Secure RAG | Chat: `Please briefly confirm that you received my claim inquiry.` RAG: `What is the status of claim CLM-1001?` | Provider becomes Foundry Local with qwen3-1.7b-generic-gpu and version 1.0.0. Same boundary/graph, different local provider. Secure Agent must show the capability rejection, with no tool dispatch. |

Provider-switch commands, from the repository root in the activated environment:

```bash
# Foundry daemon/model already running and loaded during preparation.
LLM_PROVIDER=foundry python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# Another terminal:
python -m scripts.check_runtime --provider foundry
```

Keep Ollama running for embeddings. To restore the original setup, Ctrl+C this
API and restart with `LLM_PROVIDER=ollama`. Do not stop unrelated processes.

If time allows, repeat Chat with Restore user PII types PERSON and EMAIL_ADDRESS
and ask: `My name is David Chen. My email is david.chen@example.com. My SSN is 123-45-6789. Repeat my name and email exactly and keep the SSN placeholder.`
Only requested types permitted by the server ceiling are restored; SSN is
excluded by default. Tool/document values are never restored through this option.

Optional record-scope demonstration, claims_adjuster:
`Look up CLM-1001 and CLM-1002.` The bulk scope is refused before planning/tools.

The UI explains control flow; it does not expose the private model payload or
vault. Use `python -m scripts.local_release_smoke --provider ollama` for instrumented
boundary evidence. The fresh-start smoke submits a real browser-protocol chat.
The model may choose not to request a tool: report that limitation rather than
changing security policy to force a pleasing demo. These are fictional claim
statuses, not automated insurance decisions. Finish by showing
[implemented vs production replacements](production-readiness.md).
