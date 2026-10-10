# Fresh startup and stale-process diagnosis

Run commands from the repository root with the same Python 3.14 virtual environment
in every terminal. No startup or verification command here kills a port owner.
The application version is `1.0.0`; this is the v1.0 release candidate until the
owner completes the [release checklist](release-checklist.md).

## Identify existing processes first

On macOS (Linux with lsof works too):

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:8501 -sTCP:LISTEN
lsof -nP -iTCP:11434 -sTCP:LISTEN
```

8000 is FastAPI; 8501 is Streamlit; 11434 is Ollama. Inspect a PID from that
output before doing anything to it:

```bash
ps -p <PID> -o pid,ppid,lstart,command
lsof -a -p <PID> -d cwd
```

On Linux without lsof use `ss -ltnp` to inspect listeners. A process can hold old
Python modules even when its working directory is the current checkout. Compare
its command/start time with the checkout and the actual HTTP identity:

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/health
curl --fail --silent --show-error http://127.0.0.1:8000/openapi.json
python -m scripts.check_runtime --provider ollama
```

The read-only checker requires matching health/OpenAPI release versions, the
requested healthy provider, and all three strict orchestration request schemas.
It never submits a prompt or stops a process. Health alone is insufficient:
a stale Phase 5 API returned 200 but rejected the Phase 8 UI's
`include_orchestration` as `extra_forbidden`. Keep strict validation and the UI
payload. Stop only the confirmed old application, preferably Ctrl+C in its owning
terminal, then restart. If that terminal is gone, recheck PID/command/working
directory and use `kill -TERM <PID>` only for the confirmed application you own.
Do not use blanket `killall`, `pkill` or automatic
port-killing scripts. If the process is not yours, use an agreed alternative port.

## Ollama, API, UI

Prerequisite: provision the existing approved qwen2.5:7b and nomic-embed-text
models separately. These commands never download a model.

```bash
source .venv/bin/activate
ollama list
```

If 11434 has no Ollama listener, start the installed Ollama application or run
`ollama serve` in its own terminal. Do not start a duplicate daemon. If required
tags are absent, stop and provision them through your normal reviewed setup.
For daemon-level cloud restrictions, set `OLLAMA_NO_CLOUD=1` before starting
Ollama; an application `.env` cannot reconfigure an existing daemon.

Build the synthetic index once, and rebuild after changing the fixtures or
embedding configuration:

```bash
python -m app.rag
```

API terminal (after safely stopping an older API on this port):

```bash
source .venv/bin/activate
LLM_PROVIDER=ollama python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Second terminal, verify before launching the UI:

```bash
source .venv/bin/activate
python -m scripts.check_runtime --provider ollama
UI_BACKEND_URL=http://127.0.0.1:8000 python -m streamlit run app/ui/app.py --server.port 8501
```

Open http://127.0.0.1:8501. Confirm Backend Available, Provider Ollama, Model
qwen2.5:7b, and **Backend version: 1.0.0**. Submit the synthetic Secure Chat prompt
in the [demo guide](demo-guide.md), and open Orchestration to confirm LangGraph.
UI file watching is disabled; restart the UI after source changes. Each terminal
must use the repository root so `.streamlit/config.toml` is applied.

## Switch to Foundry Local

Keep Ollama available: both providers use its nomic-embed-text for embeddings.
Use the already installed Foundry CLI and model; do not download another model.

```bash
foundry server status --output json
# Only if not already running:
foundry server start
foundry model load qwen3-1.7b-generic-gpu:2
```

Stop your previous API with Ctrl+C, then use the same port/UI:

```bash
LLM_PROVIDER=foundry python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# Separate terminal:
python -m scripts.check_runtime --provider foundry
```

Refresh the UI status. Chat/RAG work; Agent must return the safe capability
rejection. The factory discovers Foundry's dynamic loopback endpoint; never copy
a machine-specific port into source. See [Foundry details](foundry-local.md).
Switch back by stopping your API and repeating the Ollama API command.

## Reproducible verification without disturbing existing applications

```bash
python -m scripts.local_ui_startup_smoke
python -m scripts.local_release_smoke --provider ollama
python -m scripts.local_release_smoke --provider foundry
python scripts/local_ui_smoke.py
python -m scripts.local_provider_smoke --provider foundry
```

The fresh-start script launches new FastAPI and Streamlit OS processes on
temporary loopback ports, checks release identity, opens a real Streamlit
WebSocket session, submits a chat widget, checks rendered LangGraph metadata and
a completed graph trace, then stops only its own processes. Existing model
daemons are reused. The release script runs real inference; its deliberately
failed connection/MCP handler belong only to its isolated test service.
