"""Real UI widgets -> HTTP FastAPI -> local Ollama; prints metadata only.

Run from the repository root: python scripts/local_ui_smoke.py
Requires the existing local RAG index and installed models; downloads nothing.
Starts temporary loopback servers and closes them after verification.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import httpx
import uvicorn
from streamlit.testing.v1 import AppTest
from app.main import build_service, create_app
from app.core.config import get_settings
from app.agent.tool_registry import ToolRegistry
from app.security.gateway import normalize_input
from local_ui_startup_smoke import verify_live_ui


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def main():
    settings = get_settings()
    service = build_service(settings)
    executions = []
    def observe(name, handler):
        def invoke(args, session):
            executions.append(name.value)
            return handler(args, session)
        return invoke
    host = service.tool_gateway.mcp.host  # handlers run inside the in-process MCP servers
    registry = host.registry
    host.registry = ToolRegistry({
        name: observe(name, handler) for name, handler in registry._handlers.items()})
    pii = ['David Chen', 'Maria Garcia', 'John Smith', 'david.chen@example.com',
           'maria.garcia@example.com', 'john.smith@example.org', '123-45-6789',
           '987-65-4321', '987.65.4321', '(415) 555-0132', '(212) 555-0187']
    boundary = {'requests': 0, 'user_tokens': False, 'document_tokens': False, 'tool_tokens': False}
    def inspect(request):
        if request.url.path != '/api/chat':
            return
        payload = json.loads(request.content)
        assert payload['model'] == 'qwen2.5:7b'
        visible = normalize_input(json.dumps(payload, ensure_ascii=False))
        assert all(value.casefold() not in visible.casefold() for value in pii), 'PII at generation boundary'
        assert 'token_mapping' not in visible and 'original_value' not in visible
        boundary['requests'] += 1
        boundary['user_tokens'] |= '<PERSON_' in visible and '<EMAIL_' in visible and '<SSN_' in visible
        boundary['document_tokens'] |= '<DOC_PERSON_' in visible
        boundary['tool_tokens'] |= '<TOOL_PERSON_' in visible
    service._llm._client.event_hooks['request'] = [inspect]
    port, ui_port = free_port(), free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(settings, service), host='127.0.0.1',
                                         port=port, log_level='warning', access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    ui_process = None
    previous_url = os.environ.get('UI_BACKEND_URL')
    os.environ['UI_BACKEND_URL'] = f'http://127.0.0.1:{port}'
    try:
        deadline = time.monotonic() + 20
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline, 'API startup failed'
            time.sleep(.05)
        # Start the actual UI web server as well as exercise the same app with AppTest.
        with tempfile.TemporaryFile() as output:
            ui_process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run',
                'app/ui/app.py', '--server.address', '127.0.0.1', '--server.port', str(ui_port),
                '--browser.gatherUsageStats', 'false'], cwd=ROOT, env=os.environ.copy(),
                stdout=output, stderr=output)
            with httpx.Client(trust_env=False, timeout=5) as client:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        health = client.get(f'http://127.0.0.1:{ui_port}/_stcore/health')
                        if health.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    assert ui_process.poll() is None and time.monotonic() < deadline, 'UI startup failed'
                    time.sleep(.1)
                assert client.get(f'http://127.0.0.1:{ui_port}').status_code == 200
            # Execute the fresh server's script, not just its static health page.
            verify_live_ui(f'http://127.0.0.1:{ui_port}')

            app = AppTest.from_file(ROOT / 'app/ui/app.py').run(timeout=30)
            assert not app.exception and app.chat_input
            results = {}
            def submit(message, key):
                app.chat_input[0].set_value(message).run(timeout=240)
                assert not app.exception, 'UI widget execution failed'
                last = app.session_state['conversation'].messages[-1]
                assert last.received is not None, f'{key} did not return a successful response'
                result = last.received.result
                assert result.orchestration is not None and result.orchestration.engine == 'langgraph'
                assert result.orchestration.nodes[:2] == ['input_guard', 'router']
                assert result.orchestration.nodes[-2:] == ['output_guard', 'finalize']
                metadata = result.model_dump(exclude={'response'})
                assert all(value not in json.dumps(metadata) for value in pii)
                assert all(value not in result.response for value in pii), 'Protected response PII'
                results[key] = metadata
                return result

            chat = submit('My name is David Chen. My email is david.chen@example.com. '
                          'My SSN is 123-45-6789. Please acknowledge my request.', 'secure_chat')
            assert chat.orchestration.route == 'chat' and 'secure_chat' in chat.orchestration.nodes
            assert {'PERSON', 'EMAIL_ADDRESS', 'US_SSN'} <= set(chat.security.entity_types)
            app.selectbox(key='mode').set_value('Secure RAG').run(timeout=30)
            rag = submit('What is the status of claim CLM-1001?', 'secure_rag')
            assert rag.orchestration.route == 'rag' and 'secure_rag' in rag.orchestration.nodes
            assert any(s.filename == 'claim_1001.txt' for s in rag.sources)
            assert any(t.value == 'claim_1001.txt' for t in app.text)
            app.selectbox(key='mode').set_value('Secure Agent').run(timeout=30)
            app.selectbox(key='demo_role').set_value('claims_adjuster').run(timeout=30)
            agent = submit('What is the status of claim CLM-1001?', 'authorized_agent')
            assert agent.orchestration.route == 'agent' and agent.orchestration.tool_protocol == 'mcp'
            assert 'policy_gate' in agent.orchestration.nodes and 'mcp_tool' in agent.orchestration.nodes
            assert agent.tools_used == ['get_claim_status'] and agent.security.tool_pii_detected
            # Exercise the new opt-in agent knowledge sources through the real UI too.
            knowledge = submit('Search the knowledge base for the privacy procedure for customer records.',
                               'agent_knowledge')
            assert 'search_knowledge' in knowledge.tools_used and knowledge.sources
            assert any(s.filename == 'procedures_privacy.txt' for s in knowledge.sources)
            before = len(executions)
            app.selectbox(key='demo_role').set_value('customer_service').run(timeout=30)
            app.chat_input[0].set_value('What is the status of claim CLM-1001?').run(timeout=240)
            assert not app.exception
            denied = app.session_state['conversation'].messages[-1]
            assert denied.error.value == 'TOOL_DENIED' and len(executions) == before
            assert any('Tool access denied' in e.value for e in app.error)
            app.button(key='clear').click().run(timeout=30)
            assert not app.session_state['conversation'].messages and not app.chat_message
            audit = Path(settings.audit_log_path).read_text()
            assert all(value not in audit for value in pii), 'PII in audit'
            assert boundary['user_tokens'] and boundary['document_tokens'] and boundary['tool_tokens']
            print(json.dumps({'result': 'PASS', 'ui_server_started': True, 'ui_widget_http_integration': True,
                'secure_chat': 'PASS', 'secure_rag': 'PASS', 'authorized_agent': 'PASS',
                'agent_knowledge_sources': 'PASS', 'unauthorized_agent': 'PASS', 'clear_conversation': 'PASS',
                'audit_privacy': 'PASS', 'generation_boundary': boundary,
                'concrete_tool_executions': executions, 'safe_response_metadata': results}, indent=2))
    finally:
        if ui_process is not None:
            ui_process.terminate()
            ui_process.wait(timeout=15)
        server.should_exit = True
        thread.join(timeout=15)
        if previous_url is None:
            os.environ.pop('UI_BACKEND_URL', None)
        else:
            os.environ['UI_BACKEND_URL'] = previous_url


if __name__ == '__main__':
    main()
