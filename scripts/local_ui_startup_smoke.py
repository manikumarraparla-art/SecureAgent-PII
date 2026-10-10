"""Verify the documented CLI in a cold server process and a real live session.

Run from the repository root: python scripts/local_ui_startup_smoke.py
Starts fresh FastAPI and Streamlit processes on temporary loopback ports.
Submits a real chat through the browser protocol; closes only owned processes.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import httpx
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]


def wait_for_health(client, url, process):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        try:
            response = client.get(url)
            if response.status_code in (200, 503):
                return response
        except httpx.HTTPError:
            pass
        if process.poll() is not None:
            raise RuntimeError('Local verification server exited before startup')
        time.sleep(.1)
    raise RuntimeError('Local verification server startup timed out')


def verify_live_ui(base_url, submit=False):
    """Execute the server-side script using the same protocol as the browser.

    Unlike GET / or /health, a WebSocket rerun actually imports and runs the UI.
    Keep XSRF/origin checks enabled and pass the server-issued cookie/token.
    """
    with httpx.Client(trust_env=False, timeout=5) as client:
        assert client.get(base_url).status_code == 200
        cookie_header = '; '.join(f'{k}={v}' for k, v in client.cookies.items())
        xsrf = client.cookies.get('_streamlit_xsrf')
    protocols = ['streamlit'] + ([xsrf] if xsrf else [])
    with connect(base_url.replace('http://', 'ws://') + '/_stcore/stream',
                 origin=base_url, subprotocols=protocols, proxy=None,
                 additional_headers={'Cookie': cookie_header}, open_timeout=10) as websocket:
        request = BackMsg()
        request.rerun_script.context_info.url = base_url
        request.rerun_script.context_info.timezone = 'UTC'
        request.rerun_script.context_info.locale = 'en-US'
        websocket.send(request.SerializeToString())
        title = False
        backend_available = False
        chat_input = False
        chat_id = None
        submitted = False
        graph_seen = False
        route_seen = False
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            message = ForwardMsg()
            try:
                message.ParseFromString(websocket.recv(timeout=10))
            except TimeoutError:
                continue
            kind = message.WhichOneof('type')
            if kind == 'delta' and message.delta.HasField('new_element'):
                element = message.delta.new_element
                element_kind = element.WhichOneof('type')
                assert element_kind != 'exception', 'Live UI script raised an exception'
                if element_kind == 'heading':
                    title |= element.heading.body == 'SecureAgent-PII'
                if element_kind == 'metric':
                    backend_available |= element.metric.label == 'Backend' and element.metric.body == 'Available'
                chat_input |= element_kind == 'chat_input'
                if element_kind == 'chat_input': chat_id = element.chat_input.id
                if element_kind == 'text':
                    graph_seen |= element.text.body == 'Orchestration: LangGraph'
                    route_seen |= element.text.body == 'Route: Secure Chat'
            if kind == 'script_finished':
                if message.script_finished == ForwardMsg.FINISHED_EARLY_FOR_RERUN:
                    continue
                assert message.script_finished == ForwardMsg.FINISHED_SUCCESSFULLY
                assert title and backend_available and chat_input, 'Live UI did not finish rendering its workspace'
                if submit and not submitted:
                    widget = request.rerun_script.widget_states.widgets.add()
                    widget.id = chat_id
                    widget.chat_input_value.data = ('My name is David Chen. My email is david.chen@example.com. '
                                                    'Please acknowledge my claim inquiry.')
                    websocket.send(request.SerializeToString())
                    submitted = True
                    deadline = time.monotonic() + 240
                    continue
                if submit:
                    assert graph_seen and route_seen, 'Fresh UI chat did not complete through LangGraph'
                return {'chat_submitted': submitted, 'langgraph_chat': graph_seen and route_seen,
                        'title_rendered': title, 'backend_available': backend_available,
                        'chat_input_rendered': chat_input, 'script_exception': False}
        raise RuntimeError('Live UI did not finish execution')


def main():
    # No port-owner discovery or termination: both application processes are ours.
    from scripts.check_runtime import inspect_runtime
    processes = []
    with socket.socket() as api_socket, socket.socket() as ui_socket:
        api_socket.bind(('127.0.0.1', 0))
        ui_socket.bind(('127.0.0.1', 0))
        api_port, ui_port = api_socket.getsockname()[1], ui_socket.getsockname()[1]
    backend_url, ui_url = f'http://127.0.0.1:{api_port}', f'http://127.0.0.1:{ui_port}'
    with tempfile.TemporaryDirectory(prefix='secureagent-fresh-') as directory, \
            tempfile.TemporaryFile() as output, httpx.Client(trust_env=False, timeout=5) as client:
        environment = {**os.environ, 'UI_BACKEND_URL': backend_url, 'LLM_PROVIDER': 'ollama',
                       'STREAMLIT_SERVER_PORT': str(ui_port),
                       'AUDIT_LOG_PATH': str(Path(directory) / 'audit.jsonl'),
                       'GRAPH_TRACE_LOG_PATH': str(Path(directory) / 'trace.jsonl')}
        try:
            api = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app',
                '--host', '127.0.0.1', '--port', str(api_port)], cwd=ROOT, env=environment,
                stdout=output, stderr=output)
            processes.append(api)
            assert wait_for_health(client, backend_url + '/health', api).status_code == 200
            with httpx.Client(base_url=backend_url, trust_env=False, timeout=5) as backend:
                identity = inspect_runtime(backend, 'ollama')
            ui = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app/ui/app.py'],
                cwd=ROOT, env=environment, stdout=output, stderr=output)
            processes.append(ui)
            wait_for_health(client, ui_url + '/_stcore/health', ui)
            evidence = verify_live_ui(ui_url, submit=True)
            assert api.poll() is None and ui.poll() is None
            trace = (Path(directory) / 'trace.jsonl').read_text()
            assert any(json.loads(line)['event'] == 'GRAPH_COMPLETED' for line in trace.splitlines())
            assert 'david.chen@example.com' not in trace and 'David Chen' not in trace
            print(json.dumps({'result': 'PASS', 'runtime': identity,
                'cold_api_process': True, 'cold_streamlit_process': True,
                'live_websocket_session': evidence}, indent=2))
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=15)


if __name__ == '__main__':
    # Direct script invocation and module invocation both work from the root.
    sys.path.insert(0, str(ROOT))
    main()
