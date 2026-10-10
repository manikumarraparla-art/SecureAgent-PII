"""Phase 8 real-provider verification: LangGraph orchestration + MCP tools.

Run from the repository root:
    python -m scripts.local_phase8_smoke --provider ollama
    python -m scripts.local_phase8_smoke --provider foundry

Uses the already-installed local models and RAG index; downloads nothing and
starts only a temporary loopback API. Evidence is gathered from the real
system, not mocks:
  - LangGraph: node paths returned by the API and the graph trace log.
  - MCP: every JSON-RPC request the official SDK sends, and proof that each
    tool handler ran inside an MCP server worker thread under the MCP host.
  - Privacy: every provider request is scanned for raw synthetic PII.
Prints only sanitized data and metadata, never token mappings.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import traceback

import httpx
import uvicorn
from mcp.shared.jsonrpc_dispatcher import JSONRPCDispatcher

from app.agent.tool_registry import ToolRegistry
from app.core.config import load_settings
from app.main import build_service, create_app
from app.mcp.catalog import MCP_TRANSPORT
from app.security.gateway import normalize_input

RAW = ('David Chen', 'Maria Garcia', 'John Smith', 'Elena Rossi', 'david.chen@example.com',
       'maria.garcia@example.com', 'john.smith@example.org', 'elena.rossi@example.com', '123-45-6789',
       '987-65-4321', '(415) 555-0132', '(212) 555-0187', '(312) 555-0144')
USER = 'My name is David Chen. My email is david.chen@example.com. My SSN is 123-45-6789. Please acknowledge my request.'
SCENARIOS = [
    ('secure_chat', '/api/v1/chat', {'message': USER}),
    ('secure_rag', '/api/v1/rag/chat', {'message': 'What is the status of claim CLM-1001?'}),
    ('secure_agent', '/api/v1/agent/chat', {'message': 'What is the status of claim CLM-1001?', 'role': 'claims_adjuster'}),
    ('multi_step_agent', '/api/v1/agent/chat', {'message': 'Check claim CLM-1001 and tell me the relevant privacy procedure.',
                                                'role': 'claims_adjuster'}),
    ('unauthorized_agent', '/api/v1/agent/chat', {'message': 'What is the status of claim CLM-1001?', 'role': 'customer_service'}),
    ('approval_agent', '/api/v1/agent/chat', {'message': 'What is the status of claim CLM-1004?', 'role': 'claims_adjuster'}),
]


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('ollama', 'foundry'), default='ollama')
    args = parser.parse_args()
    current = {'scenario': None}
    jsonrpc, executions, boundary, tool_results = [], [], {'requests': 0}, []

    original_send = JSONRPCDispatcher.send_raw_request

    async def spy(self, method, params, opts=None, **kwargs):
        entry = {'scenario': current['scenario'], 'method': method}
        if method == 'tools/call':
            entry.update(tool=params.get('name'), validated_arguments=params.get('arguments'))
        jsonrpc.append(entry)
        return await original_send(self, method, params, opts, **kwargs)
    JSONRPCDispatcher.send_raw_request = spy

    with tempfile.TemporaryDirectory(prefix='secureagent-phase8-') as directory:
        settings = replace(load_settings(), llm_provider=args.provider, audit_log_path=str(Path(directory) / 'audit.jsonl'),
                           graph_trace_log_path=str(Path(directory) / 'graph_trace.jsonl'))
        service = build_service(settings)
        assert service.llm_health().value == 'available', 'Selected local model is not loaded/available'

        def inspect(request):
            if request.url.path not in ('/api/chat', '/v1/chat/completions'):
                return
            visible = normalize_input(json.dumps(json.loads(request.content), ensure_ascii=False))
            assert all(value.casefold() not in visible.casefold() for value in RAW), 'Raw PII at provider boundary'
            boundary['requests'] += 1
            for message in json.loads(request.content).get('messages', []):
                if message.get('role') == 'tool':
                    tool_results.append({'scenario': current['scenario'],
                                         'sanitized_result': json.loads(message['content'])['UNTRUSTED TOOL DATA']})
        service._llm._client.event_hooks['request'] = [inspect]

        host = service.tool_gateway.mcp.host

        def observe(name, handler):
            def run(arguments, session):
                stack = ''.join(traceback.format_stack())
                executions.append({'scenario': current['scenario'], 'tool': name.value,
                                   'inside_mcp_server_thread': 'AnyIO worker thread' in threading.current_thread().name,
                                   'dispatched_by_mcp_host': 'app/mcp/host.py' in stack,
                                   'graph_planner_on_stack': 'agent_planner' in stack})
                return handler(arguments, session)
            return run
        host.registry = ToolRegistry({name: observe(name, h) for name, h in host.registry._handlers.items()})

        port = free_port()
        server = uvicorn.Server(uvicorn.Config(create_app(settings, service), host='127.0.0.1', port=port,
                                               log_level='warning', access_log=False))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        results = {}
        try:
            deadline = time.monotonic() + 30
            while not server.started:
                assert thread.is_alive() and time.monotonic() < deadline, 'API failed to start'
                time.sleep(0.05)
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=300) as api:
                for name, path, body in SCENARIOS:
                    current['scenario'] = name
                    response = api.post(path, json={**body, 'include_orchestration': True})
                    data = response.json()
                    # No scenario requests re-identification, so no raw value may appear.
                    assert all(value not in response.text for value in RAW), 'PII in response'
                    outcome = {'http_status': response.status_code}
                    if response.status_code == 202:
                        outcome['approval'] = data['approval']
                        outcome['executions_before_decision'] = sum(e['scenario'] == name for e in executions)
                        decided = api.post(f"/api/v1/agent/approvals/{data['approval']['approval_id']}",
                                           json={'decision': 'approve', 'approver_role': 'supervisor'})
                        assert all(value not in decided.text for value in RAW), 'PII in response'
                        outcome['after_supervisor_approval'] = {'http_status': decided.status_code,
                                                                **_summary(decided.json())}
                    else:
                        outcome.update(_summary(data))
                    outcome['mcp_tools_called'] = [e['tool'] for e in jsonrpc if e['scenario'] == name and e['method'] == 'tools/call']
                    outcome['registry_executions'] = [e['tool'] for e in executions if e['scenario'] == name]
                    results[name] = outcome
            if service.tool_calling_supported:
                current['scenario'] = 'ui_approval_demo'
                results['ui_approval_demo'] = _ui_approval_demo(f'http://127.0.0.1:{port}')
        finally:
            server.should_exit = True
            thread.join(timeout=15)
            JSONRPCDispatcher.send_raw_request = original_send

        audit = (Path(directory) / 'audit.jsonl').read_text()
        trace = (Path(directory) / 'graph_trace.jsonl').read_text()
        for log in (audit, trace):
            assert all(value not in log for value in RAW), 'PII in audit/trace log'
            assert all(marker not in log for marker in ('<PERSON_', '<TOOL_', '<DOC_', 'UNTRUSTED', 'CLM-'))
        trace_events = [json.loads(line) for line in trace.splitlines()]
        nodes_seen = sorted({e['node'] for e in trace_events if 'node' in e})
        assert all(e['inside_mcp_server_thread'] and e['dispatched_by_mcp_host'] and not e['graph_planner_on_stack']
                   for e in executions), 'A tool ran outside an MCP server'
        assert len(executions) == sum(e['method'] == 'tools/call' for e in jsonrpc), 'Execution without MCP tools/call'
        print(json.dumps({
            'provider': args.provider, 'model': service.model_name, 'tool_calling': service.tool_calling_supported,
            'langgraph': {'trace_events': len(trace_events), 'nodes_executed': nodes_seen,
                          'graph_completed_events': sum(e['event'] == 'GRAPH_COMPLETED' for e in trace_events)},
            'mcp': {'sdk': 'mcp (official Python SDK)', 'transport': MCP_TRANSPORT + ' JSON-RPC 2.0 over memory streams',
                    'jsonrpc_methods': _counts([e['method'] for e in jsonrpc]),
                    'tool_calls': [{k: e[k] for k in ('scenario', 'tool', 'validated_arguments')}
                                   for e in jsonrpc if e['method'] == 'tools/call'],
                    'every_execution_inside_mcp_server': True, 'planner_never_called_tools_directly': True,
                    'sanitized_results_seen_by_model': tool_results},
            'generation_boundary': {'provider_requests': boundary['requests'], 'raw_pii_found': False},
            'audit_and_trace_privacy': 'PASS', 'scenarios': results}, indent=2))


def _ui_approval_demo(backend_url):
    """Real Streamlit widgets -> real API: pause on CLM-1004, approve, render metadata."""
    from streamlit.testing.v1 import AppTest
    previous = os.environ.get('UI_BACKEND_URL')
    os.environ['UI_BACKEND_URL'] = backend_url
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app/ui/app.py')).run(timeout=30)
        app.selectbox(key='mode').set_value('Secure Agent').run(timeout=30)
        app.selectbox(key='demo_role').set_value('claims_adjuster').run(timeout=30)
        app.chat_input[0].set_value('What is the status of claim CLM-1004?').run(timeout=300)
        assert not app.exception and app.session_state['conversation'].messages[-1].pending is not None
        approve = next(button for button in app.button if button.key.startswith('approve-'))
        approve.click().run(timeout=300)
        last = app.session_state['conversation'].messages[-1]
        assert not app.exception and last.received is not None, 'UI approval did not complete'
        rendered = [t.value for t in app.text]
        assert all(value not in ' '.join(rendered) for value in RAW)
        return {'paused_in_ui': True, 'approved_in_ui': True,
                'orchestration_panel': [v for v in rendered if v.startswith(('Orchestration:', 'Route:', 'MCP servers:',
                                                                             'Human approval:', 'Tool protocol:'))]}
    finally:
        if previous is None:
            os.environ.pop('UI_BACKEND_URL', None)
        else:
            os.environ['UI_BACKEND_URL'] = previous


def _summary(data):
    if 'error' in data:
        return {'error_code': data['error']['code']}
    summary = {key: data[key] for key in ('tools_used', 'agent_steps') if key in data}
    summary['orchestration'] = data.get('orchestration')
    summary['security'] = data.get('security')
    summary['response_excerpt'] = data.get('response', '')[:240]
    if data.get('sources'):
        summary['sources'] = sorted({s['filename'] for s in data['sources']})
    return summary


def _counts(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


if __name__ == '__main__':
    main()
