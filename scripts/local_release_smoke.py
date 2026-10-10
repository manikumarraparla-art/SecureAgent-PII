"""Final release scenarios on real local providers, FastAPI, LangGraph and MCP.

Run: python -m scripts.local_release_smoke --provider ollama (or foundry).
No downloads or external process termination. Model requests stay real, including
negative cases. Fault injection is confined to a test-owned provider connection
and test-owned MCP handler; live daemons and application code are not modified.
Only closed status/metadata is printed, never prompts, answers or reasoning.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import socket
import tempfile
import threading
import time

import httpx
import uvicorn

from app.agent.schemas import ToolName
from app.agent.tool_registry import ToolRegistry
from app.core.config import load_settings
from app.main import build_service, create_app
from app.security.gateway import normalize_input
from scripts.check_runtime import inspect_runtime
from scripts.local_phase8_smoke import RAW

USER = ('My name is David Chen. My email is david.chen@example.com. My SSN is 123-45-6789. '
        'Repeat my name and email exactly and keep the SSN placeholder.')


@contextmanager
def serve(app):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False))
        thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 30
            while not server.started:
                assert thread.is_alive() and time.monotonic() < deadline, 'API startup failed'
                time.sleep(.05)
            with httpx.Client(base_url=f'http://127.0.0.1:{listener.getsockname()[1]}',
                              trust_env=False, timeout=240) as client:
                yield client
        finally:
            server.should_exit = True
            thread.join(timeout=15)
            assert not thread.is_alive(), 'API shutdown failed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('ollama', 'foundry'), default='ollama')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='secureagent-release-') as directory:
        settings = replace(load_settings(), llm_provider=args.provider,
                           audit_log_path=str(Path(directory) / 'audit.jsonl'),
                           graph_trace_log_path=str(Path(directory) / 'trace.jsonl'))
        service = build_service(settings)
        counts = {'requests': 0, 'user_tokens': False, 'document_tokens': False, 'tool_tokens': False}
        executions = []
        scenarios = {}

        def inspect(request):
            if request.url.path not in ('/api/chat', '/v1/chat/completions'): return
            visible = normalize_input(json.dumps(json.loads(request.content), ensure_ascii=False))
            assert all(v.casefold() not in visible.casefold() for v in RAW), 'Raw PII at model boundary'
            assert 'token_mapping' not in visible and 'original_value' not in visible
            counts['requests'] += 1
            counts['user_tokens'] |= all(t in visible for t in ('<PERSON_', '<EMAIL_', '<SSN_'))
            counts['document_tokens'] |= '<DOC_PERSON_' in visible
            counts['tool_tokens'] |= '<TOOL_PERSON_' in visible
        service._llm._client.event_hooks['request'] = [inspect]
        host = service.tool_gateway.mcp.host
        original = host.registry
        def observe(name, handler):
            def invoke(arguments, session):
                executions.append(name.value)
                return handler(arguments, session)
            return invoke
        host.registry = ToolRegistry({n: observe(n, h) for n, h in original._handlers.items()})

        with serve(create_app(settings, service)) as api:
            identity = inspect_runtime(api, args.provider)
            def post(name, path, body, status=200, code=None, restored=False):
                response = api.post(path, json=body)
                assert response.status_code == status, f'{name}: unexpected HTTP status {response.status_code}'
                data = response.json()
                assert '<think' not in response.text.lower() and '</think' not in response.text.lower()
                protected = RAW if not restored else tuple(v for v in RAW if v not in ('David Chen','david.chen@example.com'))
                assert all(v not in response.text for v in protected), f'{name}: protected PII in response'
                if code: assert data['error']['code'] == code, f'{name}: unexpected error code'
                if status == 200:
                    assert data['orchestration']['engine'] == 'langgraph'
                    assert data['orchestration']['nodes'][-2:] == ['output_guard', 'finalize']
                scenarios[name] = {'status': status, 'result': 'PASS'}
                if code: scenarios[name]['code'] = code
                return data

            base = {'message': USER, 'include_orchestration': True}
            chat = post('chat_tokenization', '/api/v1/chat', base)
            assert {'PERSON', 'EMAIL_ADDRESS', 'US_SSN'} <= set(chat['security']['entity_types'])
            restored = post('permitted_restoration_ssn_blocked', '/api/v1/chat',
                            {**base, 'allowed_reidentification_types': ['PERSON','EMAIL_ADDRESS','US_SSN']}, restored=True)
            assert 'David Chen' in restored['response'] and 'david.chen@example.com' in restored['response']
            # Decision counts only tokens the model actually returned. If it omits
            # SSN, ALLOWED is correct for the returned name/email tokens.
            assert restored['security']['reidentification_decision'] in ('ALLOWED', 'PARTIAL')
            assert not service.effective_policy(['PERSON','EMAIL_ADDRESS','US_SSN']).permits('US_SSN')
            rag = post('rag_document_protection', '/api/v1/rag/chat',
                       {'message': 'What is the status of claim CLM-1001?', 'include_orchestration': True, 'top_k': 3})
            assert rag['sources'] and rag['security']['document_pii_detected']

            before = counts['requests']
            for route, path in (('chat', '/api/v1/chat'), ('rag', '/api/v1/rag/chat'), ('agent', '/api/v1/agent/chat')):
                post('malformed_' + route, path, {**base, 'include_orchestration': 'true'},
                     422, 'INVALID_REQUEST')
                post('extra_key_' + route, path, {**base, 'david.chen@example.com': 'synthetic'},
                     422, 'INVALID_REQUEST')
            assert counts['requests'] == before
            agent = {'message': 'What is the status of claim CLM-1001?', 'role': 'claims_adjuster',
                     'include_orchestration': True, 'include_sources': True}
            if service.tool_calling_supported:
                result = post('authorized_mcp', '/api/v1/agent/chat', agent)
                assert result['tools_used'] == ['get_claim_status'] and result['security']['tool_pii_detected']
                assert result['orchestration']['tool_protocol'] == 'mcp'
                before = len(executions)
                post('unauthorized_mcp', '/api/v1/agent/chat', {**agent, 'role':'customer_service'}, 403, 'TOOL_DENIED')
                post('bulk_record_scope', '/api/v1/agent/chat',
                     {**agent, 'message': 'Look up CLM-1001 and CLM-1002.'}, 403, 'TOOL_DENIED')
                assert len(executions) == before
                for decision in ('approve', 'deny'):
                    before = len(executions)
                    pending = post('pause_' + decision, '/api/v1/agent/chat',
                                   {**agent, 'message': 'What is the status of claim CLM-1004?'}, 202)
                    assert pending['status'] == 'APPROVAL_REQUIRED' and len(executions) == before
                    result = post('approval_' + decision, '/api/v1/agent/approvals/' + pending['approval']['approval_id'],
                                  {'decision': decision, 'approver_role':'supervisor'},
                                  200 if decision == 'approve' else 403,
                                  None if decision == 'approve' else 'APPROVAL_DENIED')
                    assert len(executions) == before + (1 if decision == 'approve' else 0)
                    if decision == 'approve': assert result['orchestration']['approval'] == 'GRANTED'

                registry = host.registry
                def broken(arguments, session):
                    raise RuntimeError('Synthetic private error david.chen@example.com')
                host.registry = ToolRegistry({**registry._handlers, ToolName.CLAIM: broken})
                before = counts['requests']
                try:
                    post('mcp_handler_failure', '/api/v1/agent/chat', agent, 500, 'TOOL_FAILED')
                    assert counts['requests'] == before + 1, 'Failed tool was followed by generation'
                finally:
                    host.registry = registry
            else:
                before = counts['requests']
                post('foundry_agent_capability', '/api/v1/agent/chat', agent, 409, 'LLM_CAPABILITY_UNSUPPORTED')
                assert counts['requests'] == before and not executions

            # Reserve a non-listening loopback socket to force a real connection failure.
            # This affects only this isolated service; never stop the user's model daemon.
            original_client = service._llm._client
            with socket.socket() as unavailable:
                unavailable.bind(('127.0.0.1', 0))
                with httpx.Client(base_url=f'http://127.0.0.1:{unavailable.getsockname()[1]}',
                                  trust_env=False, timeout=2) as failed_client:
                    service._llm._client = failed_client
                    try: post('provider_unavailable', '/api/v1/chat', base, 503, 'LLM_UNAVAILABLE')
                    finally: service._llm._client = original_client

        for name in ('audit.jsonl', 'trace.jsonl'):
            log = (Path(directory) / name).read_text()
            assert all(v not in log for v in RAW), 'PII in audit/trace'
            assert all(t not in log for t in ('<think', '<PERSON_', '<DOC_', '<TOOL_', 'UNTRUSTED'))
        assert counts['user_tokens'] and counts['document_tokens']
        if args.provider == 'ollama': assert counts['tool_tokens']
        print(json.dumps({'result': 'PASS', 'runtime': identity, 'scenarios': scenarios,
                          'generation_boundary': counts, 'audit_trace_privacy': 'PASS',
                          'fault_injection': ['test-owned provider connection', 'test-owned MCP handler (Ollama)']}, indent=2))


if __name__ == '__main__':
    main()
