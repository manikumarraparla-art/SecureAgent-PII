"""Real Streamlit widgets -> UI client -> TCP -> FastAPI -> LangGraph/MCP.

Only generation and retrieval data are scripted; validation, security, graph,
policy and MCP execution are real. No UI client or API route is mocked.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from streamlit.testing.v1 import AppTest

from app.api.errors import register_error_handlers
from app.core.config import Settings
from app.main import create_app
from secureagent_ui.api_client import APIClient, FailureCode, MESSAGES, UIFailure, safe_error
from secureagent_ui.contracts import ENDPOINTS, Mode
from tests.phase8_support import USER_VALUES, TOOL_VALUES, build
from tests.test_rag import FixedRetriever

APP = Path(__file__).resolve().parents[1] / 'app/ui/app.py'
MESSAGE = ('My name is David Chen and my email is '
           '[david.chen@example.com](mailto:david.chen@example.com). '
           'Please write a short confirmation that you received my claim inquiry.')


@contextmanager
def live_api(app):
    # Reserve the socket until Uvicorn takes ownership; no free-port race.
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False))
        thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 15
            while not server.started:
                assert thread.is_alive() and time.monotonic() < deadline, 'API startup failed'
                time.sleep(.01)
            yield f'http://127.0.0.1:{listener.getsockname()[1]}'
        finally:
            server.should_exit = True
            thread.join(timeout=15)
            assert not thread.is_alive(), 'API shutdown failed'


@pytest.mark.parametrize('mode', list(Mode))
def test_streamlit_request_contract_reaches_real_graph(mode, gateway, monkeypatch):
    h = build(gateway, retriever=FixedRetriever())
    api = create_app(Settings(), h.service)
    requests = []

    @api.middleware('http')
    async def capture(request, call_next):
        if request.method == 'POST':
            body = await request.json()
            response = await call_next(request)
            # Record contract/options only. Never retain the raw input message.
            requests.append((request.url.path, set(body),
                             {k: v for k, v in body.items() if k != 'message'}, response.status_code))
            return response
        return await call_next(request)

    with live_api(api) as url:
        monkeypatch.setenv('UI_BACKEND_URL', url)
        ui = AppTest.from_file(APP).run(timeout=20)
        ui.selectbox(key='mode').set_value(mode.value).run()
        if mode is Mode.AGENT:
            ui.selectbox(key='demo_role').set_value('claims_adjuster').run()
        if mode is Mode.RAG:
            ui.slider(key='top_k').set_value(2).run()
        ui.chat_input[0].set_value(MESSAGE + (' Check claim CLM-1002.' if mode is Mode.AGENT else '')).run(timeout=30)
        assert not ui.exception and not ui.error
        received = ui.session_state['conversation'].messages[-1].received
        assert received is not None, 'Real UI request failed'
        result = received.result
        route = {Mode.CHAT: 'chat', Mode.RAG: 'rag', Mode.AGENT: 'agent'}[mode]
        assert result.orchestration.engine == 'langgraph' and result.orchestration.route == route
        assert result.orchestration.nodes[:2] == ['input_guard', 'router']
        assert result.orchestration.nodes[-2:] == ['output_guard', 'finalize']
        options = {'allowed_reidentification_types': [], 'include_orchestration': True}
        if mode is Mode.RAG:
            options['top_k'] = 2
            assert 'secure_rag' in result.orchestration.nodes and result.sources
            assert result.security.document_pii_detected
        elif mode is Mode.AGENT:
            options.update(role='claims_adjuster', include_sources=True)
            assert result.tools_used == ['get_claim_status']
            assert result.orchestration.tool_protocol == 'mcp'
            assert h.mcp.called == ['get_claim_status']
            assert 'policy_gate' in result.orchestration.nodes and 'tool_result_guard' in result.orchestration.nodes
        else:
            assert result.orchestration.nodes == ['input_guard', 'router', 'secure_chat', 'output_guard', 'finalize']
        assert requests == [(ENDPOINTS[mode], {'message', *options}, options, 200)]
        assert {'PERSON', 'EMAIL_ADDRESS'} <= set(result.security.entity_types)
        assert h.fake.payloads
        assert all(value not in h.fake.wire for value in USER_VALUES + TOOL_VALUES)
        assert all(value not in result.response for value in USER_VALUES + TOOL_VALUES)
        assert any(e.label == 'Orchestration' for e in ui.expander)

        # Real validation still rejects extra fields and non-boolean flags,
        # and rejected requests must not reach the model or the graph.
        before = (len(h.fake.payloads), len(h.trace.events))
        with httpx.Client(base_url=url, trust_env=False) as client:
            for invalid, field, kind in [({'unexpected': 'private@example.com'}, 'unknown_field', 'extra_forbidden'),
                                         ({'include_orchestration': 'true'}, 'include_orchestration', 'bool_type')]:
                response = client.post(ENDPOINTS[mode], json={'message': MESSAGE, **options, **invalid})
                assert response.status_code == 422
                assert response.json()['error']['fields'] == [{'loc': ['body', field], 'type': kind}]
                assert all(value not in response.text for value in USER_VALUES + ('private@example.com',))
        assert before == (len(h.fake.payloads), len(h.trace.events))


class LegacyChatRequest(BaseModel):
    """The observed 0.5.0 wire contract: orchestration is an extra field."""
    model_config = ConfigDict(extra='forbid')
    message: str
    allowed_reidentification_types: list[str] = []


@pytest.mark.parametrize('mode', list(Mode))
def test_stale_backend_rejection_is_actionable_without_retry(mode):
    api = FastAPI()
    register_error_handlers(api)
    calls = []

    @api.post(ENDPOINTS[mode])
    def legacy(body: LegacyChatRequest):
        pytest.fail('Rejected request must not execute the old pipeline')

    @api.middleware('http')
    async def capture(request, call_next):
        response = await call_next(request)
        calls.append(response.status_code)
        return response

    with live_api(api) as url, APIClient(url) as client:
        with pytest.raises(UIFailure) as exc:
            client.send(mode, MESSAGE)
    assert exc.value.code is FailureCode.BACKEND_INCOMPATIBLE
    assert 'Restart' in str(exc.value)
    assert all(value not in str(exc.value) for value in USER_VALUES)
    assert calls == [422]  # Never strip the flag or retry against the old pipeline.


@pytest.mark.parametrize('fields', [None, {}, 'private@example.com', [None],
    [{'loc': ['body', 'message'], 'type': 'extra_forbidden'}],
    [{'loc': ['body', 'include_orchestration'], 'type': 'bool_type'}]])
def test_other_validation_failures_keep_the_generic_safe_message(fields):
    code = safe_error(422, {'error': {'code': 'INVALID_REQUEST', 'message': 'private@example.com', 'fields': fields}})
    assert code is FailureCode.INVALID_REQUEST
    assert 'private@example.com' not in MESSAGES[code]
