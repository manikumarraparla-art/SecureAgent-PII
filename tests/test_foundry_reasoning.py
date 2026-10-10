"""Foundry Local qwen3 reasoning blocks never leave the provider boundary."""
import json
import logging
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app.audit.logger import AuditLogger, InMemoryAuditSink
from app.core.config import Settings
from app.core.exceptions import LLMResponseError
from app.llm.foundry_client import FoundryLocalClient, final_answer
from app.llm.ollama_client import OllamaClient
from app.main import create_app
from app.security.secure_llm import SecureLLMService
from app.ui.api_client import APIClient, BackendHealth
from app.ui.contracts import Mode

MODEL = 'qwen3-1.7b-generic-gpu'
SECRET = 'REASONING-SECRET'
THINK = f'<think>\nThe user is <PERSON_1>. {SECRET} I should acknowledge.\nStep 2: be brief.\n</think>\n\nAcknowledged, <PERSON_1>.'


def foundry(content):
    def handler(request):
        if request.url.path == '/v1/models':
            return httpx.Response(200, json={'data': [{'id': MODEL}]})
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                                                      'message': {'role': 'assistant', 'content': content}}]})
    return FoundryLocalClient('http://127.0.0.1:5272', MODEL, transport=httpx.MockTransport(handler))


@pytest.mark.parametrize('content,expected', [
    ('Acknowledged, <PERSON_1>.', 'Acknowledged, <PERSON_1>.'),                      # normal response
    ('<think>Plan the reply.</think>Acknowledged.', 'Acknowledged.'),                 # inline block
    (THINK, 'Acknowledged, <PERSON_1>.'),                                             # multiline block
    ('<THINK id="r1">\nplan\n</Think >\n\nFinal.', 'Final.'),                          # tag variants
    ('Plan without an opening tag.\n</think>\nFinal answer.', 'Final answer.'),       # template omitted opener
    ('<think>a</think><think>b</think>\nFinal.', 'Final.'),                           # repeated blocks
    ('Claim CLM-1001 is APPROVED.', 'Claim CLM-1001 is APPROVED.'),
])
def test_final_answer_keeps_only_the_answer(content, expected):
    assert foundry(content).generate('Hello <PERSON_1>') == expected


@pytest.mark.parametrize('content', [
    '<think>\nreasoning that never closes ' + SECRET,     # missing closing tag
    'Answer. <think>trailing reasoning ' + SECRET,         # unclosed block after text
    '<think>only reasoning ' + SECRET + '</think>',        # no final answer
    '<think>x</think>\n   \n',
    '<think>a</think>Final <think>b',                      # malformed nesting
])
def test_malformed_reasoning_fails_closed_without_leaking(content):
    with pytest.raises(LLMResponseError) as exc:
        foundry(content).generate('Hello')
    assert SECRET not in str(exc.value) and exc.value.__cause__ is None


def test_normalization_is_foundry_only():
    seen = []

    def handler(request):
        if request.url.path == '/api/tags':
            return httpx.Response(200, json={'models': [{'name': 'qwen2.5:7b'}]})
        return httpx.Response(200, json={'message': {'role': 'assistant', 'content': '<think>kept</think> answer'},
                                          'done': True})
    ollama = OllamaClient(transport=httpx.MockTransport(handler))
    assert ollama.generate('Hello') == '<think>kept</think> answer'
    assert final_answer('plain text') == 'plain text'


@pytest.mark.parametrize('route', ['/api/v1/chat', '/api/v1/rag/chat'])
def test_reasoning_absent_from_api_audit_trace_and_logs(gateway, route, caplog):
    from tests.test_rag import FixedRetriever
    audit, trace = InMemoryAuditSink(), InMemoryAuditSink()
    service = SecureLLMService(gateway, foundry(THINK), AuditLogger([audit]), ['PERSON'],
                               retriever=FixedRetriever(), trace=AuditLogger([trace]))
    with caplog.at_level(logging.DEBUG), TestClient(create_app(Settings(), service)) as client:
        response = client.post(route, json={'message': 'My name is David Chen.', 'include_orchestration': True,
                                            'allowed_reidentification_types': ['PERSON']})
    assert response.status_code == 200
    assert response.json()['response'] == 'Acknowledged, David Chen.'
    logs = json.dumps([e.to_dict() for e in audit.events + trace.events]) + caplog.text + response.text
    for marker in (SECRET, '<think', '</think', 'Step 2'):
        assert marker not in logs


def test_malformed_reasoning_returns_safe_api_error(gateway):
    service = SecureLLMService(gateway, foundry('<think>never closed ' + SECRET), AuditLogger([]), [])
    with TestClient(create_app(Settings(), service)) as client:
        response = client.post('/api/v1/chat', json={'message': 'hello'})
    assert response.status_code == 502 and response.json()['error']['code'] == 'LLM_BAD_RESPONSE'
    assert SECRET not in response.text


def test_reasoning_absent_from_streamlit(gateway, monkeypatch):
    service = SecureLLMService(gateway, foundry(THINK), AuditLogger([]), ['PERSON'])
    with TestClient(create_app(Settings(), service)) as api:
        def forward(request):
            reply = api.post(request.url.path, content=request.content, headers={'content-type': 'application/json'})
            return httpx.Response(reply.status_code, content=reply.content, headers={'content-type': 'application/json'})
        with APIClient(transport=httpx.MockTransport(forward)) as client:
            received = client.send(Mode.CHAT, 'My name is David Chen.')
    assert SECRET not in received.result.response and '<think' not in received.result.response
    monkeypatch.setattr(APIClient, 'health', lambda s: BackendHealth(True, True, MODEL, 'foundry', False))
    monkeypatch.setattr(APIClient, 'send', lambda *a, **k: received)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app/ui/app.py')).run(timeout=15)
    app.chat_input[0].set_value('My name is David Chen.').run()
    rendered = ' '.join(t.value for t in app.text) + ' '.join(m.value for m in app.markdown)
    assert not app.exception and 'Acknowledged, <PERSON_1>.' in rendered
    assert SECRET not in rendered and '<think' not in rendered
