"""Release identity and stale-process diagnostics; no generation service needed."""
import copy
import json
from pathlib import Path

import httpx
import pytest
import toml
from streamlit.testing.v1 import AppTest

from app.main import create_app
from app.mcp.servers._common import SERVER_VERSION
from app.version import VERSION
from scripts.check_runtime import inspect_runtime
from secureagent_ui.api_client import APIClient, BackendHealth

ROOT = Path(__file__).resolve().parents[1]


def test_release_identity_matches_live_health_openapi_and_mcp(make_harness):
    h = make_harness()
    assert inspect_runtime(h.client, 'ollama')['version'] == VERSION
    assert h.client.get('/openapi.json').json()['info']['version'] == VERSION
    assert toml.loads((ROOT / 'pyproject.toml').read_text())['project']['version'] == SERVER_VERSION == VERSION
    assert not h.fake.chat_payloads


@pytest.mark.parametrize('defect', ['old_version', 'missing_version', 'wrong_provider', 'degraded',
                                   'missing_flag', 'weak_validation', 'malformed', 'unavailable'])
def test_runtime_check_rejects_stale_or_incompatible_backend_without_echoing_body(defect):
    schema = copy.deepcopy(create_app().openapi())
    health = {'version': VERSION, 'status': 'healthy', 'provider': 'ollama', 'llm': 'available',
              'ollama': 'available', 'tool_calling': True, 'model': 'qwen2.5:7b'}
    if defect == 'old_version': schema['info']['version'] = '0.5.0'
    if defect == 'missing_version': health.pop('version')
    if defect == 'wrong_provider': health['provider'] = 'foundry'
    if defect == 'degraded': health['status'] = 'degraded'
    if defect == 'missing_flag': schema['components']['schemas']['ChatRequest']['properties'].pop('include_orchestration')
    if defect == 'weak_validation': schema['components']['schemas']['AgentChatRequest']['additionalProperties'] = True
    seen = []
    def handler(request):
        seen.append((request.method, request.url.path))
        if defect == 'unavailable': raise httpx.ConnectError('private@example.com')
        if defect == 'malformed': return httpx.Response(200, text='private@example.com')
        return httpx.Response(200, json=health if request.url.path == '/health' else schema)
    with httpx.Client(base_url='http://127.0.0.1', transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError) as exc: inspect_runtime(client, 'ollama')
    assert 'private@example.com' not in str(exc.value)
    assert all(method == 'GET' for method, path in seen)


def test_ui_displays_validated_backend_version(monkeypatch):
    monkeypatch.setattr(APIClient, 'health', lambda self: BackendHealth(True, True, version=VERSION))
    ui = AppTest.from_file(ROOT / 'app/ui/app.py').run(timeout=20)
    assert not ui.exception
    assert any(c.value == f'Backend version: {VERSION}' for c in ui.caption)


def test_ui_does_not_display_arbitrary_version_text():
    def handler(request):
        return httpx.Response(200, json={'version': 'private@example.com', 'status': 'healthy',
                                        'ollama': 'available', 'model': 'qwen2.5:7b'})
    with APIClient(transport=httpx.MockTransport(handler)) as client:
        health = client.health()
    assert not health.backend_available and health.version == 'unknown'


@pytest.mark.parametrize('path', ['/api/v1/chat', '/api/v1/rag/chat', '/api/v1/agent/chat'])
def test_unknown_json_keys_cannot_echo_pii_in_validation_errors(path, make_harness, caplog):
    h = make_harness()
    values = ('synthetic.person@example.com', '123-45-6789', 'David Chen')
    response = h.client.post(path, json={'message': 'Synthetic check', **{value: 'ignored' for value in values}})
    assert response.status_code == 422
    assert response.json()['error']['fields'] == [
        {'loc': ['body', 'unknown_field'], 'type': 'extra_forbidden'} for _ in values]
    assert all(value not in response.text + caplog.text for value in values)
    assert not h.fake.chat_payloads and not h.audit.events
