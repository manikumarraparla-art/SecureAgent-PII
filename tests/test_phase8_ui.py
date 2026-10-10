"""Phase 8 UI: safe orchestration visibility and the demonstration approval flow."""
import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from app.ui.api_client import APIClient, BackendHealth, FailureCode, PendingApproval, ReceivedResponse, UIFailure, safe_error
from app.ui.components import orchestration_rows
from app.ui.contracts import APIResponse, ApprovalResponse, Mode
from pydantic import ValidationError

APP = Path(__file__).resolve().parents[1] / 'app/ui/app.py'
APPROVAL_ID = 'b' * 32
ORCHESTRATION = {'engine': 'langgraph', 'route': 'agent', 'tool_protocol': 'mcp', 'mcp_transport': 'in-process',
                 'mcp_servers': ['claims'], 'approval': 'NOT_REQUIRED',
                 'nodes': ['input_guard', 'router', 'tool_discovery', 'agent_planner', 'policy_gate', 'mcp_tool',
                           'tool_result_guard', 'agent_planner', 'output_guard', 'finalize']}
RESULT = {'request_id': 'request-id', 'response': 'Claim status: APPROVED.', 'model': 'qwen2.5:7b',
          'security': {'pii_detected': False, 'entity_types': [], 'reidentification_decision': 'NOT_REQUIRED',
                       'tool_pii_detected': True}, 'sources': [], 'tools_used': ['get_claim_status'],
          'agent_steps': 2, 'orchestration': ORCHESTRATION}
PENDING = {'request_id': 'request-id', 'status': 'APPROVAL_REQUIRED', 'model': 'qwen2.5:7b',
           'approval': {'approval_id': APPROVAL_ID, 'tool': 'get_claim_status', 'record_id': 'CLM-1004',
                        'reason': 'RESTRICTED_RECORD', 'approver_role': 'supervisor', 'expires_in_seconds': 300}}


def test_ui_requests_orchestration_metadata_and_handles_pause():
    seen = []

    def handler(request):
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(202, json=PENDING) if request.url.path.endswith('/chat') else httpx.Response(200, json=RESULT)
    with APIClient(transport=httpx.MockTransport(handler)) as client:
        pending = client.send(Mode.AGENT, 'CLM-1004', 'claims_adjuster')
        decided = client.decide(APPROVAL_ID, 'approve')
    assert isinstance(pending, PendingApproval) and pending.approval.approval.record_id == 'CLM-1004'
    assert seen[0][1]['include_orchestration'] is True
    assert seen[1] == (f'/api/v1/agent/approvals/{APPROVAL_ID}', {'decision': 'approve', 'approver_role': 'supervisor'})
    assert isinstance(decided, ReceivedResponse) and decided.result.orchestration.engine == 'langgraph'


@pytest.mark.parametrize('approval_id,decision', [('../../health', 'approve'), ('B' * 32, 'approve'),
                                                  (APPROVAL_ID + '/x', 'approve'), (APPROVAL_ID, 'maybe')])
def test_approval_path_is_built_only_from_validated_ids(approval_id, decision):
    seen = []
    with APIClient(transport=httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(200, json=RESULT))) as client:
        with pytest.raises(UIFailure) as exc:
            client.decide(approval_id, decision)
    assert exc.value.code is FailureCode.INVALID_REQUEST and seen == []


@pytest.mark.parametrize('change', [{'nodes': ['input_guard', 'chain_of_thought']}, {'engine': 'crewai'},
                                    {'mcp_servers': ['filesystem']}, {'approval': 'Maria Garcia'}])
def test_orchestration_metadata_is_strictly_validated(change):
    with pytest.raises(ValidationError):
        APIResponse.model_validate({**RESULT, 'orchestration': {**ORCHESTRATION, **change}})


def test_orchestration_rows_show_only_safe_names():
    rows = orchestration_rows(APIResponse.model_validate(RESULT).orchestration)
    assert ('Orchestration', 'LangGraph') in rows and ('MCP servers', 'claims') in rows
    assert ('Tool protocol', 'MCP (in-process)') in rows
    text = json.dumps(rows)
    for forbidden in ('system', 'prompt', 'thought', 'UNTRUSTED', 'TOOL_PERSON'):
        assert forbidden not in text


@pytest.mark.parametrize('code,expected', [('APPROVAL_DENIED', FailureCode.APPROVAL_DENIED),
                                           ('APPROVAL_NOT_FOUND', FailureCode.APPROVAL_EXPIRED),
                                           ('MCP_UNAVAILABLE', FailureCode.TOOLS_UNAVAILABLE)])
def test_new_error_codes_map_to_fixed_messages(code, expected):
    assert safe_error(403, {'error': {'code': code, 'message': 'Maria Garcia'}}) is expected


def test_malformed_pause_body_is_rejected():
    bad = {**PENDING, 'approval': {**PENDING['approval'], 'approval_id': 'not-hex'}}
    with APIClient(transport=httpx.MockTransport(lambda r: httpx.Response(202, json=bad))) as client:
        with pytest.raises(UIFailure) as exc:
            client.send(Mode.AGENT, 'CLM-1004', 'claims_adjuster')
    assert exc.value.code is FailureCode.INVALID_RESPONSE


def test_streamlit_approval_demo_flow(monkeypatch):
    decisions = []
    monkeypatch.setattr(APIClient, 'health', lambda s: BackendHealth(True, True))
    monkeypatch.setattr(APIClient, 'send', lambda *a, **k: PendingApproval(ApprovalResponse.model_validate(PENDING)))

    def decide(client, approval_id, decision):
        decisions.append((approval_id, decision))
        return ReceivedResponse(APIResponse.model_validate({**RESULT, 'orchestration': {**ORCHESTRATION, 'approval': 'GRANTED'}}), 90)
    monkeypatch.setattr(APIClient, 'decide', decide)
    app = AppTest.from_file(APP).run(timeout=15)
    app.selectbox(key='mode').set_value(Mode.AGENT.value).run()
    app.chat_input[0].set_value('What is the status of claim CLM-1004?').run()
    assert not app.exception and any('supervisor approval' in w.value for w in app.warning)
    app.button(key='approve-1').click().run()
    assert not app.exception and decisions == [(APPROVAL_ID, 'approve')]
    assert any(e.label == 'Orchestration' for e in app.expander)
    assert any('Granted (demo supervisor)' in t.value for t in app.text)
    assert not any(b.key == 'approve-1' for b in app.button)  # a decision is made once
