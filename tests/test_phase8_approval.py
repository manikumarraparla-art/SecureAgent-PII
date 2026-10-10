"""Phase 8: LangGraph human-in-the-loop approval (demonstration approval, not identity) and API contract."""
import json

import pytest
from fastapi.testclient import TestClient

from app.agent.schemas import Role
from app.audit.events import AuditEventType
from app.core.config import Settings
from app.core.exceptions import SecureAgentError
from app.main import create_app
from tests.phase8_support import agent, build, call, events, turn

RESTRICTED = [turn([call(arguments={'claim_id': 'CLM-1004'})]),
              turn(content='Claim is UNDER SPECIAL INVESTIGATION for <TOOL_PERSON_1>.')]
ASK = {'message': 'What is the status of claim CLM-1004?', 'role': 'claims_adjuster', 'include_orchestration': True}


def client_for(h):
    return TestClient(create_app(Settings(), h.service))


def pause(client):
    response = client.post('/api/v1/agent/chat', json=ASK)
    assert response.status_code == 202, response.text
    return response.json()


def test_restricted_record_pauses_before_mcp_execution(gateway):
    h = build(gateway, RESTRICTED)
    with client_for(h) as client:
        body = pause(client)
        assert len(h.service.runtime.approvals) == 1
        stored = json.dumps([c['channel_values'] for c in _checkpoints(h)], default=repr)
    assert body['status'] == 'APPROVAL_REQUIRED' and set(body) == {'request_id', 'status', 'approval', 'model'}
    assert body['approval']['tool'] == 'get_claim_status' and body['approval']['record_id'] == 'CLM-1004'
    assert body['approval']['approver_role'] == 'supervisor' and len(body['approval']['approval_id']) == 32
    assert 'response' not in body and 'Elena Rossi' not in json.dumps(body)
    assert h.mcp.called == [] and h.executed == []  # graph paused before the MCP call
    assert 'HUMAN_APPROVAL_REQUIRED' in events(h.audit) and 'AGENT_COMPLETED' not in events(h.audit)
    assert stored and 'Elena Rossi' not in stored and 'TokenVault' not in stored
    assert len(h.service.runtime.approvals) == 0  # app shutdown destroyed the paused run


def _checkpoints(h):
    saver = h.service.runtime.checkpointer
    return [saver.get_tuple({'configurable': {'thread_id': thread}}).checkpoint for thread in saver.storage]


def test_supervisor_approval_resumes_graph_and_executes_once(gateway):
    h = build(gateway, RESTRICTED)
    with client_for(h) as client:
        approval_id = pause(client)['approval']['approval_id']
        decided = client.post(f'/api/v1/agent/approvals/{approval_id}',
                              json={'decision': 'approve', 'approver_role': 'supervisor'})
        assert decided.status_code == 200, decided.text
        reused = client.post(f'/api/v1/agent/approvals/{approval_id}',
                             json={'decision': 'approve', 'approver_role': 'supervisor'})
    body = decided.json()
    assert body['tools_used'] == ['get_claim_status'] and 'Elena Rossi' not in decided.text
    assert '<TOOL_PERSON_1>' in body['response'] and 'Elena Rossi' not in h.fake.wire
    assert h.executed == [('get_claim_status', {'claim_id': 'CLM-1004'})] and h.mcp.called == ['get_claim_status']
    assert reused.status_code == 404 and reused.json()['error']['code'] == 'APPROVAL_NOT_FOUND'  # single use
    order = events(h.audit)
    assert order.index('HUMAN_APPROVAL_GRANTED') < order.index('MCP_TOOL_REQUESTED') and order[-1] == 'AGENT_COMPLETED'
    assert len(h.service.runtime.approvals) == 0 and not h.service.runtime.checkpointer.storage


def test_denied_approval_never_executes(gateway):
    h = build(gateway, RESTRICTED)
    with client_for(h) as client:
        approval_id = pause(client)['approval']['approval_id']
        denied = client.post(f'/api/v1/agent/approvals/{approval_id}',
                             json={'decision': 'deny', 'approver_role': 'supervisor'})
    assert denied.status_code == 403 and denied.json()['error']['code'] == 'APPROVAL_DENIED'
    assert h.executed == [] and h.mcp.called == []
    assert 'HUMAN_APPROVAL_DENIED' in events(h.audit) and events(h.audit)[-1] == 'AGENT_FAILED'
    assert not h.service.runtime.checkpointer.storage


def test_only_supervisor_may_decide_and_wrong_role_does_not_consume(gateway):
    h = build(gateway, RESTRICTED)
    with client_for(h) as client:
        approval_id = pause(client)['approval']['approval_id']
        for role in ('claims_adjuster', 'customer_service'):
            response = client.post(f'/api/v1/agent/approvals/{approval_id}', json={'decision': 'approve', 'approver_role': role})
            assert response.status_code == 403 and response.json()['error']['code'] == 'APPROVER_NOT_PERMITTED'
        assert h.executed == [] and len(h.service.runtime.approvals) == 1
        ok = client.post(f'/api/v1/agent/approvals/{approval_id}', json={'decision': 'approve', 'approver_role': 'supervisor'})
    assert ok.status_code == 200 and len(h.executed) == 1


@pytest.mark.parametrize('path,body', [
    ('/api/v1/agent/approvals/' + 'a' * 32, {'decision': 'approve', 'approver_role': 'supervisor'}),
    ('/api/v1/agent/approvals/not-an-id', {'decision': 'approve', 'approver_role': 'supervisor'}),
    ('/api/v1/agent/approvals/' + 'a' * 32, {'decision': 'maybe', 'approver_role': 'supervisor'}),
    ('/api/v1/agent/approvals/' + 'a' * 32, {'decision': 'approve', 'approver_role': 'root'}),
    ('/api/v1/agent/approvals/' + 'a' * 32, {'decision': 'approve', 'approver_role': 'supervisor', 'tool': 'x'}),
])
def test_invalid_or_unknown_approvals_are_rejected_safely(gateway, path, body):
    h = build(gateway, RESTRICTED)
    with client_for(h) as client:
        response = client.post(path, json=body)
    assert response.status_code in (404, 422) and set(response.json()) == {'request_id', 'error'}


def test_expired_approval_is_cleared(gateway, monkeypatch):
    h = build(gateway, RESTRICTED)
    with client_for(h) as client:
        approval_id = pause(client)['approval']['approval_id']
        paused = h.service.runtime.approvals.get(approval_id)
        context = paused.context
        paused.expires_at = 0
        response = client.post(f'/api/v1/agent/approvals/{approval_id}', json={'decision': 'approve', 'approver_role': 'supervisor'})
    assert response.status_code == 404 and h.executed == []
    assert len(context.vault) == 0 and len(context.session.vault) == 0  # mappings destroyed
    assert not h.service.runtime.checkpointer.storage


def test_pending_capacity_is_bounded(gateway):
    h = build(gateway, RESTRICTED, max_pending_approvals=1)
    with client_for(h) as client:
        first = pause(client)['approval']['approval_id']
        h.fake.payloads.clear()
        second = pause(client)['approval']['approval_id']
        evicted = client.post(f'/api/v1/agent/approvals/{first}', json={'decision': 'approve', 'approver_role': 'supervisor'})
        kept = client.post(f'/api/v1/agent/approvals/{second}', json={'decision': 'approve', 'approver_role': 'supervisor'})
    assert evicted.status_code == 404 and kept.status_code == 200 and len(h.service.runtime.checkpointer.storage) == 0


def test_supervisor_reads_restricted_record_without_approval(gateway):
    h = build(gateway, RESTRICTED)
    result = agent(h, 'What is the status of claim CLM-1004?', Role.SUPERVISOR)
    assert result['tools_used'] == ['get_claim_status'] and result['orchestration']['approval'] == 'NOT_REQUIRED'
    assert 'human_approval' not in result['orchestration']['nodes']


def test_approval_cannot_override_authorization(gateway):
    h = build(gateway, RESTRICTED)
    with pytest.raises(SecureAgentError) as exc:
        agent(h, 'What is the status of claim CLM-1004?', Role.CUSTOMER_SERVICE)
    assert exc.value.error_code.value == 'TOOL_DENIED'
    assert 'HUMAN_APPROVAL_REQUIRED' not in events(h.audit) and len(h.service.runtime.approvals) == 0


def test_approval_path_is_visible_in_orchestration_metadata(gateway):
    h = build(gateway, RESTRICTED)
    pending = agent(h, 'What is the status of claim CLM-1004?')
    result = h.service.decide_approval(pending['approval']['approval_id'], 'approve', Role.SUPERVISOR)
    assert result['orchestration']['approval'] == 'GRANTED'
    assert result['orchestration']['nodes'][:7] == ['input_guard', 'router', 'tool_discovery', 'agent_planner',
                                                     'policy_gate', 'human_approval', 'mcp_tool']


def test_gateway_refuses_restricted_call_without_recorded_approval(gateway):
    from app.security.tool_results import ToolSecuritySession
    from app.agent.schemas import ToolName
    h = build(gateway)
    session = ToolSecuritySession('direct', gateway, 16000)
    with pytest.raises(SecureAgentError) as exc:
        h.service.tool_gateway.execute((ToolName.CLAIM, {'claim_id': 'CLM-1004'}), session, lambda *a, **k: None, 1,
                                       Role.CLAIMS_ADJUSTER, {'claim_id': {'CLM-1004'}, 'policy_id': set()})
    assert exc.value.error_code.value == 'TOOL_DENIED' and h.executed == []


# ------------------------------------------------------------------ API metadata contract

def test_orchestration_metadata_is_opt_in_and_safe(gateway):
    h = build(gateway, [turn([call()]), turn(content='Status for <TOOL_PERSON_1>.')] * 2)
    with client_for(h) as client:
        plain = client.post('/api/v1/agent/chat', json={'message': 'CLM-1002', 'role': 'claims_adjuster'})
        rich = client.post('/api/v1/agent/chat', json={'message': 'CLM-1002', 'role': 'claims_adjuster',
                                                       'include_orchestration': True})
        chat = client.post('/api/v1/chat', json={'message': 'hello', 'include_orchestration': True})
    assert 'orchestration' not in plain.json()
    meta = rich.json()['orchestration']
    assert meta == {'engine': 'langgraph', 'route': 'agent', 'nodes': meta['nodes'], 'tool_protocol': 'mcp',
                    'mcp_transport': 'in-process', 'mcp_servers': ['claims'], 'approval': 'NOT_REQUIRED'}
    assert chat.json()['orchestration']['route'] == 'chat'
    for forbidden in ('messages', 'system', 'UNTRUSTED', 'arguments', 'claim_id', 'vault', 'thought', 'reasoning'):
        assert forbidden not in json.dumps(meta)


def test_agent_audit_records_approval_decisions_with_closed_schema(gateway):
    h = build(gateway, RESTRICTED)
    pending = agent(h, 'What is the status of claim CLM-1004?')
    h.service.decide_approval(pending['approval']['approval_id'], 'approve', Role.SUPERVISOR)
    granted = [e for e in h.audit.events if e.event is AuditEventType.HUMAN_APPROVAL_GRANTED]
    assert len(granted) == 1 and granted[0].authorized is True and granted[0].role is Role.CLAIMS_ADJUSTER
    assert 'CLM-1004' not in json.dumps([e.to_dict() for e in h.audit.events])


def test_expired_and_shutdown_pauses_are_destroyed(gateway):
    h = build(gateway, RESTRICTED * 2)
    first = agent(h, 'What is the status of claim CLM-1004?')
    context = h.service.runtime.approvals.get(first['approval']['approval_id']).context
    h.service.runtime.approvals.get(first['approval']['approval_id']).expires_at = 0
    h.fake.payloads.clear()
    second = agent(h, 'What is the status of claim CLM-1004?')  # any new run purges expired pauses
    assert len(context.vault) == 0 and len(h.service.runtime.approvals) == 1
    pending = h.service.runtime.approvals.get(second['approval']['approval_id']).context
    h.service.close()
    assert len(h.service.runtime.approvals) == 0 and len(pending.vault) == 0
    assert not h.service.runtime.checkpointer.storage
