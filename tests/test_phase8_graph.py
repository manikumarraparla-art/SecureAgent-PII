"""Phase 8: LangGraph orchestration — compilation, routing, policy, limits, failures, isolation."""
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from app.agent.schemas import Role, ToolName
from app.agent.tool_registry import build_registry
from app.audit.events import AuditEventType, GraphTraceEvent, GraphTraceType, AuditStatus
from app.core.config import ConfigurationError, load_settings
from app.core.exceptions import (LLMCapabilityError, PIIDetectionError, PIILeakError, SecureAgentError)
from app.graph.builder import route_after_policy, route_after_router
from app.graph.names import GraphNode
from app.graph.runtime import recursion_limit_for
from app.rag.chunking import Chunk
from app.rag.vector_store import SearchHit
from tests.phase8_support import (INJECTION, TOOL_VALUES, USER, USER_VALUES, ScriptedNative, agent, answer, build,
                                  call, events, failed_nodes, tool_turn, turn)
from tests.test_rag import DOC, RAW, FixedRetriever

AGENT_PATH = ['input_guard', 'router', 'tool_discovery', 'agent_planner', 'policy_gate', 'mcp_tool',
              'tool_result_guard', 'agent_planner', 'output_guard', 'finalize']


class Knowledge:
    def __init__(self, text='Privacy procedure: verify identity before disclosing claim data.'):
        self.text, self.queries = text, []

    def retrieve(self, query, top_k):
        self.queries.append(query)
        return [SearchHit(Chunk(DOC, 'procedures_privacy.txt', DOC + '-0', self.text), 0.9)]

    def close(self):
        pass


# ------------------------------------------------------------------ compilation

def test_graph_compiles_with_every_node_and_conditional_edge(gateway):
    graph = build(gateway).service.runtime.graph.get_graph()
    assert set(graph.nodes) == {node.value for node in GraphNode} | {'__start__', '__end__'}
    conditional = {(e.source, e.target) for e in graph.edges if e.conditional}
    for edge in [('router', 'secure_chat'), ('router', 'secure_rag'), ('router', 'tool_discovery'),
                 ('agent_planner', 'policy_gate'), ('agent_planner', 'output_guard'),
                 ('policy_gate', 'mcp_tool'), ('policy_gate', 'human_approval'), ('policy_gate', 'safe_error'),
                 ('human_approval', 'mcp_tool'), ('mcp_tool', 'tool_result_guard'),
                 ('tool_result_guard', 'agent_planner'), ('tool_result_guard', 'mcp_tool')]:
        assert edge in conditional, edge
    # Planner can never reach MCP execution without passing the policy gate.
    assert ('agent_planner', 'mcp_tool') not in conditional
    assert 'policy_gate' in graph.draw_mermaid()  # offline rendering, no service


def test_edges_fail_closed_on_unknown_values():
    assert route_after_router({'route': 'shell'}) == 'safe_error'
    assert route_after_router({'route': 'agent', 'error_code': 'X'}) == 'safe_error'
    assert route_after_policy({'policy_decision': 'maybe'}) == 'safe_error'
    assert route_after_policy({'policy_decision': 'allow', 'error_code': 'TOOL_DENIED'}) == 'safe_error'


# ------------------------------------------------------------------ routing

def test_chat_route_through_graph(gateway):
    h = build(gateway, [turn(content='Hello <PERSON_1>.')])
    result = h.service.chat(USER, ['PERSON'], 'p8-chat', include_orchestration=True)
    assert result.orchestration['nodes'] == ['input_guard', 'router', 'secure_chat', 'output_guard', 'finalize']
    assert result.orchestration['route'] == 'chat' and result.orchestration['engine'] == 'langgraph'
    assert 'David Chen' in result.response
    assert all(value not in h.fake.wire for value in USER_VALUES)
    assert events(h.audit) == ['PII_SANITIZED', 'LLM_REQUEST_COMPLETED', 'REQUEST_COMPLETED']
    assert not h.mcp.listed and not h.mcp.called  # chat never touches MCP


def test_rag_route_through_graph_reuses_secure_retrieval(gateway):
    h = build(gateway, [turn(content='Grounded <DOC_PERSON_1>.')], retriever=FixedRetriever())
    result = h.service.rag_chat(USER, [], 'p8-rag', include_orchestration=True)
    assert result['orchestration']['nodes'] == ['input_guard', 'router', 'secure_rag', 'output_guard', 'finalize']
    assert result['security']['document_pii_detected'] and result['sources'][0]['filename'] == 'claim_1001.txt'
    assert '<DOC_PERSON_' in h.fake.wire
    assert all(value not in h.fake.wire for value in USER_VALUES + TOOL_VALUES[:3])
    assert not h.mcp.listed


def test_agent_route_executes_mcp_tool_exactly_once(gateway):
    h = build(gateway)
    result = agent(h)
    assert result['orchestration']['nodes'] == AGENT_PATH
    assert result['orchestration']['mcp_servers'] == ['claims'] and result['orchestration']['tool_protocol'] == 'mcp'
    assert h.executed == [('get_claim_status', {'claim_id': 'CLM-1002'})]
    assert h.mcp.called == ['get_claim_status'] and h.mcp.listed == 3
    assert result['tools_used'] == ['get_claim_status'] and result['agent_steps'] == 2


def test_router_is_not_an_authorization_point():
    import inspect
    from app.graph.nodes import SecureGraphNodes
    source = inspect.getsource(SecureGraphNodes.router)
    for decision in ('permits', 'authorize', 'scope', 'approval'):
        assert decision not in source


# ------------------------------------------------------------------ policy gate

def test_unauthorized_role_denied_before_mcp(gateway):
    h = build(gateway)
    with pytest.raises(SecureAgentError) as exc:
        agent(h, 'What is the status of claim CLM-1001?', Role.CUSTOMER_SERVICE)
    assert exc.value.error_code.value == 'TOOL_DENIED'
    assert h.executed == [] and h.mcp.called == []
    assert failed_nodes(h) == ['policy_gate']
    assert 'MCP_TOOL_REQUESTED' not in events(h.audit) and events(h.audit)[-1] == 'AGENT_FAILED'


def test_authorized_role_but_out_of_scope_record_denied_before_mcp(gateway):
    h = build(gateway, [turn([call(arguments={'claim_id': 'CLM-1001'})])])
    with pytest.raises(SecureAgentError) as exc:
        agent(h, 'What is the status of claim CLM-1002?')
    assert exc.value.error_code.value == 'TOOL_DENIED' and h.mcp.called == [] and h.executed == []


def test_claims_adjuster_allowed_through_policy_gate(gateway):
    h = build(gateway)
    agent(h)
    authorized = [e for e in h.audit.events if e.event is AuditEventType.TOOL_AUTHORIZED]
    mcp = [e for e in h.audit.events if e.event is AuditEventType.MCP_TOOL_AUTHORIZED]
    assert len(authorized) == 1 and authorized[0].role is Role.CLAIMS_ADJUSTER
    assert len(mcp) == 1 and mcp[0].mcp_server.value == 'claims'
    order = events(h.audit)
    assert order.index('TOOL_AUTHORIZED') < order.index('MCP_TOOL_REQUESTED') < order.index('MCP_TOOL_COMPLETED')


# ------------------------------------------------------------------ multi-step

def test_multi_step_claim_then_knowledge_authorizes_each_call(gateway):
    knowledge = Knowledge()
    h = build(gateway, [turn([call(arguments={'claim_id': 'CLM-1001'})]),
                        turn([call('search_knowledge', {'query': 'privacy procedure for claims'})]),
                        turn(content='Claim APPROVED; follow the privacy procedure.')], retriever=knowledge)
    result = agent(h, 'Check claim CLM-1001 and tell me the relevant privacy procedure.')
    assert result['tools_used'] == ['get_claim_status', 'search_knowledge'] and result['agent_steps'] == 3
    assert h.mcp.called == ['get_claim_status', 'search_knowledge']
    assert [name for name, _ in h.executed] == ['get_claim_status', 'search_knowledge']
    authorized = [e.tool_name for e in h.audit.events if e.event is AuditEventType.TOOL_AUTHORIZED]
    assert authorized == [ToolName.CLAIM, ToolName.KNOWLEDGE]
    assert sorted(result['orchestration']['mcp_servers']) == ['claims', 'knowledge']
    assert result['orchestration']['nodes'].count('agent_planner') == 3


def test_authorization_is_not_carried_between_tools(gateway):
    # customer_service may search knowledge, but a later claim request in the same run is still denied.
    h = build(gateway, [turn([call('search_knowledge', {'query': 'privacy'})]), turn([call()])],
              retriever=Knowledge())
    with pytest.raises(SecureAgentError) as exc:
        agent(h, 'Privacy procedure and claim CLM-1002', Role.CUSTOMER_SERVICE)
    assert exc.value.error_code.value == 'TOOL_DENIED'
    assert h.mcp.called == ['search_knowledge'] and [n for n, _ in h.executed] == ['search_knowledge']


def test_batch_of_calls_executes_each_through_mcp_tool_node(gateway):
    h = build(gateway, [turn([call(), call('get_policy_details', {'policy_id': 'POL-2002'})]),
                        turn(content='Both found.')])
    result = agent(h, 'Claim CLM-1002 and policy POL-2002')
    assert result['orchestration']['nodes'].count('mcp_tool') == 2
    assert result['orchestration']['nodes'].count('tool_result_guard') == 2
    assert h.mcp.called == ['get_claim_status', 'get_policy_details']


# ------------------------------------------------------------------ limits

def test_graph_recursion_limit_fails_safely_and_cleans_up(gateway):
    h = build(gateway, [turn([call()]), turn(content='done')])
    h.service.runtime.recursion_limit = 4
    with pytest.raises(SecureAgentError) as exc:
        agent(h)
    assert exc.value.error_code.value == 'AGENT_LIMIT_REACHED'
    assert any(e.event is GraphTraceType.GRAPH_LIMIT_REACHED for e in h.trace.events)
    assert events(h.audit)[-2:] == ['AGENT_STEP_LIMIT_REACHED', 'AGENT_FAILED']
    assert not h.service.runtime.checkpointer.storage


def test_recursion_limit_covers_configured_budget():
    assert recursion_limit_for(5, 5) == 75 and recursion_limit_for(10, 10) == 240


@pytest.mark.parametrize('limits,turns,expected_calls', [
    ({'max_agent_steps': 2}, [turn([call()]), turn([call('search_knowledge', {'query': 'x'})])], 1),
    ({'max_agent_tool_calls': 1}, [turn([call(), call('get_policy_details', {'policy_id': 'POL-2002'})])], 0),
    ({}, [turn([call()]), turn([call()])], 1),
])
def test_agent_budgets_bound_mcp_execution(gateway, limits, turns, expected_calls):
    h = build(gateway, turns, **limits)
    with pytest.raises(SecureAgentError) as exc:
        agent(h, 'CLM-1002 POL-2002')
    assert exc.value.error_code.value == 'AGENT_LIMIT_REACHED' and len(h.mcp.called) == expected_calls


def test_model_cannot_create_infinite_cycle(gateway):
    h = build(gateway, [lambda payload, n=[0]: turn([call('search_knowledge', {'query': f'q{len(payload["messages"])}'})])],
              retriever=Knowledge())
    with pytest.raises(SecureAgentError) as exc:
        agent(h, 'loop forever')
    assert exc.value.error_code.value == 'AGENT_LIMIT_REACHED' and len(h.mcp.called) <= 5


# ------------------------------------------------------------------ failure routes

def test_failure_routes_reach_safe_error(gateway, monkeypatch):
    # PII detection failure: nothing reaches the model or MCP.
    h = build(gateway)
    monkeypatch.setattr(gateway, 'protect', lambda *a, **k: (_ for _ in ()).throw(PIIDetectionError('safe')))
    with pytest.raises(PIIDetectionError):
        agent(h)
    assert failed_nodes(h) == ['input_guard'] and not h.fake.payloads and not h.mcp.listed
    monkeypatch.undo()

    # LLM unavailable.
    h = build(gateway, [httpx.ConnectError('down')])
    h.fake.turns = [lambda payload: (_ for _ in ()).throw(httpx.ConnectError('Maria Garcia'))]
    with pytest.raises(SecureAgentError) as exc:
        agent(h)
    assert exc.value.error_code.value == 'LLM_UNAVAILABLE' and failed_nodes(h) == ['agent_planner']

    # RAG unavailable.
    h = build(gateway)
    with pytest.raises(SecureAgentError) as exc:
        h.service.rag_chat('What is the privacy procedure?', [], 'p8-rag-down')
    assert exc.value.error_code.value == 'RETRIEVAL_FAILED' and failed_nodes(h) == ['secure_rag']

    # Invalid tool arguments.
    h = build(gateway, [turn([call(arguments={'claim_id': '../etc/passwd'})])])
    with pytest.raises(SecureAgentError) as exc:
        agent(h)
    assert exc.value.error_code.value == 'TOOL_ARGUMENTS_INVALID' and failed_nodes(h) == ['policy_gate']

    # Tool-result security failure (oversized result).
    handlers = dict(build_registry(None, gateway, 3)._handlers)
    handlers[ToolName.CLAIM] = lambda a, s: {'text': 'x' * 2000}
    h = build(gateway, handlers=handlers, max_result_chars=256)
    with pytest.raises(SecureAgentError) as exc:
        agent(h)
    assert exc.value.error_code.value == 'TOOL_RESULT_TOO_LARGE' and failed_nodes(h) == ['mcp_tool']
    assert len(h.fake.payloads) == 1


def test_unsupported_provider_capability_is_gated_before_mcp(gateway):
    class PlainOnly:
        model_name = 'qwen3-1.7b-generic-gpu'
        provider_name = 'foundry'
        tool_calling_supported = False
        def generate(self, prompt, *, system=None): return 'ok'
        def complete(self, messages, tools): raise AssertionError('must not be called')
    h = build(gateway, llm=PlainOnly())
    with pytest.raises(LLMCapabilityError):
        agent(h)
    assert failed_nodes(h) == ['router'] and not h.mcp.listed and not h.executed
    assert h.service.chat('hello', [], 'p8-foundry-chat').response == 'ok'


def test_failures_do_not_leak_stack_traces_or_state(gateway, caplog):
    handlers = dict(build_registry(None, gateway, 3)._handlers)
    def broken(arguments, session): raise RuntimeError(RAW + ' Traceback secret')
    handlers[ToolName.CLAIM] = broken
    h = build(gateway, handlers=handlers)
    with caplog.at_level(logging.DEBUG), pytest.raises(SecureAgentError) as exc:
        agent(h)
    text = str(exc.value) + caplog.text + json.dumps([e.to_dict() for e in h.audit.events + h.trace.events])
    assert exc.value.error_code.value == 'TOOL_FAILED'
    for value in ('Maria Garcia', '987-65-4321', 'Traceback secret'):
        assert value not in text


# ------------------------------------------------------------------ state isolation & privacy

def test_graph_state_never_contains_raw_pii_or_vaults(gateway):
    from langgraph.checkpoint.memory import InMemorySaver
    snapshots = []

    class Recording(InMemorySaver):
        def put(self, config, checkpoint, metadata, new_versions):
            snapshots.append(json.dumps(checkpoint['channel_values'], default=repr))
            return super().put(config, checkpoint, metadata, new_versions)

    h = build(gateway, [turn([call()]), turn(content='Hello <PERSON_1>, status for <TOOL_PERSON_1>.')])
    runtime = h.service.runtime
    runtime._saver = Recording()
    runtime.graph = runtime.graph.copy(update={'checkpointer': runtime._saver})
    result = agent(h, USER + ' Claim CLM-1002?', types=['PERSON'])
    assert 'David Chen' in result['response']  # restored for the caller only
    assert snapshots
    for snapshot in snapshots:
        for value in USER_VALUES + TOOL_VALUES:
            assert value not in snapshot
        assert 'TokenVault' not in snapshot and 'UNTRUSTED TOOL DATA' in snapshot
        assert '<PERSON_1>' in snapshot and '<TOOL_PERSON_1>' in snapshot
    assert not runtime.checkpointer.storage  # deleted at the end of the run


def test_sequential_requests_share_no_state(gateway):
    h = build(gateway, [turn([call()]), turn(content='Status for <PERSON_1>.')] * 2)
    first = agent(h, 'I am Priya Patel. CLM-1002?', types=['PERSON'])
    second = agent(h, 'CLM-1002?', types=['PERSON'])
    assert 'Priya Patel' in first['response'] and 'Priya Patel' not in second['response']
    assert second['orchestration']['nodes'] == AGENT_PATH  # fresh state, not appended
    assert len(h.fake.payloads[2]['messages']) == 2  # second run starts from system + user only


def test_concurrent_mixed_requests_are_isolated(gateway):
    barrier = threading.Barrier(4)

    def step(messages):
        if len(messages) == 2:
            barrier.wait(timeout=10)
            which = 'CLM-1001' if 'CLM-1001' in messages[1]['content'] else 'CLM-1002'
            return tool_turn(('get_claim_status', {'claim_id': which}))
        return answer('Hello <PERSON_1>. ' + messages[-1]['content'])

    native = ScriptedNative([step] * 8)
    h = build(gateway, llm=native)
    requests = [('My name is Priya Patel. CLM-1001', 'a'), ('My name is John Smith. CLM-1002', 'b'),
                ('My name is Ana Lopez. CLM-1001', 'c'), ('My name is Wei Zhang. CLM-1002', 'd')]
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda r: h.service.agent_chat(r[0], ['PERSON'], r[1], Role.CLAIMS_ADJUSTER), requests))
    names = ['Priya Patel', 'John Smith', 'Ana Lopez', 'Wei Zhang']
    for result, name in zip(results, names):
        assert name in result['response'] and all(other not in result['response'] for other in names if other != name)
    assert 'APPROVED' in results[0]['response'] and 'PENDING REVIEW' in results[1]['response']
    assert all(value not in native.wire for value in ('David Chen', 'Maria Garcia') + tuple(names))
    assert not h.service.runtime.checkpointer.storage and len(h.service.runtime.approvals) == 0


# ------------------------------------------------------------------ namespaces

def test_namespaces_never_cross_restore(gateway):
    knowledge = Knowledge(RAW)
    h = build(gateway, [turn([call()]), turn([call('search_knowledge', {'query': 'privacy'})]),
                        turn(content='User <PERSON_1>; tool <TOOL_PERSON_1>; document <DOC_PERSON_1>.')],
              retriever=knowledge)
    result = agent(h, 'My name is David Chen. CLM-1002 privacy?', types=['PERSON', 'EMAIL_ADDRESS'])
    assert 'David Chen' in result['response']
    assert '<TOOL_PERSON_1>' in result['response'] and '<DOC_PERSON_1>' in result['response']
    assert 'Maria Garcia' not in result['response']


def test_tool_token_lookalikes_cannot_alias_user_values(gateway):
    handlers = dict(build_registry(None, gateway, 3)._handlers)
    handlers[ToolName.CLAIM] = lambda a, s: {'note': 'Echo <PERSON_1> <TOOL_PERSON_1> <DOC_PERSON_1>'}
    h = build(gateway, [turn([call()]), lambda payload: turn(content=payload['messages'][-1]['content'])],
              handlers=handlers)
    result = agent(h, 'My name is David Chen. CLM-1002', types=['PERSON'])
    tool_data = h.fake.payloads[-1]['messages'][-1]['content']
    assert '<PERSON_1>' not in tool_data and 'literal placeholder' in tool_data
    assert 'David Chen' not in result['response']


# ------------------------------------------------------------------ prompt injection via MCP result

def test_malicious_mcp_result_is_delimited_data_and_cannot_escalate(gateway):
    handlers = dict(build_registry(None, gateway, 3)._handlers)
    handlers[ToolName.CLAIM] = lambda a, s: {'claim_id': 'CLM-1002', 'status': 'OPEN', 'note': INJECTION + ' ' + RAW}
    h = build(gateway, [turn([call()]), turn([call(arguments={'claim_id': 'CLM-1001'})])], handlers=handlers)
    with pytest.raises(SecureAgentError) as exc:
        agent(h)
    assert exc.value.error_code.value == 'TOOL_DENIED'  # follow-up enumeration denied by record scope
    second = h.fake.payloads[1]['messages']
    assert INJECTION not in second[0]['content'] and second[0]['role'] == 'system'
    assert second[-1]['role'] == 'tool' and 'UNTRUSTED TOOL DATA' in second[-1]['content']
    assert h.mcp.called == ['get_claim_status'] and all(v not in h.fake.wire for v in TOOL_VALUES[:3])


# ------------------------------------------------------------------ observability & tracing

TRACE_KEYS = {'timestamp', 'request_id', 'event', 'status', 'node', 'route', 'step', 'tool_name', 'provider',
              'model', 'latency_ms', 'error_category'}


def test_trace_and_audit_are_privacy_safe(gateway, caplog):
    h = build(gateway, [turn([call()]), turn(content='Status for <PERSON_1> <TOOL_PERSON_1>.')])
    with caplog.at_level(logging.DEBUG):
        agent(h, USER + ' CLM-1002?', types=['PERSON'])
    traced = [e.to_dict() for e in h.trace.events]
    assert {'GRAPH_STARTED', 'GRAPH_ROUTED', 'GRAPH_NODE_ENTERED', 'GRAPH_NODE_COMPLETED', 'GRAPH_COMPLETED'} <= {
        e['event'] for e in traced}
    assert all(set(e) <= TRACE_KEYS for e in traced)
    blob = json.dumps(traced) + json.dumps([e.to_dict() for e in h.audit.events]) + caplog.text
    for value in USER_VALUES + TOOL_VALUES + ('CLM-1002', '<PERSON_', '<TOOL_', 'UNTRUSTED', 'Ignore'):
        assert value not in blob
    assert {'MCP_TOOL_DISCOVERED', 'MCP_TOOL_REQUESTED', 'MCP_TOOL_AUTHORIZED', 'MCP_TOOL_COMPLETED'} <= set(events(h.audit))


@pytest.mark.parametrize('fields', [{'node': 'agent_planner'}, {'route': 'shell'}, {'step': 101},
                                    {'tool_name': 'execute_shell'}, {'model': 'name@example.com'}])
def test_trace_schema_rejects_untrusted_fields(fields):
    with pytest.raises(ValueError):
        GraphTraceEvent(request_id='trace', event=GraphTraceType.GRAPH_STARTED, status=AuditStatus.SUCCESS, **fields)


@pytest.mark.parametrize('variable', ['LANGSMITH_TRACING', 'LANGCHAIN_TRACING_V2'])
def test_cloud_tracing_is_refused_at_startup(monkeypatch, variable):
    monkeypatch.setenv(variable, 'true')
    with pytest.raises(ConfigurationError):
        load_settings()


def test_graph_runs_with_tracing_forced_off(gateway, monkeypatch):
    import langsmith.utils
    monkeypatch.setenv('LANGSMITH_TRACING', 'true')
    seen = []

    class Probe(ScriptedNative):
        def generate(self, prompt, *, system=None):
            seen.append(langsmith.utils.tracing_is_enabled())
            return 'ok'

    build(gateway, llm=Probe([])).service.chat('hello', [], 'p8-tracing')
    assert seen == [False]


def test_model_boundary_never_sees_raw_user_or_mcp_pii(gateway):
    """Critical: every generation boundary across chat, RAG, single and multi-step agent."""
    handlers = dict(build_registry(Knowledge(RAW), gateway, 3)._handlers)
    handlers[ToolName.CLAIM] = lambda a, s: {'claim_id': a['claim_id'], 'customer_name': 'Maria Garcia',
                                             'email': 'maria.garcia@example.com', 'ssn': '987-65-4321',
                                             'status': 'Under Review'}
    native = ScriptedNative([tool_turn(('get_claim_status', {'claim_id': 'CLM-1001'})),
                             tool_turn(('search_knowledge', {'query': 'privacy procedure for Maria Garcia'})),
                             answer('Done for <PERSON_1>; customer <TOOL_PERSON_1>.')])
    h = build(gateway, llm=native, handlers=handlers, retriever=FixedRetriever())
    h.service.chat(USER, ['PERSON'], 'p8-boundary-chat')
    h.service.rag_chat(USER, ['PERSON'], 'p8-boundary-rag')
    result = agent(h, USER + ' Check claim CLM-1001 and the privacy procedure.', types=['PERSON'])
    wire = native.wire
    for value in USER_VALUES + ('Maria Garcia', 'maria.garcia@example.com', '987-65-4321'):
        assert value not in wire
    for token in ('<PERSON_1>', '<EMAIL_1>', '<SSN_1>', '<TOOL_PERSON_1>', '<TOOL_EMAIL_1>', '<TOOL_SSN_1>',
                  '<DOC_PERSON_'):
        assert token in wire
    tool_message = next(m for p in native.payloads if 'messages' in p for m in p['messages'] if m['role'] == 'tool')
    data = json.loads(tool_message['content'])['UNTRUSTED TOOL DATA']
    assert data['customer_name'] == '<TOOL_PERSON_1>' and data['email'] == '<TOOL_EMAIL_1>'
    assert data['status'] == 'Under Review'
    assert 'David Chen' in result['response'] and '<TOOL_PERSON_1>' in result['response']
