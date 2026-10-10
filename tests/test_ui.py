"""Phase 5 public HTTP client and Streamlit widgets; no graphical browser needed."""
import json
from pathlib import Path
from unittest.mock import patch
import httpx
import pytest
import toml
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest
from app.ui.api_client import APIClient, BackendHealth, FailureCode, UIFailure, ReceivedResponse, local_backend_url, safe_error
from app.ui.contracts import APIResponse, Mode, ROLES, Source, SecurityMetadata
from app.ui.state import Conversation
from app.ui.components import source_rows, tool_rows, security_rows

DOC='a'*32
SOURCE={'document_id':DOC,'filename':'claim_1001.txt','chunk_id':DOC+'-0','score':0.9}

def payload(mode=Mode.CHAT):
    data={'request_id':'request-id','response':'A safe answer.','model':'qwen2.5:7b',
          'security':{'pii_detected':True,'entity_types':['PERSON','EMAIL_ADDRESS','US_SSN'],'reidentification_decision':'DENIED'}}
    if mode is Mode.RAG: data.update(sources=[SOURCE])
    if mode is Mode.AGENT: data.update(sources=[SOURCE],tools_used=['search_knowledge'],agent_steps=2)
    return data

@pytest.mark.parametrize('url',['http://127.0.0.1:8000','http://localhost:8123/','http://[::1]:8000','https://localhost:8000'])
def test_loopback_urls_allowed(url): assert local_backend_url(url)==url.rstrip('/')

@pytest.mark.parametrize('url',['http://example.com','http://192.168.1.2','http://127.0.0.1.evil.com',
    'http://localhost@evil.com','http://evil.com@localhost','http://localhost:11434',
    'ftp://localhost','http://localhost:0','http://localhost:99999','http://localhost/api/chat',
    'http://localhost/?model=test','http://localhost/#secret','http://localhost\\@evil.com',
    'http://127.0.0.1%2eevil.com','http://localhost\n','http://2130706433','http://0.0.0.0'])
def test_nonlocal_and_ambiguous_urls_rejected(url):
    with pytest.raises(UIFailure) as exc: APIClient(url)
    assert exc.value.code is FailureCode.CONFIGURATION
    assert url not in str(exc.value)

@pytest.mark.parametrize('mode,path',[(Mode.CHAT,'/api/v1/chat'),(Mode.RAG,'/api/v1/rag/chat'),(Mode.AGENT,'/api/v1/agent/chat')])
def test_mode_endpoint_mapping_and_payload(mode,path):
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(200,json=payload(mode))
    with APIClient(transport=httpx.MockTransport(handler)) as client:
        result=client.send(mode,'Synthetic question','claims_adjuster',['PERSON'],2)
    assert requests[0].url.path==path
    body=json.loads(requests[0].content)
    assert body['message']=='Synthetic question' and body['allowed_reidentification_types']==['PERSON']
    assert ('role' in body)==(mode is Mode.AGENT)
    assert ('top_k' in body)==(mode is Mode.RAG)
    assert ('include_sources' in body)==(mode is Mode.AGENT)
    assert result.latency_ms>=0

@pytest.mark.parametrize('exception,code',[(httpx.ConnectError,FailureCode.BACKEND_UNAVAILABLE),(httpx.ReadTimeout,FailureCode.TIMEOUT),(httpx.ConnectTimeout,FailureCode.TIMEOUT)])
def test_network_errors_are_safe(exception,code):
    def handler(request): raise exception('Maria Garcia 987-65-4321')
    with APIClient(transport=httpx.MockTransport(handler)) as client,pytest.raises(UIFailure) as exc:
        client.send(Mode.CHAT,'Synthetic query')
    assert exc.value.code is code and 'Maria Garcia' not in str(exc.value) and exc.value.__cause__ is None

@pytest.mark.parametrize('status,code,expected',[(403,'TOOL_DENIED',FailureCode.TOOL_DENIED),(422,'INVALID_REQUEST',FailureCode.INVALID_REQUEST),
    (503,'VECTOR_STORE_FAILED',FailureCode.RAG_UNAVAILABLE),(503,'EMBEDDING_FAILED',FailureCode.RAG_UNAVAILABLE),
    (503,'RETRIEVAL_FAILED',FailureCode.RAG_UNAVAILABLE),(503,'LLM_UNAVAILABLE',FailureCode.LLM_UNAVAILABLE),
    (503,'LLM_MODEL_UNAVAILABLE',FailureCode.LLM_UNAVAILABLE),(504,'LLM_TIMEOUT',FailureCode.TIMEOUT),
    (409,'AGENT_LIMIT_REACHED',FailureCode.LIMIT_REACHED),(500,'PII_DETECTION_FAILED',FailureCode.SERVER_ERROR),
    (500,'Maria Garcia',FailureCode.SERVER_ERROR)])
def test_safe_mapped_errors(status,code,expected):
    body={'error':{'code':code,'message':'Maria Garcia 987-65-4321'},'prompt':'raw secret'}
    assert safe_error(status,body) is expected
    with APIClient(transport=httpx.MockTransport(lambda r:httpx.Response(status,json=body))) as client,pytest.raises(UIFailure) as exc:
        client.send(Mode.CHAT,'query')
    assert exc.value.code is expected and 'Maria Garcia' not in str(exc.value)

@pytest.mark.parametrize('body',[{'oops':'Maria Garcia'},[],None,'raw secret'])
def test_malformed_error_body_never_shown(body): assert safe_error(500,body) is FailureCode.SERVER_ERROR


def test_no_proxy_or_redirect_forwarding(monkeypatch):
    monkeypatch.setenv('HTTP_PROXY','http://example.com:8080')
    requests=[]
    def redirect(request):
        requests.append(request)
        return httpx.Response(307,headers={'location':'https://example.com/steal'},text='Maria Garcia')
    with APIClient(transport=httpx.MockTransport(redirect)) as client:
        assert client._client.trust_env is False and client._client.follow_redirects is False
        with pytest.raises(UIFailure): client.send(Mode.CHAT,'question')
    assert len(requests)==1 and requests[0].url.host=='127.0.0.1'

@pytest.mark.parametrize('body',[{'response':'raw secret'},'raw body',None,{'request_id':'name@example.com','response':'answer'}])
def test_invalid_response_is_redacted(body):
    with APIClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=body))) as client,pytest.raises(UIFailure) as exc:
        client.send(Mode.CHAT,'query')
    assert exc.value.code is FailureCode.INVALID_RESPONSE and 'raw' not in str(exc.value)


def test_unknown_response_fields_not_retained():
    data=payload();data.update(token_mapping={'PERSON_1':'Maria Garcia'},prompt='raw',vectors=[1,2],raw_tool_result='987-65-4321')
    data['security'].update(original_pii='Maria Garcia',presidio_results='secret')
    result=APIResponse.model_validate(data)
    serialized=result.model_dump_json()
    for raw in ['Maria Garcia','987-65-4321','token_mapping','presidio','raw_tool_result','vectors','prompt']: assert raw not in serialized

@pytest.mark.parametrize('filename',['/Users/mani/claim_1001.txt','../claim_1001.txt','maria.garcia@example.com','claim_1001.txt\nsecret','https://example.com/a.txt','<img src=x>'])
def test_source_rendering_rejects_unsafe_metadata(filename):
    with pytest.raises(ValidationError): Source.model_validate({**SOURCE,'filename':filename})


def test_source_rows_contain_only_safe_public_metadata():
    rows=source_rows([Source.model_validate({**SOURCE,'text':'Maria Garcia','vectors':[1,2]})])
    assert rows[0]['Filename']=='claim_1001.txt' and rows[0]['Similarity']=='0.900'
    assert 'Maria Garcia' not in json.dumps(rows) and 'vectors' not in json.dumps(rows)


def test_multiple_agent_searches_can_return_more_than_ten_safe_sources():
    sources=[{**SOURCE,'chunk_id':DOC+f'-{i}'} for i in range(12)]
    result=APIResponse.model_validate({**payload(Mode.AGENT),'sources':sources})
    assert len(source_rows(result.sources))==12
    with pytest.raises(ValidationError):
        APIResponse.model_validate({**payload(Mode.AGENT),'sources':sources*9})

@pytest.mark.parametrize('change',[{'tools_used':['shell']},{'tools_used':['Maria Garcia']},{'agent_steps':True},{'agent_steps':11},
    {'security':{'pii_detected':True,'entity_types':['Maria Garcia'],'reidentification_decision':'DENIED'}},
    {'security':{'pii_detected':'987-65-4321','entity_types':[],'reidentification_decision':'DENIED'}},
    {'security':{'pii_detected':True,'entity_types':[],'reidentification_decision':'Maria Garcia'}}])
def test_unsafe_tool_security_metadata_rejected(change):
    with pytest.raises(ValidationError): APIResponse.model_validate({**payload(Mode.AGENT),**change})


def test_metadata_panels_only_allowlisted_values():
    result=APIResponse.model_validate(payload(Mode.AGENT))
    assert ('Authorization','Allowed') in tool_rows(result)
    assert ('PII detected','Yes') in security_rows(result.security)
    text=json.dumps(tool_rows(result)+security_rows(result.security))
    assert 'get_claim_status' not in text and 'search_knowledge' in text


def test_session_memory_and_clear():
    a,b=Conversation(),Conversation()
    a.add_user('Maria Garcia',Mode.CHAT)
    a.add_response(ReceivedResponse(APIResponse.model_validate(payload()),120),Mode.CHAT)
    a.add_error(FailureCode.TOOL_DENIED,Mode.AGENT)
    assert len(a.messages)==3 and not b.messages
    assert 'Maria Garcia' not in repr(a) and 'Maria Garcia' not in repr(a.messages)
    assert isinstance(a.messages,tuple)
    a.clear();assert a.messages==()

@pytest.mark.parametrize('role',ROLES)
def test_role_allowlist(role):
    with APIClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=payload(Mode.AGENT)))) as client:
        assert client.send(Mode.AGENT,'CLM-1001',role).result.agent_steps==2

@pytest.mark.parametrize('option',[{'mode':'unknown'}, {'role':'admin'}, {'message':' '}, {'message':'x'*8001},
    {'allowed_types':['US_SSN']}, {'allowed_types':['Maria Garcia']}, {'top_k':True}])
def test_invalid_request_options_stop_before_network(option):
    seen=[]
    def handler(request): seen.append(request);return httpx.Response(200,json=payload())
    kwargs={'mode':Mode.RAG if 'top_k' in option else Mode.AGENT,'message':'CLM-1001','role':'claims_adjuster'};kwargs.update(option)
    with APIClient(transport=httpx.MockTransport(handler)) as client,pytest.raises(UIFailure): client.send(**kwargs)
    assert seen==[]

@pytest.mark.parametrize('args',[(0,180),(31,180),(5,601),(5,float('nan')),(True,180)])
def test_timeout_configuration_bounds(args):
    with pytest.raises(UIFailure): APIClient(connect_timeout=args[0],response_timeout=args[1])

@pytest.mark.parametrize('status,body,available,ollama',[(200,{'status':'healthy','ollama':'available','model':'qwen2.5:7b'},True,True),
    (503,{'status':'degraded','ollama':'model_missing','model':'qwen2.5:7b'},True,False),
    (200,{'status':'healthy','ollama':'Maria Garcia','model':'qwen2.5:7b'},False,False),(500,{'secret':'raw'},False,False)])
def test_health_contract(status,body,available,ollama):
    paths=[]
    def handler(request): paths.append(request.url.path);return httpx.Response(status,json=body)
    with APIClient(transport=httpx.MockTransport(handler)) as client: health=client.health()
    assert health.backend_available is available and health.ollama_available is ollama
    assert paths==['/health']


def test_project_config_disables_telemetry_and_keeps_loopback():
    cfg=toml.loads(Path('.streamlit/config.toml').read_text())
    assert cfg['browser']['gatherUsageStats'] is False
    assert cfg['server']['address']=='127.0.0.1'
    assert cfg['server']['enableCORS'] and cfg['server']['enableXsrfProtection']
    assert cfg['client']['showErrorDetails']=='none'
    assert cfg['runner']['magicEnabled'] is False


def test_streamlit_modes_and_clear(monkeypatch):
    calls=[]
    def send(client,mode,message,role='customer_service',allowed_types=(),top_k=3):
        calls.append((mode,message,role))
        return ReceivedResponse(APIResponse.model_validate(payload(mode)),150)
    monkeypatch.setattr(APIClient,'health',lambda s:BackendHealth(True,True))
    monkeypatch.setattr(APIClient,'send',send)
    app=AppTest.from_file(Path(__file__).resolve().parents[1] / 'app/ui/app.py').run(timeout=15)
    assert not app.exception and app.title[0].value=='SecureAgent-PII'
    for mode in Mode:
        app.selectbox(key='mode').set_value(mode.value).run()
        if mode is Mode.AGENT: app.selectbox(key='demo_role').set_value('claims_adjuster').run()
        app.chat_input[0].set_value('Synthetic question').run()
        assert not app.exception
    assert len(calls)==3 and len(app.session_state['conversation'].messages)==6
    assert any(e.label=='Security status' for e in app.expander)
    assert any(e.label=='Sources' for e in app.expander)
    assert any(e.label=='Tool activity' for e in app.expander)
    app.button(key='clear').click().run()
    assert app.session_state['conversation'].messages==() and len(app.chat_message)==0
    assert len(calls)==3


def test_streamlit_denial_uses_safe_message(monkeypatch):
    monkeypatch.setattr(APIClient,'health',lambda s:BackendHealth(True,True))
    def deny(*args,**kwargs): raise UIFailure(FailureCode.TOOL_DENIED)
    monkeypatch.setattr(APIClient,'send',deny)
    app=AppTest.from_file(Path(__file__).resolve().parents[1] / 'app/ui/app.py').run(timeout=15)
    app.selectbox(key='mode').set_value(Mode.AGENT.value).run()
    app.chat_input[0].set_value('What is CLM-1001 status?').run()
    assert not app.exception
    assert any('Tool access denied' in e.value for e in app.error)
    assert not any('Traceback' in e.value for e in app.error)


def test_streamlit_plain_text_blocks_remote_markdown_and_html(monkeypatch):
    monkeypatch.setattr(APIClient,'health',lambda s:BackendHealth(True,True))
    data=payload();data['response']='![remote](https://example.com/track) <script>fetch("https://example.com")</script>'
    monkeypatch.setattr(APIClient,'send',lambda *a,**k:ReceivedResponse(APIResponse.model_validate(data),50))
    app=AppTest.from_file(Path(__file__).resolve().parents[1] / 'app/ui/app.py').run(timeout=15)
    app.chat_input[0].set_value('<img src=https://example.com>').run()
    assert not app.exception
    assert any(data['response']==e.value for e in app.text)
    assert not any('example.com' in e.value for e in app.markdown)


def test_clear_removes_drafts_and_sample_text(monkeypatch):
    monkeypatch.setattr(APIClient,'health',lambda s:BackendHealth(True,True))
    app=AppTest.from_file(Path(__file__).resolve().parents[1] / 'app/ui/app.py').run(timeout=15)
    app.button(key='sample').click().run()
    app.text_area(key='sample_text').set_value('Maria Garcia').run()
    app.button(key='clear').click().run()
    assert app.session_state['conversation'].messages==()
    assert 'draft' not in app.session_state and 'sample_text' not in app.session_state
