"""Real provider -> secured HTTP API -> live Streamlit verification.

Run from root: python -m scripts.local_provider_smoke --provider foundry
Uses existing installed/loaded models and index; downloads nothing. Temporary
loopback servers avoid replacing a developer's server; only ours are stopped.
Prints metadata and a short sanitized context, never prompts/mappings/responses.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

import httpx
import uvicorn
from streamlit.testing.v1 import AppTest

from app.agent.tool_registry import ToolRegistry
from app.core.config import load_settings
from app.main import build_service, create_app
from scripts.local_ui_startup_smoke import verify_live_ui

ROOT = Path(__file__).resolve().parents[1]
RAW = ('David Chen','Maria Garcia','John Smith','david.chen@example.com',
       'maria.garcia@example.com','john.smith@example.org','123-45-6789',
       '987-65-4321','987.65.4321','(415) 555-0132','(212) 555-0187')


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        return listener.getsockname()[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider',choices=('ollama','foundry'),default='foundry')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='secureagent-provider-') as directory:
        audit_path=Path(directory)/'audit.jsonl'
        settings=replace(load_settings(),llm_provider=args.provider,audit_log_path=str(audit_path))
        service=build_service(settings)
        assert service.llm_health().value=='available', 'Selected local model not loaded/available'
        boundaries={'requests':0,'user_tokens':False,'document_tokens':False,'tool_tokens':False}
        sanitized_context=[]
        def inspect(request):
            if request.url.path not in ('/api/chat','/v1/chat/completions'):return
            payload=json.loads(request.content)
            assert payload['model']==service.model_name
            visible=json.dumps(payload,ensure_ascii=False)
            assert all(value.casefold() not in visible.casefold() for value in RAW), 'PII at provider boundary'
            assert 'token_mapping' not in visible and 'original_value' not in visible and 'vault' not in visible
            boundaries['requests']+=1
            boundaries['user_tokens'] |= all(t in visible for t in ('<PERSON_','<EMAIL_','<SSN_'))
            boundaries['document_tokens'] |= '<DOC_PERSON_' in visible
            boundaries['tool_tokens'] |= '<TOOL_PERSON_' in visible
            if not sanitized_context:
                for message in payload['messages']:
                    if '<DOC_PERSON_' in message.get('content',''):
                        context=json.loads(message['content'])['RETRIEVED CONTEXT']
                        text=context[0]['text']
                        sanitized_context.append(text[:1200])
        service._llm._client.event_hooks['request']=[inspect]
        executions=[]
        def observe(name,handler):
            def invoke(arguments,session):
                executions.append(name.value)
                return handler(arguments,session)
            return invoke
        host=service.tool_gateway.mcp.host  # handlers run inside the in-process MCP servers
        registry=host.registry
        host.registry=ToolRegistry({name:observe(name,h) for name,h in registry._handlers.items()})
        port,ui_port=free_port(),free_port()
        server=uvicorn.Server(uvicorn.Config(create_app(settings,service),host='127.0.0.1',port=port,
            log_level='warning',access_log=False))
        thread=threading.Thread(target=server.run,daemon=True);thread.start()
        ui_process=None
        old_url=os.environ.get('UI_BACKEND_URL')
        os.environ['UI_BACKEND_URL']=f'http://127.0.0.1:{port}'
        try:
            deadline=time.monotonic()+30
            while not server.started:
                assert thread.is_alive() and time.monotonic()<deadline,'API failed to start'
                time.sleep(.05)
            results={}
            with httpx.Client(base_url=os.environ['UI_BACKEND_URL'],trust_env=False,timeout=180) as api:
                health=api.get('/health').json()
                assert health['provider']==args.provider and health['llm']=='available'
                assert health['tool_calling']==service.tool_calling_supported
                for name,path,message in [
                    ('chat','/api/v1/chat','My name is David Chen. My email is david.chen@example.com. My SSN is 123-45-6789. Please acknowledge my request.'),
                    ('rag','/api/v1/rag/chat','What is the status of claim CLM-1001?')]:
                    response=api.post(path,json={'message':message})
                    assert response.status_code==200, f'{name} failed with status {response.status_code}'
                    body=response.json()
                    assert all(value not in body['response'] for value in RAW), 'PII in protected answer'
                    if name=='rag':
                        assert any(s['filename']=='claim_1001.txt' for s in body['sources'])
                        assert 'approved' in body['response'].lower(), 'Claim answer not grounded'
                    results[name]={k:v for k,v in body.items() if k!='response'}
                for role in ('claims_adjuster','customer_service'):
                    before=len(executions)
                    response=api.post('/api/v1/agent/chat',json={'message':'What is the status of claim CLM-1001?','role':role})
                    if service.tool_calling_supported and role=='claims_adjuster':
                        assert response.status_code==200 and response.json()['tools_used']==['get_claim_status']
                        results[role]={'status':200,'tools_used':response.json()['tools_used']}
                    else:
                        expected='TOOL_DENIED' if service.tool_calling_supported else 'LLM_CAPABILITY_UNSUPPORTED'
                        assert response.json()['error']['code']==expected and len(executions)==before
                        results[role]={'status':response.status_code,'error':expected}
            with tempfile.TemporaryFile() as output:
                environment={**os.environ,'STREAMLIT_SERVER_PORT':str(ui_port)}
                ui_process=subprocess.Popen([sys.executable,'-m','streamlit','run','app/ui/app.py'],
                    cwd=ROOT,env=environment,stdout=output,stderr=output)
                with httpx.Client(trust_env=False,timeout=5) as browser:
                    deadline=time.monotonic()+30
                    while True:
                        try:
                            if browser.get(f'http://127.0.0.1:{ui_port}/_stcore/health').status_code==200:break
                        except httpx.HTTPError:pass
                        assert ui_process.poll() is None and time.monotonic()<deadline,'Streamlit startup failed'
                        time.sleep(.1)
                live=verify_live_ui(f'http://127.0.0.1:{ui_port}')
                app=AppTest.from_file(ROOT/'app/ui/app.py').run(timeout=30)
                assert not app.exception
                metrics={m.label:m.value for m in app.metric}
                assert metrics['Backend']=='Available' and metrics['Model']==service.model_name
                assert metrics['Provider']==('Foundry Local' if args.provider=='foundry' else 'Ollama')
                for mode,message in [
                    ('Secure Chat','My name is David Chen. My email is david.chen@example.com. My SSN is 123-45-6789. Please acknowledge my request.'),
                    ('Secure RAG','What is the status of claim CLM-1001?')]:
                    app.selectbox(key='mode').set_value(mode).run(timeout=30)
                    app.chat_input[0].set_value(message).run(timeout=240)
                    assert not app.exception and app.session_state['conversation'].messages[-1].received is not None
                if not service.tool_calling_supported:
                    app.selectbox(key='mode').set_value('Secure Agent').run(timeout=30)
                    app.chat_input[0].set_value('What is the status of claim CLM-1001?').run(timeout=30)
                    assert app.session_state['conversation'].messages[-1].error.value=='CAPABILITY_UNSUPPORTED'
                audit=audit_path.read_text()
                assert all(value not in audit for value in RAW)
                assert all(marker not in audit for marker in ('<PERSON_','<DOC_','<TOOL_','token_mapping','UNTRUSTED'))
                events=[json.loads(line) for line in audit.splitlines()]
                assert all(event['provider']==args.provider for event in events)
                assert boundaries['user_tokens'] and boundaries['document_tokens']
                if service.tool_calling_supported:assert boundaries['tool_tokens']
                print(json.dumps({'result':'PASS','provider':args.provider,'model':service.model_name,
                    'tool_calling':service.tool_calling_supported,'health':health,
                    'generation_boundary':boundaries,'audit_privacy':'PASS','audit_example':events[-1],
                    'ui_live_session':live,'ui_provider_model_display':'PASS','ui_chat_rag':'PASS',
                    'api_results':results,'sanitized_context':sanitized_context[0]},indent=2))
        finally:
            if ui_process is not None:
                ui_process.terminate();ui_process.wait(timeout=15)
            server.should_exit=True;thread.join(timeout=15)
            if old_url is None:os.environ.pop('UI_BACKEND_URL',None)
            else:os.environ['UI_BACKEND_URL']=old_url


if __name__=='__main__':main()
