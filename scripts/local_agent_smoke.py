"""Real local Phase 4 verification. Prints safe evidence, never token mappings."""
import json
from pathlib import Path
import socket
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
import uvicorn
from app.main import build_service, create_app
from app.core.config import get_settings
from app.agent.tool_registry import ToolRegistry
from app.agent.schemas import ToolName
from app.security.gateway import normalize_input


def main():
    settings=get_settings()
    service=build_service(settings)
    host=service.tool_gateway.mcp.host  # handlers run inside the in-process MCP servers
    original=host.registry
    executions=[]
    def observe(name,handler):
        def invoke(args,session):
            executions.append(name.value)
            return handler(args,session)
        return invoke
    host.registry=ToolRegistry({name:observe(name,handler) for name,handler in original._handlers.items()})
    pii=['David Chen','Maria Garcia','John Smith','david.chen@example.com','maria.garcia@example.com',
         'john.smith@example.org','123-45-6789','987-65-4321','(415) 555-0132','(212) 555-0187']
    safe_results=[]
    model_requests=[]
    def inspect(request):
        if request.url.path!='/api/chat': return
        payload=json.loads(request.content)
        assert payload['model']=='qwen2.5:7b'
        assert {t['function']['name'] for t in payload['tools']}=={n.value for n in ToolName}
        assert payload['messages'][0]['role']=='system'
        visible=normalize_input(json.dumps(payload,ensure_ascii=False))
        for value in pii: assert value.casefold() not in visible.casefold(), 'Raw PII at model boundary'
        model_requests.append(payload)
        for message in payload['messages']:
            if message['role']=='tool':
                assert '<TOOL_PERSON_' in message['content'] and '<TOOL_SSN_' in message['content']
                safe_results.append(json.loads(message['content']))
    service._llm._client.event_hooks['request']=[inspect]
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(create_app(settings,service),host='127.0.0.1',port=port,log_level='warning'))
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    try:
        deadline=time.monotonic()+15
        while not server.started:
            if not thread.is_alive() or time.monotonic()>deadline: raise RuntimeError('Local API startup failed')
            time.sleep(0.05)
        with httpx.Client(trust_env=False,timeout=180) as client:
            allowed=client.post(f'http://127.0.0.1:{port}/api/v1/agent/chat',json={
                'message':'What is the status of claim CLM-1001?', 'role':'claims_adjuster',
                'allowed_reidentification_types':['PERSON']})
            assert allowed.status_code==200, f'Authorized request status {allowed.status_code}: {allowed.json().get("error",{}).get("code")}'
            body=allowed.json()
            assert body['tools_used']==['get_claim_status'] and body['agent_steps']==2
            assert executions==['get_claim_status'] and safe_results
            assert body['security']['tool_pii_detected']
            before=len(executions)
            denied=client.post(f'http://127.0.0.1:{port}/api/v1/agent/chat',json={
                'message':'What is the status of claim CLM-1001?', 'role':'customer_service'})
        assert denied.status_code==403 and denied.json()['error']['code']=='TOOL_DENIED'
        assert len(executions)==before, 'Unauthorized tool executed'
        audit=Path(settings.audit_log_path).read_text()
        for value in pii:
            assert value not in audit and value not in allowed.text and value not in denied.text
        events=[json.loads(line) for line in audit.splitlines()]
        complete=next(e for e in reversed(events) if e['request_id']==body['request_id'] and e['event']=='AGENT_COMPLETED')
        denial=next(e for e in reversed(events) if e['request_id']==denied.json()['request_id'] and e['event']=='TOOL_DENIED')
        print(json.dumps({'result':'PASS','authorized_agent_test':'PASS','unauthorized_agent_test':'PASS',
            'model_requests':len(model_requests),'concrete_tool_executions':executions,
            'sanitized_tool_result':safe_results[0],'api_response':body,'denied_response':denied.json(),
            'audit_event':complete,'denial_audit_event':denial},indent=2))
    finally:
        server.should_exit=True;thread.join(timeout=15)

if __name__=='__main__': main()
