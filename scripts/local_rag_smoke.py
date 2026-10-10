"""Real local integration check; no mappings or raw prompts are printed.

Run from the repository root: python scripts/local_rag_smoke.py
Uses existing Ollama models, builds index first with python -m app.rag.
"""
import json
from pathlib import Path
import socket
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
import uvicorn
from app.core.config import get_settings
from app.main import build_service, create_app
from app.security.gateway import normalize_input


def main():
    settings = get_settings()
    service = build_service(settings)
    observed = []
    pii = ['David Chen','Maria Garcia','John Smith','david.chen@example.com',
           'maria.garcia@example.com','john.smith@example.org','123-45-6789',
           '987-65-4321','987.65.4321','(415) 555-0132','(212) 555-0187','(312) 555-0199']
    def inspect(request):
        if request.url.path != '/api/chat': return
        payload = json.loads(request.content)
        visible = '\n'.join(m['content'] for m in payload['messages'])
        for value in pii:
            assert value.casefold() not in normalize_input(visible).casefold(), 'PII on generation boundary'
        assert '<DOC_PERSON_' in visible and '<DOC_EMAIL_' in visible
        assert payload['model'] == 'qwen2.5:7b'
        assert [m['role'] for m in payload['messages']] == ['system','user']
        observed.append(json.loads(payload['messages'][1]['content']))
    def inspect_embedding(request):
        if request.url.path == '/api/embed':
            visible = normalize_input(request.content.decode())
            for value in pii:
                assert value.casefold() not in visible.casefold(), 'PII on embedding boundary'
    service._retriever.embeddings._client.event_hooks['request'] = [inspect_embedding]
    service._llm._client.event_hooks['request'] = [inspect]
    # Verify actual nomic retrieval independently of model wording.
    for query,expected in [('What is the status of claim CLM-1001?','claim_1001.txt'),
                           ('What is the privacy procedure for customer records?','procedures_privacy.txt')]:
        hits=service._retriever.retrieve(query,3)
        assert expected in [h.chunk.filename for h in hits], 'Relevant source missing'
        print(json.dumps({'retrieval_query':query,'sources':[h.chunk.filename for h in hits]}))
    app = create_app(settings,service)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='warning'))
    thread = threading.Thread(target=server.run,daemon=True);thread.start()
    try:
        deadline=time.monotonic()+15
        while not server.started:
            if not thread.is_alive() or time.monotonic()>deadline: raise RuntimeError('API startup failed')
            time.sleep(0.05)
        with httpx.Client(trust_env=False,timeout=180) as client:
            response=client.post(f'http://127.0.0.1:{port}/api/v1/rag/chat',json={
                'message':"My name is David Chen. What is the status of claim CLM-1001?",
                'allowed_reidentification_types':['PERSON'],'top_k':3})
        assert response.status_code==200, f'RAG request failed: {response.status_code}'
        body=response.json()
        assert observed and any(s['filename']=='claim_1001.txt' for s in body['sources'])
        assert all('/' not in s['filename'] for s in body['sources'])
        for value in pii[1:]: assert value not in body['response']
        audit = Path(settings.audit_log_path).read_text()
        for value in pii: assert value not in audit
        events = [json.loads(line) for line in audit.splitlines() if json.loads(line)['request_id']==body['request_id']]
        assert events[-1]['retrieval_enabled']
        print(json.dumps({'result':'PASS','generation_boundary_verified':True,
                          'sanitized_context_example':observed[0]['RETRIEVED CONTEXT'][0],
                          'api_response':body,'audit_event':events[-1]},indent=2))
    finally:
        server.should_exit=True;thread.join(timeout=15)

if __name__ == '__main__': main()
