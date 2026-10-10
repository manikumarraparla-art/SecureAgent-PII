"""Opt-in agent sources keep the original Phase 4 response contract unchanged."""
from fastapi.testclient import TestClient
from app.main import create_app
from app.core.config import Settings
from app.agent.tool_registry import build_registry
from app.rag.chunking import Chunk
from app.rag.vector_store import SearchHit
from tests.test_agent import harness,turn,call,run
from tests.test_rag import DOC,RAW

class Retriever:
    def retrieve(self,query,top_k):
        return [SearchHit(Chunk(DOC,'claim_1001.txt',DOC+'-0',RAW),0.9)]


def test_opt_in_agent_sources_are_safe_and_backward_compatible(gateway):
    for include in (False,True):
        h=harness(gateway,[turn([call('search_knowledge',{'query':'privacy records'})]),turn(content='Follow privacy procedures.')],build_registry(Retriever(),gateway,3)._handlers)
        with TestClient(create_app(Settings(),h.service)) as client:
            body={'message':'privacy records','role':'customer_service','include_sources':include}
            response=client.post('/api/v1/agent/chat',json=body)
        assert response.status_code==200
        data=response.json()
        assert ('sources' in data)==include
        if include:
            assert set(data['sources'][0])=={'document_id','filename','chunk_id','score'}
            assert data['sources'][0]['filename']=='claim_1001.txt'
            assert 'Maria Garcia' not in response.text and '/Users/' not in response.text and 'text' not in data['sources'][0]
        assert '987-65-4321' not in h.fake.wire


def test_source_opt_in_boolean_is_strict(gateway):
    h=harness(gateway)
    with TestClient(create_app(Settings(),h.service)) as client:
        response=client.post('/api/v1/agent/chat',json={'message':'CLM-1002','include_sources':'yes'})
        assert response.status_code==422 and not h.executed
