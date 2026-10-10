"""Phase 6 meaningful configuration and release-hygiene regressions."""
import json
import re
from pathlib import Path
import subprocess
import sys
import pytest
import httpx
import yaml
from app.core.config import Settings, load_settings
from app.core.exceptions import ConfigurationError
from app.llm.ollama_client import OllamaClient
from app.rag.embeddings import OllamaEmbeddingClient
from scripts.check_repository import scan_text, private_artifact, check

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('url', ['https://example.com','http://'+'user:'+'pass@localhost:11434',
    'http://localhost:11434/api/chat','http://localhost:11434/?key=value',
    'http://localhost:11434/#value','http://localhost:0','http://localhost:99999',
    'http://[::1','http://localhost\\@example.com','http://localhost\n'])
def test_all_ollama_entrypoints_reject_unsafe_urls(url):
    for construct in (lambda:Settings(ollama_base_url=url),
                      lambda:OllamaClient(base_url=url),lambda:OllamaEmbeddingClient(base_url=url)):
        with pytest.raises(ConfigurationError) as exc:
            construct()
        assert url not in str(exc.value)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,0,True])
def test_direct_settings_and_clients_reject_invalid_timeouts(value):
    for construct in (lambda:Settings(ollama_request_timeout=value),
                      lambda:OllamaClient(request_timeout=value),lambda:OllamaEmbeddingClient(timeout=value)):
        with pytest.raises(ConfigurationError): construct()


@pytest.mark.parametrize('options',[{'strict_ssn_mode':'false'}, {'pii_score_threshold':float('nan')},
    {'pii_entities':()}, {'pii_entities':('CREDIT_CARD',)}, {'reidentification_ceiling':('INVALID',)},
    {'rag_top_k':True}, {'rag_chunk_overlap':True}, {'rag_collection':'../outside'},
    {'log_level':'UNSAFE'}, {'ollama_model':'name@example.com'}, {'ollama_embed_model':''}])
def test_direct_configuration_cannot_bypass_startup_validation(options):
    with pytest.raises(ConfigurationError): Settings(**options)


@pytest.mark.parametrize('key',['STRICT_SSN_MODE','PII_SCORE_THRESHOLD','OLLAMA_REQUEST_TIMEOUT','PII_ENTITIES'])
def test_bad_configuration_error_does_not_echo_environment_value(monkeypatch,tmp_path,key):
    marker='confidential_configuration_value'
    monkeypatch.setenv(key,marker)
    with pytest.raises(ConfigurationError) as exc: load_settings(tmp_path/'missing.env')
    assert marker not in str(exc.value) and exc.value.__cause__ is None


def test_generation_client_never_follows_redirects_or_proxies(monkeypatch):
    monkeypatch.setenv('HTTPS_PROXY','http://example.com:8888')
    seen=[]
    def redirect(request):
        seen.append(request.url)
        return httpx.Response(307,headers={'location':'https://example.com/redirect'})
    client=OllamaClient(transport=httpx.MockTransport(redirect))
    try:
        assert not client._client.trust_env and not client._client.follow_redirects
        client.check_health()
        assert len(seen)==1 and seen[0].host=='localhost'
    finally: client.close()


def test_secret_checker_detects_credentials_without_returning_their_values():
    examples=['api_key="'+'x'*30+'"', 'Bearer '+'z'*40,
              '-----BEGIN '+'RSA PRIVATE KEY-----', 'postgres://user:'+'p'*20+'@host/db']
    for example in examples:
        findings=scan_text(example)
        assert findings and example not in json.dumps(findings)
    assert not scan_text('SYNTHETIC TEST DATA: David Chen; david.chen@example.com; 123-45-6789')


def test_gitignore_excludes_private_files_without_hiding_required_source():
    blocked=['.env','.env.local','.venv/file','logs/audit.jsonl','other/index.sqlite3','other/index.db-wal',
             '.streamlit/secrets.toml','.ruff_cache/result','temporary.tmp','model.gguf']
    allowed=['.env.example','.streamlit/config.toml','data/documents/claim_1001.txt',
             'secureagent_ui/workspace.py','requirements-dev.txt']
    result=subprocess.run(['git','check-ignore','--no-index','--stdin'],cwd=ROOT,
        input='\n'.join(blocked+allowed)+'\n',text=True,capture_output=True)
    assert set(result.stdout.splitlines())==set(blocked)
    assert private_artifact('.env.local') and not private_artifact('.env.example')


def test_repository_release_scan_is_clean():
    result=subprocess.run([sys.executable,'scripts/check_repository.py'],cwd=ROOT,
                           capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout


def test_tracked_private_artifact_is_found_even_when_gitignored(tmp_path):
    subprocess.run(['git','init','--quiet',str(tmp_path)],check=True)
    (tmp_path/'.gitignore').write_text('.env\n')
    (tmp_path/'.env').write_text('LOCAL_SETTING=example\n')
    subprocess.run(['git','add','--force','.env'],cwd=tmp_path,check=True)
    report=check(tmp_path)
    assert report['result']=='FAIL'
    assert {'file':'.env','category':'private_or_generated_artifact'} in report['findings']


def test_readme_and_phase_six_documentation_local_links_resolve():
    paths=[ROOT/'README.md',ROOT/'architecture.md',*(ROOT/'docs').glob('*.md')]
    for path in paths:
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',path.read_text()):
            if '://' in target or target.startswith('#'): continue
            local=target.split('#',1)[0]
            assert (path.parent/local).is_file(),f'Broken local documentation link in {path.name}'


def test_ci_workflow_parses_and_runs_all_tests_without_model_downloads():
    workflow=yaml.safe_load((ROOT/'.github/workflows/test.yml').read_text())
    assert set(workflow['on'])=={'push','pull_request','workflow_dispatch'}
    assert workflow['permissions']=={'contents':'read'}
    steps=workflow['jobs']['test']['steps']
    assert steps[-1]['run']=='python -m pytest -v'
    for step in steps:
        if 'uses' in step: assert re.fullmatch(r'actions/[a-z-]+@[0-9a-f]{40}',step['uses'])
        assert 'ollama pull' not in str(step) and ' -k ' not in str(step)
